#!/usr/bin/env python
"""Convert CaImAn-MATLAB plane outputs into per-plane Suite2p outputs.

Special-case script for the bi_hemisphere dataset:
  X:/lbm/jdemas_2021-lbm-paper/bi_hemisphere/output/caiman_output_plane_*.mat

Each caiman_output_plane_N.mat -> output_dir/plane{N-1}/{stat,ops,iscell,F,Fneu,spks}.npy

Mapping:
  Ac_keep (n, 9, 9) + acx, acy  -> stat (ypix, xpix, lam reconstructed in full FOV)
  T_keep  (frames, n)           -> F.npy   (n, frames)
  C_keep  (frames, n)           -> spks.npy (denoised temporal; closest stand-in)
  Cn / Ym                       -> ops max_proj / meanImg
  rVals                         -> iscell column 1 (confidence)
  Fneu                          -> zeros (caiman background is global b*f, not per-roi)
"""

import argparse
import traceback
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import convolve1d
from scipy.stats import norm


def _hsm(data):
    """half-sample mode (single sorted 1D array). matches mode_robust in compute_event_exceptionality.m."""
    n = len(data)
    if n == 1:
        return float(data[0])
    if n == 2:
        return float(data.mean())
    if n == 3:
        i1 = data[1] - data[0]
        i2 = data[2] - data[1]
        if i1 < i2:
            return float(data[:2].mean())
        elif i2 > i1:
            return float(data[1:].mean())
        return float(data[1])
    # matlab: N = floor(n/2) + mod(n,2) - 1
    N = n // 2 + (n % 2) - 1
    diffs = data[N - 1:N - 1 + N] - data[:N]
    j = int(np.argmin(diffs))
    return _hsm(data[j:j + N + 1])


def mode_robust(traces):
    """row-wise half-sample mode of (n_rois, n_frames)."""
    sorted_traces = np.sort(traces, axis=1)
    out = np.empty(traces.shape[0], dtype=np.float64)
    for i in range(traces.shape[0]):
        out[i] = _hsm(sorted_traces[i])
    return out


def compute_event_exceptionality(traces, N=2):
    """port of caiman compute_event_exceptionality.m (robust_std=False)."""
    traces = traces.astype(np.float64, copy=False)
    md = np.maximum(mode_robust(traces), 0.0).reshape(-1, 1)

    ff1 = traces - md
    ff1 = -ff1 * (ff1 < 0)
    Ns = (ff1 > 0).sum(axis=1, keepdims=True).astype(np.float64)
    Ns = np.maximum(Ns, 1.0)
    sd_r = np.sqrt((ff1 ** 2).sum(axis=1, keepdims=True) / Ns)
    sd_r = np.maximum(sd_r, 1e-12)

    z = (traces - md) / (3.0 * sd_r)
    # 1 - cdf(N(0,1), z) == norm.sf(z); clip to keep log finite
    erf = np.clip(norm.sf(z), 1e-300, 1.0)
    log_erf = np.log(erf)
    if N > 1:
        log_erf = convolve1d(log_erf, np.ones(N), axis=1, mode="constant", cval=0.0)
    return log_erf.min(axis=1)


def fitness_to_snr(fitness):
    """matlab: snre = -norminv(exp(ftns./2))."""
    p = np.clip(np.exp(fitness / 2.0), 1e-300, 1.0 - 1e-15)
    return -norm.ppf(p)


def compute_snr(traces):
    """SNR per ROI matching the minSNR_1p4 threshold convention."""
    return fitness_to_snr(compute_event_exceptionality(traces, N=2))


def load_caiman_plane(filepath):
    """load fields from a caiman matlab v7.3 output file (mirrors the notebook)."""
    data = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            try:
                arr = f[key][:]
                # transpose 2d image arrays (matlab is column-major)
                if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
                    if 0.1 < arr.shape[0] / arr.shape[1] < 10:
                        arr = arr.T
                data[key] = arr
            except Exception as e:
                print(f"  could not load {key}: {e}")
    return data


def load_caiman_plane_v5(filepath):
    """load a v5/v7 .mat (high_resolution dataset format) via scipy.io.

    fields:
      A_keep (n_pixels, n_neurons) sparse  : full-fov spatial footprints
      C_keep, T_keep (n_neurons, n_frames)
      Cn, Ym (Ly, Lx)
      rVals (n_neurons, 1)
    """
    import scipy.io
    raw = scipy.io.loadmat(filepath)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def load_caiman_plane_auto(filepath):
    """try v7.3 (h5py) first, fall back to v5 (scipy.io)."""
    try:
        return load_caiman_plane(filepath), "v7.3"
    except OSError:
        return load_caiman_plane_v5(filepath), "v5"


def full_fov_to_stat(A_keep, fov_shape):
    """build suite2p stat from a sparse full-fov footprint matrix.

    A_keep: scipy.sparse.csc_matrix (n_pixels, n_neurons), matlab column-major flat.
    fov_shape: (Ly, Lx)
    """
    import scipy.sparse
    Ly, Lx = fov_shape
    if not scipy.sparse.issparse(A_keep):
        A_keep = scipy.sparse.csc_matrix(A_keep)
    A = A_keep.tocsc()
    if A.shape[0] != Ly * Lx:
        raise ValueError(
            f"A_keep n_pixels={A.shape[0]} != Ly*Lx={Ly*Lx} for fov {fov_shape}"
        )

    indptr = A.indptr
    indices = A.indices
    values = A.data
    n_neurons = A.shape[1]

    stat = []
    for j in range(n_neurons):
        s, e = indptr[j], indptr[j + 1]
        flat_idx = indices[s:e]
        weights = values[s:e].astype(np.float32)
        ys, xs = np.unravel_index(flat_idx, (Ly, Lx), order="F")
        npix = int(len(flat_idx))

        if npix == 0:
            stat.append({
                "ypix": np.array([0], dtype=np.int32),
                "xpix": np.array([0], dtype=np.int32),
                "lam": np.array([1.0], dtype=np.float32),
                "npix": 1,
                "overlap": np.zeros(1, dtype=bool),
                "med": [0.0, 0.0],
                "radius": 1.0,
                "aspect_ratio": 1.0,
                "compact": 1.0,
                "footprint": 0.0,
                "skew": 0.0,
                "std": 0.0,
            })
            continue

        keep = weights > 0
        if not keep.all():
            ys = ys[keep]
            xs = xs[keep]
            weights = weights[keep]
            npix = int(len(ys))
            if npix == 0:
                stat.append({
                    "ypix": np.array([0], dtype=np.int32),
                    "xpix": np.array([0], dtype=np.int32),
                    "lam": np.array([1.0], dtype=np.float32),
                    "npix": 1,
                    "overlap": np.zeros(1, dtype=bool),
                    "med": [0.0, 0.0],
                    "radius": 1.0,
                    "aspect_ratio": 1.0,
                    "compact": 1.0,
                    "footprint": 0.0,
                    "skew": 0.0,
                    "std": 0.0,
                })
                continue

        w_sum = float(weights.sum())
        lam = (weights / w_sum) if w_sum > 0 else np.full(npix, 1.0 / npix, dtype=np.float32)

        med_y = float(np.median(ys))
        med_x = float(np.median(xs))
        y_range = int(ys.max() - ys.min() + 1)
        x_range = int(xs.max() - xs.min() + 1)
        aspect = max(y_range, x_range) / max(1, min(y_range, x_range))
        radius = float(np.sqrt(npix / np.pi))

        stat.append({
            "ypix": ys.astype(np.int32),
            "xpix": xs.astype(np.int32),
            "lam": lam.astype(np.float32),
            "npix": npix,
            "overlap": np.zeros(npix, dtype=bool),
            "med": [med_y, med_x],
            "radius": radius,
            "aspect_ratio": float(aspect),
            "compact": float(npix / (np.pi * radius**2)) if radius > 0 else 1.0,
            "footprint": 0.0,
            "skew": 0.0,
            "std": 0.0,
        })

    return np.array(stat, dtype=object)


def caiman_to_stat(ac_keep, acx, acy, fov_shape):
    """build a suite2p stat array from cropped footprints and centroids."""
    n_neurons, fp_h, fp_w = ac_keep.shape
    half_y = fp_h // 2
    half_x = fp_w // 2
    Ly, Lx = fov_shape

    stat = []
    for i in range(n_neurons):
        fp = ac_keep[i]
        cx, cy = int(acx[i]), int(acy[i])

        local_y, local_x = np.where(fp > 0)
        gx = local_x + cx - half_x
        gy = local_y + cy - half_y
        valid = (gx >= 0) & (gx < Lx) & (gy >= 0) & (gy < Ly)
        gx = gx[valid]
        gy = gy[valid]
        weights = fp[local_y[valid], local_x[valid]].astype(np.float32)
        npix = int(len(gx))

        if npix == 0:
            # degenerate roi: keep a single-pixel placeholder at the centroid
            cy_c = int(np.clip(cy, 0, Ly - 1))
            cx_c = int(np.clip(cx, 0, Lx - 1))
            stat.append({
                "ypix": np.array([cy_c], dtype=np.int32),
                "xpix": np.array([cx_c], dtype=np.int32),
                "lam": np.array([1.0], dtype=np.float32),
                "npix": 1,
                "overlap": np.zeros(1, dtype=bool),
                "med": [float(cy_c), float(cx_c)],
                "radius": 1.0,
                "aspect_ratio": 1.0,
                "compact": 1.0,
                "footprint": 0.0,
                "skew": 0.0,
                "std": 0.0,
            })
            continue

        w_sum = float(weights.sum())
        lam = (weights / w_sum) if w_sum > 0 else np.full(npix, 1.0 / npix, dtype=np.float32)

        y_range = int(gy.max() - gy.min() + 1)
        x_range = int(gx.max() - gx.min() + 1)
        aspect = max(y_range, x_range) / max(1, min(y_range, x_range))
        radius = float(np.sqrt(npix / np.pi))

        stat.append({
            "ypix": gy.astype(np.int32),
            "xpix": gx.astype(np.int32),
            "lam": lam.astype(np.float32),
            "npix": npix,
            "overlap": np.zeros(npix, dtype=bool),
            "med": [float(cy), float(cx)],
            "radius": radius,
            "aspect_ratio": float(aspect),
            "compact": float(npix / (np.pi * radius**2)) if radius > 0 else 1.0,
            "footprint": 0.0,
            "skew": 0.0,
            "std": 0.0,
        })

    return np.array(stat, dtype=object)


def _orient_traces(arr, n_neurons):
    """ensure trace array is (n_rois, n_frames)."""
    if arr.shape[0] == n_neurons:
        return arr.astype(np.float32)
    if arr.shape[1] == n_neurons:
        return arr.T.astype(np.float32)
    raise ValueError(f"trace shape {arr.shape} does not match n_neurons={n_neurons}")


def write_plane_outputs(data, out_dir, fs=10.0, source_label="caiman_matlab",
                        source_file="", snr_min=None):
    """write suite2p plane outputs from a per-plane caiman field dict.

    handles two source formats:
      bi_hemisphere (v7.3): cropped Ac_keep (n,9,9) + acx, acy
      high_resolution (v5): sparse A_keep (n_pixels, n) full-fov

    if snr_min is set, computes SNR (matlab compute_event_exceptionality + norminv)
    on T_keep and drops cells below the threshold before writing.
    """
    cn = data.get("Cn")
    ym = data.get("Ym")
    if cn is not None:
        fov_shape = cn.shape
    elif ym is not None:
        fov_shape = ym.shape
    else:
        raise ValueError("no Cn or Ym available for FOV shape")
    Ly, Lx = int(fov_shape[0]), int(fov_shape[1])

    is_full_fov = "A_keep" in data and "Ac_keep" not in data
    if is_full_fov:
        A = data["A_keep"]
        n_neurons = int(A.shape[1])
    else:
        ac = data["Ac_keep"]
        n_neurons = int(ac.shape[0])
        acx = np.asarray(data["acx"]).flatten()
        acy = np.asarray(data["acy"]).flatten()

    F = _orient_traces(data["T_keep"], n_neurons)
    n_frames = int(F.shape[1])
    spks = _orient_traces(data["C_keep"], n_neurons)

    rvals = None
    if "rVals" in data:
        rv = np.asarray(data["rVals"]).flatten()
        if rv.size == n_neurons:
            rvals = rv

    if snr_min is not None:
        print(f"  computing SNR for {n_neurons:,} rois (caiman event exceptionality, N=2)...")
        snr = compute_snr(F)
        keep = snr >= float(snr_min)
        kept = int(keep.sum())
        print(f"  SNR>={snr_min}: kept {kept:,} / {n_neurons:,} "
              f"({100.0 * kept / n_neurons:.1f}%)")
        if kept == 0:
            raise RuntimeError("no rois passed SNR threshold")
        F = F[keep]
        spks = spks[keep]
        if is_full_fov:
            A = A[:, keep]
        else:
            ac = ac[keep]
            acx = acx[keep]
            acy = acy[keep]
        if rvals is not None:
            rvals = rvals[keep]
        n_neurons = kept

    Fneu = np.zeros_like(F)

    print(f"  rois={n_neurons:,} frames={n_frames} fov={Ly}x{Lx}")
    if is_full_fov:
        stat = full_fov_to_stat(A, (Ly, Lx))
    else:
        stat = caiman_to_stat(ac, acx, acy, (Ly, Lx))

    iscell = np.ones((n_neurons, 2), dtype=np.float32)
    if rvals is not None:
        iscell[:, 1] = rvals.astype(np.float32)

    ops = {
        "Ly": Ly,
        "Lx": Lx,
        "nframes": n_frames,
        "fs": float(fs),
        "nplanes": 1,
        "nchannels": 1,
        "tau": 1.3,
        "save_path": str(out_dir),
        "save_path0": str(out_dir.parent),
        "save_folder": out_dir.name,
        "do_registration": False,
        "roidetect": False,
        "spikedetect": False,
        "diameter": 4,
        "yrange": [0, Ly],
        "xrange": [0, Lx],
        "source": source_label,
        "source_file": str(source_file),
    }
    if cn is not None:
        cn32 = cn.astype(np.float32)
        ops["Vcorr"] = cn32
        ops["max_proj"] = cn32
    if ym is not None:
        ym32 = ym.astype(np.float32)
        ops["meanImg"] = ym32
        ops["meanImgE"] = ym32
        if cn is None:
            ops["max_proj"] = ym32

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "stat.npy", stat)
    np.save(out_dir / "ops.npy", ops)
    np.save(out_dir / "iscell.npy", iscell)
    np.save(out_dir / "F.npy", F)
    np.save(out_dir / "Fneu.npy", Fneu)
    np.save(out_dir / "spks.npy", spks)
    print(f"  wrote -> {out_dir}")


def convert_plane(mat_path, out_dir, fs=10.0, snr_min=None):
    """convert one caiman_output_plane_N.mat into a suite2p plane directory."""
    print(f"[{mat_path.name}] loading...")
    data, fmt = load_caiman_plane_auto(mat_path)
    write_plane_outputs(
        data, out_dir, fs=fs,
        source_label=f"caiman_matlab_{fmt}", source_file=mat_path,
        snr_min=snr_min,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "input_dir",
        nargs="?",
        default=r"X:\lbm\jdemas_2021-lbm-paper\bi_hemisphere\output",
        help="directory containing caiman_output_plane_*.mat",
    )
    p.add_argument(
        "output_dir",
        nargs="?",
        default=r"X:\lbm\jdemas_2021-lbm-paper\bi_hemisphere\output\suite2p",
        help="root output directory (each plane goes into a planeN subfolder)",
    )
    p.add_argument("--fs", type=float, default=10.0, help="sampling rate per plane")
    p.add_argument(
        "--planes", type=int, nargs="*", default=None,
        help="subset of plane numbers to convert (1-based, matches filename)",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="skip planes whose output stat.npy already exists",
    )
    p.add_argument(
        "--snr-min", type=float, default=None,
        help="filter rois by SNR (matches the collated minSNR_1p4 convention; e.g. 1.4)",
    )
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)

    files = sorted(
        in_dir.glob("caiman_output_plane_*.mat"),
        key=lambda f: int(f.stem.rsplit("_", 1)[-1]),
    )
    if args.planes:
        wanted = set(args.planes)
        files = [f for f in files if int(f.stem.rsplit("_", 1)[-1]) in wanted]

    if not files:
        print(f"no caiman_output_plane_*.mat found in {in_dir}")
        return

    print(f"converting {len(files)} planes")
    print(f"  input:  {in_dir}")
    print(f"  output: {out_root}\n")

    for f in files:
        plane_num = int(f.stem.rsplit("_", 1)[-1])
        out_dir = out_root / f"plane{plane_num - 1}"
        if args.skip_existing and (out_dir / "stat.npy").exists():
            print(f"[{f.name}] skipping, exists at {out_dir}")
            continue
        try:
            convert_plane(f, out_dir, fs=args.fs, snr_min=args.snr_min)
        except Exception:
            print(f"[{f.name}] FAILED")
            traceback.print_exc()


if __name__ == "__main__":
    main()
