#!/usr/bin/env python
"""Convert the collated 3D-merged caiman atlas to per-plane suite2p outputs.

Source:  collated_caiman_output_minSNR_1p4.mat
  C_all   (n_frames, n_total)   denoised traces
  T_all   (n_frames, n_total)   raw traces
  nx, ny  (1, n_total)          per-cell coords in MICRONS (anisotropic grid)
  nz      (1, n_total)          continuous z in microns (no plane index)
  offsets (2, n_planes)         per-plane (x, y) registration shifts in PIXELS

The per-plane caiman_output_plane_N.mat files store acx, acy in plane PIXELS,
so we convert nx, ny -> pixels using (dx_um, dy_um) inferred from plane 1
(default ~4.62, ~5.20 um/px for this dataset; anisotropic).

There are no spatial footprints in the collated file, so each ROI is written
as a single-pixel mask. Summary images (Cn, Ym) are copied from the per-plane
caiman_output_plane_N.mat files. nz is binned into n_planes equal slabs to
assign plane membership.

Memory: loads C_all and T_all into RAM (~7.6 GB total).

Output: <output_dir>/plane{N-1}/{stat,ops,iscell,F,Fneu,spks}.npy
"""

import argparse
import traceback
from pathlib import Path

import h5py
import numpy as np


def _maybe_transpose(arr):
    if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
        if 0.1 < arr.shape[0] / arr.shape[1] < 10:
            return arr.T
    return arr


def load_summary_images(per_plane_dir, plane_num):
    """returns (cn, ym, fov_shape) from caiman_output_plane_{plane_num}.mat."""
    path = per_plane_dir / f"caiman_output_plane_{plane_num}.mat"
    if not path.exists():
        return None, None, None
    cn = ym = None
    with h5py.File(path, "r") as f:
        if "Cn" in f:
            cn = _maybe_transpose(np.asarray(f["Cn"][:]))
        if "Ym" in f:
            ym = _maybe_transpose(np.asarray(f["Ym"][:]))
    fov = cn.shape if cn is not None else (ym.shape if ym is not None else None)
    return cn, ym, fov


def infer_um_per_pixel(per_plane_dir, plane_num, nx_plane, ny_plane):
    """infer (dx_um, dy_um) by matching per-plane acx/acy max to collated nx/ny max."""
    path = per_plane_dir / f"caiman_output_plane_{plane_num}.mat"
    if not path.exists():
        return None, None
    with h5py.File(path, "r") as f:
        acx_max = float(np.asarray(f["acx"][:]).max())
        acy_max = float(np.asarray(f["acy"][:]).max())
    if acx_max <= 0 or acy_max <= 0:
        return None, None
    return float(nx_plane.max()) / acx_max, float(ny_plane.max()) / acy_max


def assign_planes(nz, n_planes=30):
    """bin continuous nz into n_planes equal-width slabs (1..n_planes)."""
    z_max = float(nz.max())
    edges = np.linspace(0.0, z_max, n_planes + 1)
    bins = np.digitize(nz, edges[1:-1]) + 1
    return np.clip(bins, 1, n_planes).astype(np.int32)


def _resolve_offset(x_global, y_global, off_x, off_y, Ly, Lx):
    """try both offset signs; return whichever fits more cells in FOV.

    returns (xs, ys, sign_str, n_in_bounds).
    """
    cand = [
        ("global - offset", x_global - off_x, y_global - off_y),
        ("global + offset", x_global + off_x, y_global + off_y),
    ]
    best = None
    best_n = -1
    for label, xs, ys in cand:
        n = int(((xs >= 0) & (xs < Lx) & (ys >= 0) & (ys < Ly)).sum())
        if n > best_n:
            best = (xs, ys, label, n)
            best_n = n
    return best


def _build_stat(xs, ys):
    """single-pixel stat entries at each (xs, ys)."""
    n = len(xs)
    stat = np.empty(n, dtype=object)
    for i in range(n):
        stat[i] = {
            "ypix": np.array([ys[i]], dtype=np.int32),
            "xpix": np.array([xs[i]], dtype=np.int32),
            "lam": np.ones(1, dtype=np.float32),
            "npix": 1,
            "overlap": np.zeros(1, dtype=bool),
            "med": [float(ys[i]), float(xs[i])],
            "radius": 1.0,
            "aspect_ratio": 1.0,
            "compact": 1.0,
            "footprint": 0.0,
            "skew": 0.0,
            "std": 0.0,
        }
    return stat


def _build_ops(Ly, Lx, n_frames, fs, save_path, cn=None, ym=None, source_file=""):
    ops = {
        "Ly": int(Ly),
        "Lx": int(Lx),
        "nframes": int(n_frames),
        "fs": float(fs),
        "nplanes": 1,
        "nchannels": 1,
        "tau": 1.3,
        "save_path": str(save_path),
        "save_path0": str(save_path.parent),
        "save_folder": save_path.name,
        "do_registration": False,
        "roidetect": False,
        "spikedetect": False,
        "diameter": 4,
        "yrange": [0, int(Ly)],
        "xrange": [0, int(Lx)],
        "source": "caiman_collated_minsnr",
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
    return ops


def convert(collated_path, per_plane_dir, out_root, fs=10.0,
            planes=None, skip_existing=False, n_planes=30,
            dx_um=None, dy_um=None):
    collated_path = Path(collated_path)
    per_plane_dir = Path(per_plane_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"opening {collated_path}")
    with h5py.File(collated_path, "r") as f:
        nx = np.asarray(f["nx"][:]).flatten()
        ny = np.asarray(f["ny"][:]).flatten()
        nz = np.asarray(f["nz"][:]).flatten()
        offsets = np.asarray(f["offsets"][:])  # (2, n_planes)
        n_total = nx.size
        n_frames = int(f["C_all"].shape[0])
        print(f"  {n_total:,} total neurons, {n_frames} frames, {n_planes} planes")
        print(f"  nz range: [{nz.min():.2f}, {nz.max():.2f}] um")

        plane_id = assign_planes(nz, n_planes=n_planes)

        # infer um/pixel from plane 1 if not provided
        if dx_um is None or dy_um is None:
            m1 = plane_id == 1
            inferred = infer_um_per_pixel(per_plane_dir, 1, nx[m1], ny[m1])
            if inferred[0] is not None:
                dx_um = dx_um or inferred[0]
                dy_um = dy_um or inferred[1]
        if dx_um is None or dy_um is None:
            raise RuntimeError("could not infer dx_um/dy_um; pass --dx --dy explicitly")
        print(f"  using pixel size: dx={dx_um:.4f} um/px, dy={dy_um:.4f} um/px")

        print("loading T_all, C_all into memory (~7.6 GB)...")
        T_all = np.asarray(f["T_all"][:])  # (n_frames, n_total)
        C_all = np.asarray(f["C_all"][:])

    for p in range(1, n_planes + 1):
        if planes and p not in planes:
            continue
        out_dir = out_root / f"plane{p - 1}"
        if skip_existing and (out_dir / "stat.npy").exists():
            print(f"[plane {p}] skipping, exists at {out_dir}")
            continue

        mask = (plane_id == p)
        n_p = int(mask.sum())
        if n_p == 0:
            print(f"[plane {p}] no cells in z-bin, skipping")
            continue

        try:
            cn, ym, fov = load_summary_images(per_plane_dir, p)
            if fov is None:
                print(f"[plane {p}] no per-plane mat for summary images, skipping")
                continue
            Ly, Lx = int(fov[0]), int(fov[1])

            # microns -> per-plane pixels
            x_pix = nx[mask] / dx_um
            y_pix = ny[mask] / dy_um
            ox = float(offsets[0, p - 1])
            oy = float(offsets[1, p - 1])
            xs_f, ys_f, sign_label, in_bounds = _resolve_offset(
                x_pix, y_pix, ox, oy, Ly, Lx
            )
            xs = np.clip(np.rint(xs_f).astype(np.int32), 0, Lx - 1)
            ys = np.clip(np.rint(ys_f).astype(np.int32), 0, Ly - 1)
            clipped = n_p - in_bounds
            print(f"[plane {p}] n={n_p:,}  fov={Ly}x{Lx}  "
                  f"offset_px=({ox:.1f},{oy:.1f})  {sign_label}  "
                  f"in_bounds={in_bounds:,}  clipped={clipped}")

            cols = np.where(mask)[0]
            F = T_all[:, cols].T.astype(np.float32)
            spks = C_all[:, cols].T.astype(np.float32)
            Fneu = np.zeros_like(F)

            stat = _build_stat(xs, ys)
            iscell = np.ones((n_p, 2), dtype=np.float32)
            ops = _build_ops(
                Ly, Lx, n_frames, fs, out_dir,
                cn=cn, ym=ym,
                source_file=f"{collated_path}::plane{p}",
            )

            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "stat.npy", stat)
            np.save(out_dir / "ops.npy", ops)
            np.save(out_dir / "iscell.npy", iscell)
            np.save(out_dir / "F.npy", F)
            np.save(out_dir / "Fneu.npy", Fneu)
            np.save(out_dir / "spks.npy", spks)
            print(f"  wrote -> {out_dir}")
        except Exception:
            print(f"[plane {p}] FAILED")
            traceback.print_exc()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "input_file",
        nargs="?",
        default=r"X:\lbm\jdemas_2021-lbm-paper\bi_hemisphere\output\collated_caiman_output_minSNR_1p4.mat",
    )
    p.add_argument(
        "output_dir",
        nargs="?",
        default=r"D:\jeff\bi_hemi_min_snr",
    )
    p.add_argument(
        "--per-plane-dir",
        default=r"X:\lbm\jdemas_2021-lbm-paper\bi_hemisphere\output",
        help="directory holding caiman_output_plane_*.mat (for Cn/Ym summary images)",
    )
    p.add_argument("--fs", type=float, default=10.0)
    p.add_argument("--n-planes", type=int, default=30)
    p.add_argument("--planes", type=int, nargs="*", default=None,
                   help="subset of plane numbers to convert (1-based)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dx", type=float, default=None,
                   help="microns per pixel in x (auto-inferred from plane 1 if omitted)")
    p.add_argument("--dy", type=float, default=None,
                   help="microns per pixel in y (auto-inferred from plane 1 if omitted)")
    args = p.parse_args()

    convert(
        args.input_file, args.per_plane_dir, args.output_dir,
        fs=args.fs, planes=args.planes,
        skip_existing=args.skip_existing, n_planes=args.n_planes,
        dx_um=args.dx, dy_um=args.dy,
    )


if __name__ == "__main__":
    main()
