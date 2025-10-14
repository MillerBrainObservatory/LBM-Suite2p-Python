from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from lbm_suite2p_python import plot_zplane_figures
from mbo_utilities.lazy_array import Suite2pArray


def group_plane_rois(input_dir):
    input_dir = Path(input_dir)
    grouped = defaultdict(list)

    for d in input_dir.iterdir():
        if (
                d.is_dir()
                and not d.name.endswith(".zarr")     # exclude zarr dirs
                and d.stem.startswith("plane")
                and "_roi" in d.stem
        ):
            parts = d.stem.split("_")
            if len(parts) == 2 and parts[1].startswith("roi"):
                plane = parts[0]  # e.g. "plane01"
                grouped[plane].append(d)

    return grouped


def _merge_images(
        ops_list,
        keys_full=("refImg", "meanImg", "meanImgE"),
        keys_cropped=("max_proj", "Vcorr"),
):
    merged = {}

    # --- determine global dimensions
    Ly = max(ops["Ly"] for ops in ops_list)
    widths = [ops["Lx"] for ops in ops_list]
    total_Lx = sum(widths)

    # Full-FOV keys: tile horizontally
    for key in keys_full:
        if all(key in ops for ops in ops_list):
            canvas = np.zeros((Ly, total_Lx), dtype=ops_list[0][key].dtype)
            xoff = 0
            for opsd in ops_list:
                arr = opsd[key]
                arr_h, arr_w = arr.shape
                slot_w = opsd["Lx"]

                # crop/pad vertically to Ly
                crop_h = min(arr_h, Ly)
                slot_h = Ly
                tmp = np.zeros((slot_h, slot_w), dtype=arr.dtype)
                tmp[:crop_h, :min(arr_w, slot_w)] = arr[:crop_h, :min(arr_w, slot_w)]

                # insert into canvas
                canvas[:slot_h, xoff:xoff + slot_w] = tmp
                xoff += slot_w

            merged[key] = canvas

    # Cropped keys: place at yrange/xrange
    for key in keys_cropped:
        if all(key in ops for ops in ops_list):
            canvas = np.zeros((Ly, total_Lx), dtype=ops_list[0][key].dtype)
            xoff = 0
            for opsd in ops_list:
                arr = opsd[key]
                arr_h, arr_w = arr.shape
                yr = np.array(opsd.get("yrange", [0, Ly]))
                xr = np.array(opsd.get("xrange", [0, opsd["Lx"]])) + xoff

                slot_h = yr[1] - yr[0]
                slot_w = xr[1] - xr[0]

                tmp = np.zeros((slot_h, slot_w), dtype=arr.dtype)
                tmp[:min(arr_h, slot_h), :min(arr_w, slot_w)] = arr[:min(arr_h, slot_h), :min(arr_w, slot_w)]

                canvas[yr[0]:yr[0]+slot_h, xr[0]:xr[0]+slot_w] = tmp
                xoff += opsd["Lx"]

            merged[key] = canvas

    return merged

def merge_mrois(input_dir, output_dir, overwrite=True):
    """
    Merge Suite2p outputs from multiple ROIs into per-plane outputs.
    Will attempt to merge everything available; skips missing files gracefully.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    grouped = group_plane_rois(input_dir)

    for plane, dirs in tqdm(sorted(grouped.items()), desc="Merging mROIs", unit="plane"):
        out_dir = output_dir / plane
        out_ops = out_dir / "ops.npy"

        if out_ops.exists() and not overwrite:
            print(f"Skipping {plane}, merged outputs already exist")
            continue

        out_dir.mkdir(exist_ok=True)

        # --- load per-ROI results
        ops_list, stat_list, iscell_list = [], [], []
        F_list, Fneu_list, spks_list = [], [], []
        bin_paths = []
        for d in sorted(dirs):
            ops_path = d / "ops.npy"
            if not ops_path.exists():
                print(f"Skipping {d}, no ops.npy")
                continue
            ops = np.load(ops_path, allow_pickle=True).item()
            ops_list.append(ops)

            if (d / "stat.npy").exists():
                stat_list.append(np.load(d / "stat.npy", allow_pickle=True))
            if (d / "iscell.npy").exists():
                iscell_list.append(np.load(d / "iscell.npy", allow_pickle=True))
            if (d / "F.npy").exists():
                F_list.append(np.load(d / "F.npy"))
            if (d / "Fneu.npy").exists():
                Fneu_list.append(np.load(d / "Fneu.npy"))
            if (d / "spks.npy").exists():
                spks_list.append(np.load(d / "spks.npy"))

            if (d / "data_raw.bin").exists():
                bin_paths.append(d / "data_raw.bin")
            elif (d / "data.bin").exists():
                bin_paths.append(d / "data.bin")

        if not ops_list:
            print(f"No valid ROIs found for {plane}, skipping merge")
            continue

        # --- dimensions
        Ly = ops_list[0]["Ly"]
        widths = [ops.get("xrange", [0, ops["Lx"]])[1] -
                  ops.get("xrange", [0, ops["Lx"]])[0] for ops in ops_list]
        total_Lx = int(sum(widths))

        # --- merge stat + traces
        stat = None
        if stat_list:
            for i, st in enumerate(stat_list):
                xoff = int(sum(widths[:i]))
                for s in st:
                    s["xpix"] = np.asarray(s["xpix"], int) + xoff
                    s["ypix"] = np.asarray(s["ypix"], int)
                    s["med"] = [float(s["med"][0]), float(s["med"][1]) + xoff]
                    if "lam" in s:
                        s["lam"] = np.asarray(s["lam"], float).ravel()
                    if "ipix_neuropil" in s:
                        ypix, xpix = s["ypix"], s["xpix"]
                        s["ipix_neuropil"] = ypix + xpix * Ly
            stat = np.concatenate(stat_list)

        iscell = np.concatenate(iscell_list, 0) if iscell_list else None
        F = np.concatenate(F_list, 0) if F_list else None
        Fneu = np.concatenate(Fneu_list, 0) if Fneu_list else None
        spks = np.concatenate(spks_list, 0) if spks_list else None

        # --- merge binary
        merged_bin = out_dir / "data.bin"
        if bin_paths:
            arrays = [Suite2pArray(p) for p in bin_paths]
            nframes = min(arr.nframes for arr in arrays)
            dtype = arrays[0].dtype
            with open(merged_bin, "wb") as f:
                for i in range(nframes):
                    frames = [arr[i] for arr in arrays]
                    f.write(np.hstack(frames).astype(dtype).tobytes())
            for arr in arrays:
                arr.close()
        else:
            merged_bin = None

        # --- merged ops header
        merged_ops = dict(ops_list[0])
        merged_ops.update({
            "Ly": Ly,
            "Lx": total_Lx,
            "yrange": [0, Ly],
            "xrange": [0, total_Lx],
            "ops_path": str(out_ops.resolve()),
            "save_path": str(out_dir.resolve()),
            "nrois": len(ops_list),
        })
        if merged_bin:
            merged_ops["reg_file"] = str(merged_bin.resolve())

        # >>> THIS IS THE ONLY PLACE YOU MERGE IMAGES <<<
        merged_ops.update(_merge_images(ops_list))

        for key in ["yoff", "xoff", "corrXY", "badframes"]:
            arrays = [ops[key] for ops in ops_list if key in ops]
            if arrays and all(np.array_equal(a, arrays[0]) for a in arrays[1:]):
                merged_ops[key] = arrays[0]

        np.save(out_ops, merged_ops)
        if stat is not None: np.save(out_dir / "stat.npy", stat)
        if iscell is not None: np.save(out_dir / "iscell.npy", iscell)
        if F is not None: np.save(out_dir / "F.npy", F)
        if Fneu is not None: np.save(out_dir / "Fneu.npy", Fneu)
        if spks is not None: np.save(out_dir / "spks.npy", spks)

        try:
            plot_zplane_figures(out_dir, run_rastermap=False)
        except Exception:
            pass


def merge_zarr_rois(input_dir, output_dir=None, overwrite=True):
    """
    Concatenate roi1 + roi2 .zarr stores for each plane into a single planeXX.zarr.

    Parameters
    ----------
    input_dir : Path or str
        Directory containing planeXX_roi1, planeXX_roi2 subfolders with ops.npy + data.zarr.
    output_dir : Path or str, optional
        Where to write merged planeXX.zarr. Defaults to `input_dir`.
    overwrite : bool
        If True, existing outputs are replaced.
    """
    import dask.array as da

    z_merged = None
    input_dir = Path(input_dir)
    output_dir = (
        Path(output_dir)
        if output_dir
        else input_dir.parent / (input_dir.name + "_merged")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    roi1_dirs = sorted(input_dir.glob("*plane*_roi1*"))
    roi2_dirs = sorted(input_dir.glob("*plane*_roi2*"))
    if not roi1_dirs or not roi2_dirs:
        print("No roi1 or roi2 in input dir")
        return None
    assert len(roi1_dirs) == len(roi2_dirs), "Mismatched ROI dirs"

    for roi1, roi2 in zip(roi1_dirs, roi2_dirs):
        zplane = roi1.stem.split("_")[0]  # "plane01"
        out_path = output_dir / f"{zplane}.zarr"
        if out_path.exists():
            if overwrite:
                import shutil

                shutil.rmtree(out_path)
            else:
                print(f"Skipping {zplane}, {out_path} exists")
                continue

        # load ops
        z1 = da.from_zarr(roi1)
        z2 = da.from_zarr(roi2)

        # sanity check
        assert z1.shape[0] == z2.shape[0], "Frame count mismatch"
        assert z1.shape[1] == z2.shape[1], "Height mismatch"

        # concatenate along width (axis=2)
        z_merged = da.concatenate([z1, z2], axis=2)
        z_merged.to_zarr(out_path, overwrite=overwrite)

    if z_merged:
        print(f"{z_merged}")

    return None


if __name__ == "__main__":
    fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw\anatomical_3_roi")
    merge_mrois(fpath, fpath.parent / "anatomical_3_merged")
    x = 2
