import time
from pathlib import Path
import numpy as np

from lbm_suite2p_python.utils import normalize_traces, dff_shot_noise, dff_rolling_percentile, load_planar_results, \
    load_ops
from lbm_suite2p_python.zplane import (
    plot_noise_distribution,
    plot_rastermap,
    plot_masks,
    plot_traces,
    plot_projection,
)
from mbo_utilities.lazy_array import Suite2pArray


def embed_into_canvas(img, yrange, xrange, canvas_shape):
    """
    Crop an image by its yrange and xrange.
    """
    full = np.zeros(canvas_shape, dtype=img.dtype)
    y0, y1 = yrange
    x0, x1 = xrange
    target_shape = (y1 - y0, x1 - x0)
    img_cropped = img[:target_shape[0], :target_shape[1]]
    full[y0:y0 + img_cropped.shape[0], x0:x0 + img_cropped.shape[1]] = img_cropped
    return full

def concat_binfiles_and_merge_metadata(f1, f2, output_bin, output_ops):
    left = Suite2pArray(f1)
    right = Suite2pArray(f2)
    md_left = load_ops(f1)
    md_right = load_ops(f2)

    assert left.Ly == right.Ly, f"Ly mismatch: {left.Ly} vs {right.Ly}"
    assert left.nframes == right.nframes, f"nframes mismatch: {left.nframes} vs {right.nframes}"

    Ly = left.Ly
    Lx = left.Lx + right.Lx
    nframes = left.nframes
    dtype = left.dtype

    output_bin = Path(output_bin)
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    with open(output_bin, "wb") as f_out:
        for i in range(nframes):
            frame = np.hstack([left[i], right[i]])
            f_out.write(frame.astype(dtype).tobytes())  # noqa

    left.close()
    right.close()

    canvas_Ly = max(md_left["yrange"][1], md_right["yrange"][1])
    canvas_Lx = max(md_left["xrange"][1] + md_left["xrange"][0],
                    md_right["xrange"][1] + md_right["xrange"][0])

    merged_md = dict(md_left)
    merged_md["Ly"] = Ly
    merged_md["Lx"] = Lx
    merged_md["yrange"] = [0, Ly]
    merged_md["xrange"] = [0, Lx]
    merged_md["raw_file"] = str(output_bin.resolve())

    for key in ["meanImg", "meanImgE", "Vcorr"]:
        if key in md_left and key in md_right:
            canvas_shape = (canvas_Ly, canvas_Lx)
            left_full = embed_into_canvas(md_left[key], md_left["yrange"], md_left["xrange"], canvas_shape)
            right_full = embed_into_canvas(md_right[key], md_right["yrange"], md_right["xrange"], canvas_shape)
            merged_md[key] = np.hstack([left_full, right_full])

    output_ops = Path(output_ops)
    output_ops.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_ops, merged_md)

    print(f"Saved:\n  bin: {output_bin}\n  ops: {output_ops}")


def merge_rois_for_planes(input_dir, output_dir, overwrite=True):
    """
    Merge Suite2p outputs from roi1 + roi2 into per-plane outputs.

    Parameters
    ----------
    input_dir : Path or str
        Directory with subfolders like plane01_roi1, plane01_roi2, etc.
    output_dir : Path or str
        Directory where merged outputs will be saved (plane01, plane02, ...).
    overwrite : bool
        If True, existing merged outputs are overwritten.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    roi1_dirs = sorted(input_dir.glob("plane*_roi1"))
    roi2_dirs = sorted(input_dir.glob("plane*_roi2"))
    assert len(roi1_dirs) == len(roi2_dirs), "Mismatched ROI dirs"

    for roi1, roi2 in zip(roi1_dirs, roi2_dirs):
        zplane = roi1.stem.split("_")[0]
        out_dir = output_dir / zplane
        out_ops = out_dir / "ops.npy"

        # --- Skip if already merged
        if out_ops.exists() and not overwrite:
            print(f"Skipping {zplane}, merged outputs already exist")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Load both ROI results
        res1 = load_planar_results(roi1)
        res2 = load_planar_results(roi2)

        ops1 = load_ops(Path(roi1) / "ops.npy")
        ops2 = load_ops(Path(roi2) / "ops.npy")

        # --- Merge traces
        F = np.vstack([res1["F"], res2["F"]])
        Fneu = np.vstack([res1["Fneu"], res2["Fneu"]])
        spks = np.vstack([res1["spks"], res2["spks"]])

        stat = list(res1["stat"]) + list(res2["stat"])
        stat = np.array(stat, dtype=object)
        np.save(out_dir / "stat.npy", stat)
        # stat = np.concatenate([res1["stat"], res2["stat"]], axis=0)
        iscell = np.concatenate([res1["iscell"], res2["iscell"]], axis=0)
        cellprob = np.concatenate([res1["cellprob"], res2["cellprob"]], axis=0)

        # save merged arrays
        np.save(out_dir / "F.npy", F)
        np.save(out_dir / "Fneu.npy", Fneu)
        np.save(out_dir / "spks.npy", spks)
        np.save(out_dir / "stat.npy", stat)
        np.save(out_dir / "iscell.npy", np.c_[iscell, cellprob])

        # --- Merge ops
        merged_ops = dict(ops1)
        merged_ops["stat"] = str(out_dir / "stat.npy")
        merged_ops["Ly"] = max(ops1["Ly"], ops2["Ly"])
        merged_ops["Lx"] = ops1["Lx"] + ops2["Lx"]
        merged_ops["yrange"] = [0, merged_ops["Ly"]]
        merged_ops["xrange"] = [0, merged_ops["Lx"]]
        merged_ops["nrois"] = 2
        merged_ops["save_path"] = str(out_dir.resolve())

        # --- Merge binary if available
        bin1 = Path(ops1.get("raw_file", "")).with_suffix(".bin")
        bin2 = Path(ops2.get("raw_file", "")).with_suffix(".bin")
        if bin1.exists() and bin2.exists():
            merged_bin = out_dir / "data_raw.bin"
            concat_binfiles_and_merge_metadata(
                ops1, ops2, merged_bin, out_ops
            )
            merged_ops["raw_file"] = str(merged_bin.resolve())
        else:
            np.save(out_ops, merged_ops)


        print(f"Merged {roi1.name} + {roi2.name} → {out_dir}")

def remake_plane_figures(plane_dir, dff_percentile=8, dff_window_size=101, run_rastermap=False, **kwargs):
    """
    Re-generate Suite2p diagnostic figures for a merged plane.

    Parameters
    ----------
    plane_dir : Path
        Path to the planeXX output directory (with ops.npy, stat.npy, etc.).
    dff_percentile : int, optional
        Percentile used for ΔF/F baseline. Default is 8.
    dff_window_size : int, optional
        Window size for ΔF/F rolling baseline. Default is 101.
    run_rastermap : bool, optional
        If True, re-run rastermap sorting (may be slow for many neurons). Default is False.
    kwargs : dict
        Extra keyword args (e.g. fig_label).
    """
    plane_dir = Path(plane_dir)

    expected_files = {
        "ops": plane_dir / "ops.npy",
        "stat": plane_dir / "stat.npy",
        "iscell": plane_dir / "iscell.npy",
        "registration": plane_dir / "registration.png",
        "segmentation": plane_dir / "segmentation.png",
        "max_proj": plane_dir / "max_projection_image.png",
        "meanImg": plane_dir / "mean_image.png",
        "meanImgE": plane_dir / "mean_image_enhanced.png",
        "traces_raw": plane_dir / "traces_raw.png",
        "traces_dff": plane_dir / "traces_dff.png",
        "traces_noise": plane_dir / "traces_noise.png",
        "noise": plane_dir / "shot_noise_distrubution.png",
        "model": plane_dir / "model.npy",
        "rastermap": plane_dir / "rastermap.png",
    }

    output_ops = load_ops(expected_files["ops"])
    def safe_delete(file_path):
        if file_path.exists():
            try:
                file_path.unlink()
            except PermissionError:
                print(f"Error: Cannot delete {file_path}, it's open elsewhere.")

    # force remake of the heavy figures
    for key in ["registration", "segmentation", "traces_raw", "traces_dff"]:
        safe_delete(expected_files[key])

    if expected_files["stat"].is_file():

        res = load_planar_results(plane_dir)
        iscell = res["iscell"]
        spks = res["spks"][iscell]

        if iscell.ndim == 2:
            iscell = iscell[:, 0]

        f_norm = normalize_traces(res["F"][iscell])
        # F = F - F.min(axis=1, keepdims=True) * 0.9

        n_neurons = f_norm.shape[0]
        if n_neurons < 10:
            print(f"Too few cells to plot traces for {plane_dir.stem}.")
            return output_ops

        # rastermap model
        if not run_rastermap:
            if expected_files["model"].is_file():
                model = np.load(expected_files["model"], allow_pickle=True).item()
            else:
                params = {
                    "n_clusters": 100 if n_neurons >= 200 else None,
                    "n_PCs": min(128, max(2, n_neurons - 1)),
                    "locality": 0.0 if n_neurons >= 200 else 0.1,
                    "time_lag_window": 15,
                    "grid_upsample": 10 if n_neurons >= 200 else 0,
                }
                import rastermap
                model = rastermap.Rastermap(**params).fit(spks)
                np.save(expected_files["model"], model)

                plot_rastermap(
                    spks,
                    model,
                    neuron_bin_size=0,
                    save_path=expected_files["rastermap"],
                    title_kwargs={"fontsize": 8, "y": 0.95},
                    title="Rastermap Sorted Activity",
                )

            if model is not None:
                isort = np.where(iscell == 1)[0][model.isort]
                output_ops["isort"] = isort
                f_norm = f_norm[model.isort]

        # compute dF/F
        # f_clipped = np.clip(f_norm, np.percentile(f_norm, 1), np.percentile(f_norm, 99))
        dff = dff_rolling_percentile(f_norm, percentile=dff_percentile, window_size=dff_window_size) * 100
        dff_noise = dff_shot_noise(dff, output_ops["fs"])

        if n_neurons >= 30:
            print(f"Plotting traces for {plane_dir.stem}...")
            _, _ = plot_traces(
                dff,
                save_path=expected_files["traces_raw"],
                num_neurons=output_ops.get("plot_n_traces", 30),
                signal_units="dffp",
            )
            _, _ = plot_traces(
                f_norm,
                save_path=expected_files["traces_raw"],
                num_neurons=output_ops.get("plot_n_traces", 30),
                signal_units="raw",
            )

        print(f"Plotting noise distribution for {plane_dir.stem}...")
        plot_noise_distribution(dff_noise, output_filename=expected_files["noise"])
        plot_masks(plane_dir)

    fig_label = kwargs.get("fig_label", plane_dir.stem)
    for key in ["meanImg", "max_proj", "meanImgE"]:
        if key in output_ops:
            plot_projection(
                output_ops,
                expected_files[key],
                fig_label=fig_label,
                display_masks=False,
                add_scalebar=True,
                proj=key,
            )

    return output_ops

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
    output_dir = Path(output_dir) if output_dir else input_dir.parent / (input_dir.name + "_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    roi1_dirs = sorted(input_dir.glob("*plane*_roi1*"))
    roi2_dirs = sorted(input_dir.glob("*plane*_roi2*"))
    if not roi1_dirs or not roi2_dirs:
        print("No roi1 or roi2 in input dir")
        return None
    assert len(roi1_dirs) == len(roi2_dirs), "Mismatched ROI dirs"

    for roi1, roi2 in zip(roi1_dirs, roi2_dirs):
        start = time.time()
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
        start_da_from_zarr = time.time()
        z1 = da.from_zarr(roi1)
        z2 = da.from_zarr(roi2)
        print(f"Loaded zarr arrays in {time.time() - start_da_from_zarr:.1f}s")

        # sanity check
        assert z1.shape[0] == z2.shape[0], "Frame count mismatch"
        assert z1.shape[1] == z2.shape[1], "Height mismatch"

        # concatenate along width (axis=2)
        start_concat = time.time()
        z_merged = da.concatenate([z1, z2], axis=2)
        print(f"Concatenated arrays in {time.time() - start_concat:.1f}s")

        # write back
        start_merge = time.time()
        z_merged.to_zarr(out_path, overwrite=overwrite)
        print(f"Wrote zarr in {time.time() - start_merge:.1f}s")
        end = time.time()
        print(f"Wrote {out_path.name} in {end - start:.1f}s")

    if z_merged:
        print(f"Merged zarrs to {output_dir}"
              f"{z_merged}")

    return None


if __name__ == "__main__":

    input_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw\functional")
    files = list(input_path.glob("*"))
    # merge_rois_for_planes(input_path, input_path.parent.joinpath("functional_merged"))
    # out_dir = input_path.parent.joinpath("functional_merged")
    #
    # # merged_ops = load_ops(out_dir / "ops.npy")
    # out_ops = out_dir / "ops.npy"

    input_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw\functional_merged")
    folers = list(input_path.glob("*"))
    for fol in folers:
        remake_plane_figures(fol)

# base = Path("D:/W2_DATA/kbarber/2025_07_17/mk355/green/processed")
# outpath = base.joinpath("output")
# bin_files = list(base.rglob("data.bin"))
# ops_files = list(base.rglob("ops.npy"))
# plane_1_bins = bin_files[:2]
# plane_1_ops = ops_files[:2]
# md1 = lsp.load_ops(ops_files[0])
# md2 = lsp.load_ops(ops_files[1])
# testing_dir = Path(r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\testing")
# output_bin = testing_dir.joinpath("concat.bin")
# output_ops = testing_dir.joinpath("ops.npy")
#
# concat_binfiles_and_merge_metadata(ops_files[0], ops_files[1], output_bin, output_ops)
#
# data = mbo.imread(r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\testing")
# fpl.ImageWidget(data).show()
# fpl.loop.run()
#
# mbo.imwrite(data, r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed", planes=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], roi=0)
#
# files = list(Path(r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed").glob("*.tif"))
#
# ops = suite2p.default_ops()
# ops["roidetect"] = False
#
# lsp.run_volume(files, ops=ops, )
#
# folder1 = r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\plane01_roi1"
# folder2 = r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\plane01_roi2"
# output_folder = r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\plane_01_roi1_roi2"
#
# ops_merged = lsp.load_ops("D:/W2_DATA/kbarber/2025_07_17/mk355/green/processed/plane_01_roi1_roi2/ops.npy")
# stat_merged = np.load("D:/W2_DATA/kbarber/2025_07_17/mk355/green/processed/plane_01_roi1_roi2/stat.npy", allow_pickle=True)
# iscell_merged = np.load("D:/W2_DATA/kbarber/2025_07_17/mk355/green/processed/plane_01_roi1_roi2/iscell.npy")
#
# ops1 = np.load(folder1 + "/ops.npy", allow_pickle=True).item()
# stat1 = np.load(folder1 + "/stat.npy", allow_pickle=True)
# iscell1 = np.load(folder1 + "/iscell.npy", allow_pickle=True)
#
# ops2 = np.load(folder2 + "/ops.npy", allow_pickle=True).item()
# stat2 = np.load(folder2 + "/stat.npy", allow_pickle=True)
# iscell2 = np.load(folder2 + "/iscell.npy", allow_pickle=True)
#
# print(f"roi1 mean-image shape: {ops1['meanImg'].shape}")
# print(f"roi2 mean-image shape: {ops2['meanImg'].shape}")
# print(f"merged mean-image shape: {ops_merged['meanImg'].shape}")
#
# print(f"roi1 xrange: {ops1['xrange']}, yrange: {ops1['yrange']}")
# print(f"roi2 xrange: {ops2['xrange']}, yrange: {ops2['yrange']}")
# print(f"merged xrange: {ops_merged['xrange']}, yrange: {ops_merged['yrange']}")
#
# print(f"Example ROI 1 first 5 : {stat1[0]['xpix'][:5]}, {stat1[0]['ypix'][:5]}")
# print(f"Example ROI 2 first 5 : {stat2[0]['xpix'][:5]}, {stat2[0]['ypix'][:5]}")
