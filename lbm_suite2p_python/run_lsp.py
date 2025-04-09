import os
import traceback
from pathlib import Path
import mbo_utilities as mbo
import numpy as np

import suite2p
from scipy.ndimage import uniform_filter1d

from .utils import load_ops, dff_percentile, plot_projection

from .zplane import (
    plot_traces,
    animate_traces
)
from .volume import (
    plot_execution_time,
    plot_volume_signal,
    plot_volume_stats,
    get_volume_stats,
    save_images_to_movie,
    load_results_dict, plot_rastermap,
)

if mbo.is_running_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

try:
    from rastermap import Rastermap, utils
    HAS_RASTERMAP = True
except ImportError:
    Rastermap = None
    utils = None
    HAS_RASTERMAP = False


def run_volume(ops, input_file_list, save_path, save_folder=None, replot=False):
    """
    Processes a full volumetric imaging dataset using Suite2p, handling plane-wise registration,
    segmentation, plotting, and aggregation of volumetric statistics and visualizations.

    Parameters
    ----------
    ops : dict or list
        Dictionary of Suite2p parameters to use for each imaging plane.
    input_file_list : list of str or Path
        List of TIFF file paths, each representing a single imaging plane.
    save_path : str or Path
        Base directory to save all outputs.
    save_folder : str, optional
        Subdirectory name within `save_path` for saving results (default: None).
    replot : bool, optional
        If True, regenerate all summary plots even if they already exist (default: False).

    Returns
    -------
    list of str
        List of paths to `ops.npy` files for each plane.

    Raises
    ------
    Exception
        If volumetric summary statistics or any visualization fails to generate.

    Example
    -------
    >> input_files = mbo.get_files(assembled_path, str_contains='tif', max_depth=3)
    >> ops = mbo.params_from_metadata(mbo.get_metadata(input_files[0]), suite2p.default_ops())

    Run volume
    >> output_ops_list = lsp.run_volume(ops, input_files, save_path)

    Notes
    -----
    At the root of `save_path` will be a folder for each z-plane with all suite2p results, as well as 
    volumetric outputs at the base of this folder.

    Each z-plane folder contains:
    - Registration, Segmentation and Extraction results (ops, spks, iscell)
    - Summary statistics: execution time, signal strength, acceptance rates
    - Optional rastermap model for visualization of activity across the volume

    Each save_path root contains:
    - Accepted/Rejected histogram, neuron-count x z-plane (acc_rej_bar.png)
    - Execution time for each step in each z-plane (execution_time.png)
    - Mean/Max images, with and without segmentation masks, in GIF/MP4
    - Traces animation over time and neurons
    - Optional rastermap clustering results
    """
    all_ops = []
    for file in tqdm(input_file_list, desc="Processing Planes"):
        print(f"Processing {file} ---------------")
        output_ops = run_plane(
            ops=ops,
            input_file_path=file,
            save_path=str(save_path),
            save_folder=save_folder,
            replot=replot
        )
        all_ops.append(output_ops)

    # batch was ran, lets accumulate data
    if isinstance(all_ops[0], dict):
        all_ops = [ops['ops_path'] for ops in all_ops]

    try:
        zstats_file = get_volume_stats(all_ops, overwrite=True)

        all_segs = mbo.get_files(save_path, "segmentation.png", 4)
        all_means = mbo.get_files(save_path, "mean_image.png", 4)
        all_maxs = mbo.get_files(save_path, "max_projection_image.png", 4)
        all_traces = mbo.get_files(save_path, "traces.png", 4)

        save_images_to_movie(all_segs, os.path.join(save_path, "segmentation_volume.mp4"))
        save_images_to_movie(all_means, os.path.join(save_path, "mean_images_volume.mp4"))
        save_images_to_movie(all_maxs, os.path.join(save_path, "max_images_volume.mp4"))
        save_images_to_movie(all_traces, os.path.join(save_path, "traces_volume.mp4"))

        plot_volume_stats(zstats_file, os.path.join(save_path, "acc_rej_bar.png"))
        plot_volume_signal(zstats_file, os.path.join(save_path, "mean_volume_signal.png"))
        plot_execution_time(zstats_file, os.path.join(save_path, "execution_time.png"))

        res_z = [load_results_dict(ops_path, apply_zscore=True, z_plane=i) for i, ops_path in enumerate(all_ops)]
        all_spks = np.concatenate([res['spks'] for res in res_z], axis=0)
        print(type(all_spks))
        # all_iscell = np.stack([res['iscell'] for res in res_z], axis=-1)
        if HAS_RASTERMAP:
            model = Rastermap(
                n_clusters=100,
                n_PCs=100,
                locality=0.75,
                time_lag_window=15,
            ).fit(all_spks)
            np.save(os.path.join(save_path, "model.npy"), model)
            title_kwargs = {"fontsize": 8, "y": 0.95}
            plot_rastermap(all_spks,model, neuron_bin_size=20, xmax=min(2000, all_spks.shape[1]), save_path=os.path.join(save_path, "rastermap.png"), title_kwargs=title_kwargs, title="Rastermap Sorted Activity")

    except Exception:
        print("Volume statistics failed. Showing ops:")
        print(all_ops)
        print("Traceback: ", traceback.format_exc())

    print(f"Processing completed for {len(input_file_list)} files.")
    return all_ops

def run_plane(ops, input_file_path, save_path, save_folder=None, replot=False, dryrun=False):
    """
    Processes a single imaging plane using suite2p, handling registration, segmentation,
    and plotting of results.

    Parameters
    ----------
    ops : dict
        Dictionary containing suite2p parameters.
    input_file_path : str or Path
        Path to the input TIFF file.
    save_path : str or Path
        Directory to save the results.
    save_folder : str, optional
        Subdirectory for saving results (default: filename of input file).
    replot : bool, optional
        If True, regenerates plots even if they exist (default: False).
    dryrun : bool, optional
        If True, print input files that will be processed and filepaths that will be created.

    Returns
    -------
    dict
        Processed ops dictionary containing results.

    Raises
    ------
    FileNotFoundError
        If `input_file_path` does not exist.
    TypeError
        If `save_folder` is not a string.
    Exception
        If plotting functions fail.

    Example
    -------
    >> import mbo_utilities as mbo
    >> import lbm_suite2p_python as lsp

    Get a list of z-planes in Txy format
    >> input_files = mbo.get_files(assembled_path, str_contains='tif', max_depth=3)
    >> metadata = mbo.get_metadata(input_files[0])
    >> ops = suite2p.default_ops()

    Automatically fill in metadata needed for processing (frame rate, pixel resolution, etc..)
    >> mbo_ops = mbo.params_from_metadata(metadata, ops) # handles framerate, Lx/Ly, etc

    Run a single z-plane through suite2p
    >> output_ops = lsp.run_plane(mbo_ops, input_files[0], save_path)
    """
    input_file_path = Path(input_file_path)
    if not input_file_path.is_file():
        if dryrun:
            print(f"Input file {input_file_path} does not exist.")
            return
        else:
            raise FileNotFoundError(f"Input data file {input_file_path} does not exist.")

    save_path = Path(save_path)
    if dryrun:
        print(f"Input file {input_file_path} will save in {save_path}")
    else:
        save_path.mkdir(parents=True, exist_ok=True)

    if save_folder is None:
        save_folder = input_file_path.stem
    elif not isinstance(save_folder, (str, Path)):
        raise TypeError("save_folder must be a string or path-like object.")
    else:
        save_folder = Path(save_folder)

    print(f"saving to {save_folder}")

    if ops is None:
        ops = suite2p.default_ops()


    plane_path = save_path / save_folder / "plane0"

    metadata = mbo.get_metadata(input_file_path)
    ops = mbo.params_from_metadata(metadata, ops)

    expected_files = {
        "ops": plane_path / "ops.npy",
        "stat": plane_path / "stat.npy",
        "iscell": plane_path / "iscell.npy",
        "registration": plane_path / "registration.png",
        "segmentation": plane_path / "segmentation.png",
        "meanImg" : plane_path / "mean_image.png",
        "max_proj" : plane_path / "max_projection_image.png",
        "traces": plane_path / "traces.png",
        "animation": plane_path / "animated_traces.mp4"
    }

    if all(expected_files[key].is_file() for key in ["ops", "stat", "iscell"]):
        print(f"{input_file_path} already has segmentation results. Skipping execution.")
        output_ops = load_ops(expected_files["ops"])
    else:
        if dryrun:
            print(f"Dryrun: results will be saved in {plane_path}")
        else:

            db = {"data_path": [str(input_file_path.parent)], "save_folder": save_folder, "save_path0": str(save_path), "tiff_list": [input_file_path.name]}
            output_ops = suite2p.run_s2p(ops=ops, db=db)

    # remove when we set data.bin path correctly
    # monkey patch to deal with default suite2p/plane0/data.bin save path
    raw_path = save_path / "suite2p" / "plane0" / "data.bin"
    where_raw_should_be_path = plane_path / "data.bin"
    if raw_path.is_file():
        if ops["keep_movie_raw"]:
            print(f"Moving {raw_path} -> {where_raw_should_be_path}")
            if not where_raw_should_be_path.exists():
                raw_path.rename(where_raw_should_be_path)
            else:
                print(f"Warning: {where_raw_should_be_path} already exists. Skipping rename.")
        else:
            try:
                print(f"Deleting {raw_path} due to ops['keep_movie_raw=False'].")
                raw_path.unlink()
                if not any(raw_path.parent.parent.iterdir()):
                    raw_path.parent.parent.rmdir()
            except Exception as e:
                print(f"Failed to delete {raw_path}: {e}")

    try:
        if replot or not all(expected_files[key].is_file() for key in [
            "registration", "segmentation", "traces"]):
            print(f"Generating missing plots for {input_file_path.stem}...")

            def safe_delete(file_path):
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except PermissionError:
                        print(f"Error: Cannot delete {file_path}. Ensure it is not open elsewhere.")

            for key in ["registration", "segmentation", "traces"]:
                safe_delete(expected_files[key])

            f = np.load(Path(output_ops["ops_path"]).parent.joinpath("F.npy"))
            dff = dff_percentile(f, percentile=2) * 100
            dff = uniform_filter1d(dff, size=5, axis=1)
            # TODO: make sure there are at least 30 traces
            plot_traces(dff, save_path=expected_files["traces"], start_neurons=30)

            animate_traces(
                dff,
                save_path=expected_files["animation"],
                start_neurons=30,
                expand_after=5,
                lw=0.5,
                speed_factor=8,
                expansion_factor=10,
            )
            plot_projection(
                output_ops,
                expected_files["segmentation"],
                fig_label=input_file_path.stem,
                display_masks=True,
                add_scalebar=True,
                proj="meanImg"
            )
            # do one for mean/max image, no masks
            for projection in ["meanImg", "max_proj"]:
                plot_projection(
                    output_ops,
                    expected_files[projection],
                    fig_label=input_file_path.stem,
                    display_masks=False,
                    add_scalebar=True,
                    proj=projection
                )
    except Exception as e:
        traceback.print_exc()

    return output_ops
