import os
import re
import shutil
import traceback
from pathlib import Path
import mbo_utilities as mbo
import numpy as np
from tifffile import memmap

from suite2p.io import tiff_to_binary
from suite2p.run_s2p import run_plane as s2p_run_plane
import suite2p

from scipy.ndimage import uniform_filter1d

from lbm_suite2p_python.utils import dff_percentile

from lbm_suite2p_python.zplane import (
    plot_traces,
    plot_projection,
    plot_noise_distribution,
    load_planar_results,
    load_ops,
)
from . import dff_shot_noise
from .volume import (
    plot_execution_time,
    plot_volume_signal,
    plot_volume_neuron_counts,
    get_volume_stats,
    save_images_to_movie,
)
if mbo.is_running_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

try:
    from rastermap import Rastermap

    HAS_RASTERMAP = True
except ImportError:
    Rastermap = None
    utils = None
    HAS_RASTERMAP = False
if HAS_RASTERMAP:
    from lbm_suite2p_python.zplane import plot_rastermap

def normalize_plane_name(path):
    name = Path(path).stem
    m = re.search(r'plane[_-](\d+)', name, re.IGNORECASE)
    if not m:
        raise ValueError(f"invalid plane name: {name}")
    return f"plane{int(m.group(1)) - 1}"

def _write_raw_binary(tiff_path: Path, raw_bin: Path):
    raw_bin.parent.mkdir(parents=True, exist_ok=True)
    arr = memmap(str(tiff_path))
    arr.astype(np.int16).tofile(str(raw_bin))
    
def _build_ops(metadata: dict, raw_bin: Path) -> dict:
    nt, Ly, Lx = metadata["shape"]
    dx, dy = metadata.get("pixel_resolution", [2, 2])
    return {
        "Ly": Ly,
        "Lx": Lx,
        "fs": round(metadata["frame_rate"], 2),
        "nframes": nt,
        "raw_file": str(raw_bin),
        "reg_file": str(raw_bin),
        "dx": dx,
        "dy": dy,
        "metadata": metadata,
        "input_format": "binary",
        "delete_bin": False,
        "move_bin": False,
    }

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
            input_tiff=file,
            save_path=str(save_path),
            save_folder=save_folder,
            replot=replot,
        )
        all_ops.append(output_ops)

    # batch was ran, lets accumulate data
    if isinstance(all_ops[0], dict):
        all_ops = [ops["ops_path"] for ops in all_ops]

    try:
        zstats_file = get_volume_stats(all_ops, overwrite=True)

        all_segs = mbo.get_files(save_path, "segmentation.png", 4)
        all_means = mbo.get_files(save_path, "mean_image.png", 4)
        all_maxs = mbo.get_files(save_path, "max_projection_image.png", 4)
        all_traces = mbo.get_files(save_path, "traces.png", 4)

        save_images_to_movie(
            all_segs, os.path.join(save_path, "segmentation_volume.mp4")
        )
        save_images_to_movie(
            all_means, os.path.join(save_path, "mean_images_volume.mp4")
        )
        save_images_to_movie(all_maxs, os.path.join(save_path, "max_images_volume.mp4"))
        save_images_to_movie(all_traces, os.path.join(save_path, "traces_volume.mp4"))

        plot_volume_neuron_counts(zstats_file, save_path)
        plot_volume_signal(
            zstats_file, os.path.join(save_path, "mean_volume_signal.png")
        )
        plot_execution_time(zstats_file, os.path.join(save_path, "execution_time.png"))

        res_z = [
            load_planar_results(ops_path, z_plane=i)
            for i, ops_path in enumerate(all_ops)
        ]
        all_spks = np.concatenate([res["spks"] for res in res_z], axis=0)
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
            plot_rastermap(
                all_spks,
                model,
                neuron_bin_size=20,
                xmax=min(2000, all_spks.shape[1]),
                save_path=os.path.join(save_path, "rastermap.png"),
                title_kwargs=title_kwargs,
                title="Rastermap Sorted Activity",
            )
        else:
            print("No rastermap is available.")

    except Exception:
        print("Volume statistics failed.")
        print("Traceback: ", traceback.format_exc())

    print(f"Processing completed for {len(input_file_list)} files.")
    return all_ops

def run_plane_any(input_path, save_path=None, keep_raw=False, keep_reg=False):
    p = Path(input_path)
    if save_path is None:
        save_path = p.parent
    else:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)

    if p.suffix in [".tiff", ".tif"]:
        metadata = mbo.get_metadata(p)
        plane_folder = normalize_plane_name(p)
        plane_dir = save_path/plane_folder
        plane_dir.mkdir(exist_ok=True)
        raw_bin = plane_dir.joinpath("data_raw.bin")

        if not raw_bin.is_file():

            arr = memmap(str(p))  # shape (T, Y, X), correct dtype (e.g. uint16)
            arr.astype(np.int16).tofile(raw_bin)

        ops_path = plane_dir.joinpath("ops.npy")

        shape = metadata["shape"]
        dx, dy = metadata.get("pixel_resolution", [2, 2])
        ops = {
            "Ly": shape[-2],
            "Lx": shape[-1],
            "fs": np.round(metadata.get("frame_rate"), 17),
            "nframes": shape[0],
            "raw_file": str(raw_bin.resolve()),
            "reg_file": str(raw_bin.resolve()),
            "dx": dx,
            "dy": dy,
            "metadata": metadata,
        }
        np.save(ops_path, ops)

    elif p.suffix in [".bin", ".raw", ".binary"]:
        plane_dir = p
    else:
        raise FileNotFoundError(f"input path {p} does not have a valid suffix.")

    data_raw = plane_dir.joinpath("data_raw.bin")
    ops_path = plane_dir.joinpath("ops.npy")
    if data_raw.is_file() and ops_path.is_file():
        ops = run_plane_bin(plane_dir)
    else:
        raise FileNotFoundError(f"input path {p} does not have a valid suffix.")

    if not keep_raw:
        print(f"Deleting raw binary: {data_raw}")
        if data_raw.is_file():
            os.remove(str(data_raw))
        else:
            print(f"keep_raw set to True, yet no file to delete: {data_raw}")
    if not keep_reg:
        data_reg = plane_dir.joinpath("data.bin")
        if data_reg.is_file():
            print(f"Deleting reg binary: {data_reg}")
            os.remove(str(data_reg))
        else:
            print(f"keep_reg set to True, yet no file to delete: {data_reg}")
    return ops

def run_plane_bin(input_path=None):
    plane_dir = Path(input_path)
    ops_path = plane_dir.joinpath("ops.npy")
    if not ops_path.is_file():
        print("Using default ops...")
        ops = suite2p.default_ops()
    else:
        ops = load_ops(ops_path)

    ops["input_format"] = "binary"
    final_ops = s2p_run_plane(ops, ops_path=str(ops_path))
    return final_ops

def run_plane(
    ops, input_tiff, save_path, save_folder=None, overwrite=False, replot=False, dryrun=False, use_suite3d=False, **kwargs
):
    """
    Processes a single imaging plane using suite2p, handling registration, segmentation,
    and plotting of results.

    Parameters
    ----------
    ops : dict
        Dictionary containing suite2p parameters.
    input_tiff : str or Path, optional
        Path to the input TIFF file. If not given, uses ops["data_path"] / ops["tiff_list"]
    save_path : str or Path, optional
        Directory to save the results.
    save_folder : str, optional
        Subdirectory for saving results (default: filename of input file).
    overwrite : bool, optional
        If True, overwrites existing ops file (default: False).
    replot : bool, optional
        If True, regenerates plots even if they exist (default: False).
    dryrun (experimental): bool, optional
        If True, print input files that will be processed and filepaths that will be created.
    use_suite3d : bool, optional
        If True, use suite3d for processing (default: False).

    Returns
    -------
    dict
        Processed ops dictionary containing results.

    Raises
    ------
    FileNotFoundError
        If `input_tiff` does not exist.
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
    input_tiff = Path(input_tiff)
    if not input_tiff.is_file():
        raise FileNotFoundError(f"Input TIFF not found: {input_tiff}")

    # Derive plane index from filename: “plane_01” → plane0, “plane_07” → plane6, etc.
    stem = input_tiff.stem
    m = re.search(r"plane_(\d+)", stem)
    plane_idx = int(m.group(1)) - 1 if m else 0
    plane_folder = f"plane{plane_idx}"

    # Prepare save directory for this plane
    base = Path(save_path).expanduser().resolve()
    plane_dir = base / plane_folder
    plane_dir.mkdir(parents=True, exist_ok=True)

    # Build the Suite2p “db” dict
    db = {
        "data_path":   [ str(input_tiff.parent) ],
        "tiff_list":   [ input_tiff.name ],
        "save_path0":  str(base),
        "save_folder": plane_folder,
    }

    metadata = mbo.get_metadata(input_tiff)
    if ops is None:
        ops = suite2p.default_ops()
        ops = mbo.params_from_metadata(metadata, ops)

    ops["dx"] = [metadata["pixel_resolution"][0]]
    ops["dy"] = [metadata["pixel_resolution"][0]]

    output_ops = suite2p.run_s2p(ops=ops, db=db)
    return output_ops
    # input_tiff = Path(input_tiff)
    # stem = input_tiff.stem
    # m = re.search(r"plane_(\d+)", stem)
    # plane_idx    = int(m.group(1)) - 1 if m else 0
    # plane_folder = f"plane{plane_idx}"
    # plane_dir    = Path(save_path).expanduser().resolve() / plane_folder
    # plane_dir.mkdir(exist_ok=True)
    #
    # # TODO: I think the only valid `num_frames` we have access to are:
    # #       - the whole recording (from metadata)
    # #       - a single tiff page (page count)
    # #       Need to clarify this in the metadata.
    #
    # expected_files = {
    #     "ops": plane_dir / "ops.npy",
    #     "stat": plane_dir / "stat.npy",
    #     "iscell": plane_dir / "iscell.npy",
    #     "registration": plane_dir / "registration.png",
    #     "segmentation": plane_dir / "segmentation.png",
    #     "max_proj": plane_dir / "max_projection_image.png",
    #     "traces": plane_dir / "traces.png",
    #     "noise": plane_dir / "shot_noise_distrubution.png",
    #     "model": plane_dir / "model.npy",
    #     "rastermap": plane_dir / "rastermap.png",
    # }
    #
    # if not overwrite and all(expected_files[key].is_file() for key in ["ops", "stat", "iscell"]):
    #     print(f"{input_tiff} already has segmentation results. Skipping execution.")
    #     output_ops = load_ops(expected_files["ops"])
    # else:
    #     if dryrun:
    #         print(f"Dryrun: results will be saved in {plane_dir}")
    #         print(f"Files that will be created: {expected_files}")
    #         print(metadata)
    #         return ops, metadata
    #     else:
    #
    #         ops["save_folder"] = plane_folder # used by run_plane and run_s2p
    #         ops["save_path"] = str(plane_dir) # used by s2p_run_plane
    #
    #         # .tiff → .bin
    #         ops["data_path"]  = [ str(input_tiff.parent) ]
    #         ops["tiff_list"]  = [ input_tiff.name      ]
    #         ops["fast_disk"]  = str(plane_dir)              # just so it puts data.bin in plane_dir
    #         ops["dx"] = [metadata["pixel_resolution"][0]]
    #         ops["dy"] = [metadata["pixel_resolution"][0]]
    #
    #         ops_bin_list      = tiff_to_binary(ops)         # this returns one ops dict per plane
    #         ops_plane         = ops_bin_list[plane_idx]
    #
    #         # save the ops.npy for this plane
    #         ops_path = plane_dir / "ops.npy"
    #         np.save(ops_path, ops_plane)
    #
    #         # now call the low-level run_plane and it will write _everything_ into plane_dir
    #         output_ops = s2p_run_plane(ops_plane, ops_path=str(ops_path))
    #         print("Suite2p run complete...")
    # try:
    #     if replot or not all(
    #         expected_files[key].is_file()
    #         for key in ["registration", "segmentation", "traces"]
    #     ):
    #         print(f"Generating missing plots for {input_tiff.stem}...")
    #
    #         def safe_delete(file_path):
    #             if file_path.exists():
    #                 try:
    #                     file_path.unlink()
    #                 except PermissionError:
    #                     print(
    #                         f"Error: Cannot delete {file_path}. Ensure it is not open elsewhere."
    #                     )
    #
    #         for key in ["registration", "segmentation", "traces"]:
    #             safe_delete(expected_files[key])
    #
    #         if ops.get("roidetect", True):
    #             res = load_planar_results(output_ops)
    #             iscell = res["iscell"]
    #             f = res["F"][iscell]
    #
    #             dff = dff_percentile(f, percentile=2) * 100
    #             dff = uniform_filter1d(dff, size=5, axis=1)
    #             dff_noise = dff_shot_noise(dff, ops["fs"])
    #
    #             ncells = min(30, dff.shape[0])
    #             print("Plotting traces...")
    #             plot_traces(dff, save_path=expected_files["traces"], num_neurons=ncells)
    #             print("Plotting noise distribution...")
    #             plot_noise_distribution(dff_noise, save_path=expected_files["noise"])
    #
    #             if HAS_RASTERMAP:
    #                 spks = res["spks"][iscell]
    #                 n_neurons = spks.shape[0]
    #                 if n_neurons < 200:
    #                     params = {
    #                         "n_clusters": None,
    #                         "n_PCs": min(64, n_neurons - 1),
    #                         "locality": 0.1,
    #                         "time_lag_window": 15,
    #                         "grid_upsample": 0
    #                     }
    #                 else:
    #                     params = {
    #                         "n_clusters": 100,
    #                         "n_PCs": 128,
    #                         "locality": 0.0,
    #                         "grid_upsample": 10
    #                     }
    #
    #                 print("Computing rastermap model...")
    #                 model = Rastermap(**params).fit(spks)
    #                 np.save(expected_files["model"], model)
    #
    #                 neuron_bin_size = 1 if n_neurons < 200 else 5 if n_neurons < 500 else 10
    #                 xmax = min(spks.shape[1], int(2000 * (200/n_neurons)**0.5))
    #                 plot_rastermap(
    #                     spks,
    #                     model,
    #                     neuron_bin_size=neuron_bin_size,
    #                     xmax=xmax,
    #                     save_path=expected_files["rastermap"],
    #                     title_kwargs={"fontsize": 8, "y": 0.95},
    #                     title="Rastermap Sorted Activity",
    #                 )
    #             else:
    #                 print("No rastermap is available.")
    #
    #         fig_label = kwargs.get("fig_label", input_tiff.stem)
    #         plot_projection(
    #             output_ops,
    #             expected_files["segmentation"],
    #             fig_label=fig_label,
    #             display_masks=True,
    #             add_scalebar=True,
    #             proj="meanImg",
    #         )
    #         plot_projection(
    #             output_ops,
    #             expected_files["max_proj"],
    #             fig_label=input_tiff.stem,
    #             display_masks=False,
    #             add_scalebar=True,
    #             proj="max_proj",
    #         )
    #         print("Plots generated successfully.")
    # except Exception:
    #     traceback.print_exc()


def run_grid_search(base_ops: dict, grid_search_dict: dict, input_file: Path | str, save_root: Path | str):
    """
    Run a grid search over all combinations of the input suite2p parameters.

    Parameters
    ----------
    base_ops : dict
        Dictionary of default Suite2p ops to start from. Each parameter combination will override values in this dictionary.

    grid_search_dict : dict
        Dictionary mapping parameter names (str) to a list of values to grid search.
        Each combination of values across parameters will be run once.

    input_file : str or Path
        Path to the input data file, currently only supports tiff.

    save_root : str or Path
        Root directory where each parameter combination's output will be saved.
        A subdirectory will be created for each run using a short parameter tag.

    Notes
    -----
    - Subfolder names for each parameter are abbreviated to 3-character keys and truncated/rounded values.

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>> import suite2p
    >>> base_ops = suite2p.default_ops()
    >>> base_ops["anatomical_only"] = 3
    >>> base_ops["diameter"] = 6
    >>> lsp.run_grid_search(
    ...     base_ops,
    ...     {"threshold_scaling": [1.0, 1.2], "tau": [0.1, 0.15]},
    ...     input_file="/mnt/data/assembled_plane_03.tiff",
    ...     save_root="/mnt/grid_search/"
    ... )

    This will create the following output directory structure::

        /mnt/data/grid_search/
        ├── thr1.00_tau0.10/
        │   └── suite2p output for threshold_scaling=1.0, tau=0.1
        ├── thr1.00_tau0.15/
        ├── thr1.20_tau0.10/
        └── thr1.20_tau0.15/

    See Also
    --------
    [](http://suite2p.readthedocs.io/en/latest/parameters.html)

    """
    from itertools import product
    from pathlib import Path
    import copy

    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True)

    print(f"Saving grid-search in {save_root}")

    param_names = list(grid_search_dict.keys())
    param_values = list(grid_search_dict.values())
    param_combos = list(product(*param_values))

    for combo in param_combos:
        ops = copy.deepcopy(base_ops)
        combo_dict = dict(zip(param_names, combo))
        ops.update(combo_dict)

        tag_parts = [
            f"{k[:3]}{v:.2f}" if isinstance(v, float) else f"{k[:3]}{v}"
            for k, v in combo_dict.items()
        ]
        tag = "_".join(tag_parts)

        print(f"Running grid search in: {save_root.joinpath(tag)}")
        run_plane(ops, input_file, save_root, save_folder=tag)
