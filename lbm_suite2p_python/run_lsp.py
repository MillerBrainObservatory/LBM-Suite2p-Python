import logging
import os
import traceback
from contextlib import nullcontext

import numpy as np

import suite2p
from suite2p.io.binary import BinaryFile
from lbm_suite2p_python.utils import dff_rolling_percentile
import mbo_utilities as mbo  # noqa

logger = mbo.log.get("run_lsp")

from lbm_suite2p_python.zplane import (
    plot_traces,
    plot_projection,
    plot_noise_distribution,
    load_planar_results,
    load_ops,
    suite2p_roi_overlay, plot_traces_noise
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

from pathlib import Path

PIPELINE_TAGS = ("plane", "roi", "z", "plane_", "roi_", "z_")


def normalize_folder(path):
    """
    Derive a folder tag from a filename based on “planeN”, “roiN”, or "tagN" patterns.

    Parameters
    ----------
    path : str or pathlib.Path
        File path or name whose stem will be parsed.

    Returns
    -------
    str
        If the stem starts with “plane”, “roi”, or “res” followed by an integer,
        returns that tag plus the integer (e.g. “plane3”, “roi7”, “res2”).
        Otherwise returns the original stem unchanged.

    Examples
    --------
    >>> normalize_folder("plane_01.tif")
    'plane1'
    >>> normalize_folder("plane2.bin")
    'plane2'
    >>> normalize_folder("roi5.raw")
    'roi5'
    >>> normalize_folder("ROI_10.dat")
    'roi10'
    >>> normalize_folder("res-3.h5")
    'res3'
    >>> normalize_folder("assembled_data_1.tiff")
    'assembled_data_1'
    >>> normalize_folder("file_12.tif")
    'file_12'
    """
    name = Path(path).stem
    for tag in PIPELINE_TAGS:
        low = name.lower()
        if low.startswith(tag):
            suffix = name[len(tag) :]
            if suffix and (suffix[0] in ("_", "-")):
                suffix = suffix[1:]
            if suffix.isdigit():
                return f"{tag}{int(suffix)}"
    return name


def get_missing_ops_keys(ops: dict) -> list[str]:
    required = ["Ly", "Lx", "fs", "nframes", "raw_file", "input_format"]
    return [k for k in required if k not in ops or ops[k] is None]


def run_volume(
    input_files: list,
    save_path: str | Path = None,
    ops: dict | str | Path = None,
    keep_reg: bool = True,
    keep_raw: bool = True,
    force_reg: bool = False,
    force_detect: bool = False,
    replot: bool = False,
):
    """
    Processes a full volumetric imaging dataset using Suite2p, handling plane-wise registration,
    segmentation, plotting, and aggregation of volumetric statistics and visualizations.

    Parameters
    ----------
    input_files : list of str or Path
        List of TIFF file paths, each representing a single imaging plane.
    save_path : str or Path, optional
        Base directory to save all outputs.
        If none, will create a "volume" directory in the parent of the first input file.
    ops : dict or list, optional
        Dictionary of Suite2p parameters to use for each imaging plane.
    save_path : str, optional
        Subdirectory name within `save_path` for saving results (default: None).
    keep_raw : bool, default false
        if true, do not delete the raw binary (`data_raw.bin`) after processing.
    keep_reg : bool, default false
        if true, do not delete the registered binary (`data.bin`) after processing.
    force_reg : bool, default false
        if true, force a new registration even if existing shifts are found in ops.npy.
    force_detect : bool, default false
        if true, force roi detection even if an existing stat.npy is present.
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
    if save_path is None:
        save_path = Path(input_files[0]).parent

    all_ops = []
    for file in tqdm(input_files, desc="Processing Planes"):
        print(f"Processing {file} ---------------")
        output_ops = run_plane(
            input_path=file,
            save_path=str(save_path),
            ops=ops,
            keep_reg=keep_reg,
            keep_raw=keep_raw,
            force_reg=force_reg,
            force_detect=force_detect,
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

    print(f"Processing completed for {len(input_files)} files.")
    return all_ops

def should_write_ops(ops_path, ops, force=False):
    if force or not ops_path.exists():
        return True
    try:
        existing_ops = np.load(ops_path, allow_pickle=True).item()
        has_registration = "xoff" in existing_ops and "meanImg" in existing_ops
        has_detection = "stat" in ops or (ops_path.parent / "stat.npy").exists()
        return not (has_registration and has_detection)
    except Exception:
        return True


def run_plane_bin(ops):
    ops = load_ops(ops)
    if "nframes" in ops and "n_frames" not in ops:
        ops["n_frames"] = ops["nframes"]
    if "n_frames" not in ops:
        raise KeyError("run_plane_bin: missing frame count (nframes or n_frames)")
    n_frames = ops["n_frames"]
    Ly, Lx = ops["Ly"], ops["Lx"]

    prior_ops = {}
    if Path(ops["ops_path"]).exists():
        prior_ops = np.load(ops["ops_path"], allow_pickle=True).item()

    with (
        suite2p.io.BinaryFile(
            Ly=Ly, Lx=Lx, filename=ops["reg_file"], n_frames=n_frames
        ) as f_reg,
        (
            suite2p.io.BinaryFile(
                Ly=Ly, Lx=Lx, filename=ops["raw_file"], n_frames=n_frames
            )
            if "raw_file" in ops and ops["raw_file"] is not None
            else nullcontext()
        ) as f_raw,
    ):
        ops = suite2p.pipeline(
            f_reg, f_raw, None, None, ops["do_registration"], ops, stat=None
        )

    # merge in any non-conflicting prior fields
    merged_ops = {**ops, **{k: v for k, v in prior_ops.items() if k not in ops}}
    np.save(ops["ops_path"], merged_ops)

    return merged_ops

def run_plane(
    input_path: str | Path,
    save_path: str | Path | None = None,
    ops: dict | str | Path = None,
    keep_raw: bool = False,
    keep_reg: bool = True,
    force_reg: bool = False,
    force_detect: bool = False,
    dff_window_size: int = 500,
    dff_percentile: int = 20,
    **kwargs,
):
    """
    Processes a single imaging plane using suite2p, handling registration, segmentation,
    and plotting of results.

    Parameters
    ----------
    input_path : str or Path
        Full path to the file to process, with the file extension.
    save_path : str or Path, optional
        Directory to save the results.
    ops : dict, str or Path, optional
        Path to or dict of user‐supplied ops.npy. If given, it overrides any existing or generated ops.
    keep_raw : bool, default false
        if true, do not delete the raw binary (`data_raw.bin`) after processing.
    keep_reg : bool, default false
        if true, do not delete the registered binary (`data.bin`) after processing.
    force_reg : bool, default false
        if true, force a new registration even if existing shifts are found in ops.npy.
    force_detect : bool, default false
        if true, force roi detection even if an existing stat.npy is present.
    dff_window_size : int, default 10
        Size of the window for calculating dF/F traces.
    dff_percentile : int, default 8
        Percentile to use for baseline F₀ estimation in dF/F calculation.
    **kwargs : dict, optional

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

    Notes
    -----
    - ops supplied to the function via `ops_file` will take precendence over previously saved ops.npy files.

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

    Run a single z-plane through suite2p, keeping raw and registered files.
    >> output_ops = lsp.run_plane(input_files[0], save_path="D://data//outputs", keep_raw=True, keep_registered=True, force_reg=True, force_detect=True)
    """
    if "debug" in kwargs:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled.")

    assert isinstance(
        input_path, (Path, str)
    ), f"input_path should be a pathlib.Path or string, not: {type(input_path)}"
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")
    input_parent = input_path.parent

    assert isinstance(
        save_path, (Path, str, type(None))
    ), f"save_path should be a pathlib.Path or string, not: {type(save_path)}"
    if save_path is None:
        logger.debug(f"save_path is None, using parent of input file: {input_parent}")
        save_path = input_parent
    else:
        save_path = Path(save_path)
        if not save_path.parent.is_dir():
            raise ValueError(
                f"save_path does not have a valid parent directory: {save_path}"
            )
        save_path.mkdir(exist_ok=True)

    ops_default = suite2p.default_ops()
    ops_user = load_ops(ops) if ops else {}
    ops = {**ops_default, **ops_user}

    file = mbo.imread(input_path)
    metadata = file.metadata
    if "plane" in ops:
        plane = ops["plane"]
        metadata["plane"] = plane
    elif "plane" in metadata:
        plane = metadata["plane"]
        ops["plane"] = plane
    else:
        # get the plane from the filename
        plane = mbo.get_plane_from_filename(input_path, ops.get("plane", None))
        ops["plane"] = plane
        metadata["plane"] = plane

    plane_dir = save_path
    needs_detect = force_detect or not (plane_dir / "stat.npy").exists()

    ops_file = plane_dir / "ops.npy"
    reg_data_file = plane_dir / "data.bin"
    reg_data_file2 = plane_dir / "reg_tif"

    if should_write_ops(ops_file, ops, force=kwargs.get("force_save", False)):
        mbo.imwrite(file, plane_dir, ext=".bin", metadata=metadata)
    else:
        print(f"Skipping ops.npy save: {ops_file.name} already contains results.")

    ops_outpath = (
        np.load(ops_file, allow_pickle=True).item()
        if (plane_dir / "ops.npy").exists()
        else {}
    )

    exists = False
    if reg_data_file.exists():
        exists = True
    if reg_data_file2.exists():
        exists = True
    if force_reg:
        needs_reg = True
    else:
        # if either reg data file exists, we assume registration is done
        needs_reg = not exists

    ops = {
        **ops,
        **ops_outpath,  # merge any existing ops
        "ops_path": str(ops_file),
        "do_registration": int(needs_reg),
        "roidetect": int(needs_detect),
        "save_path": str(plane_dir),
        "raw_file": str((plane_dir / "data_raw.bin").resolve()),
        "reg_file": str((plane_dir / "data.bin").resolve())
    }

    if "nframes" not in ops and "shape" in ops.get("metadata", {}):
        ops["nframes"] = ops["metadata"]["shape"][0]

    ops = run_plane_bin(ops)
    output_ops = load_ops(ops_file)

    # cleanup ourselves
    if not keep_raw:
        (plane_dir / "data_raw.bin").unlink(missing_ok=True)
    if not keep_reg:
        (plane_dir / "data.bin").unlink(missing_ok=True)

    expected_files = {
        "ops": plane_dir / "ops.npy",
        "stat": plane_dir / "stat.npy",
        "iscell": plane_dir / "iscell.npy",
        "registration": plane_dir / "registration.png",
        "segmentation": plane_dir / "segmentation.png",
        "segmentation_traces": plane_dir / "segmentation_match_traces.png",
        "max_proj": plane_dir / "max_projection_image.png",
        "meanImg": plane_dir / "mean_image.png",
        "meanImgE": plane_dir / "mean_image_enhanced.png",
        "traces": plane_dir / "traces.png",
        "traces_noise": plane_dir / "traces_noise.png",
        "noise": plane_dir / "shot_noise_distrubution.png",
        "model": plane_dir / "model.npy",
        "rastermap": plane_dir / "rastermap.png",
    }
    try:
        if not all(
            expected_files[key].is_file()
            for key in ["registration", "segmentation", "traces"]
        ):
            print(f"Generating missing plots for {plane_dir.stem}...")

            def safe_delete(file_path):
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except PermissionError:
                        print(
                            f"Error: Cannot delete {file_path}. Ensure it is not open elsewhere."
                        )

            for key in ["registration", "segmentation", "traces"]:
                safe_delete(expected_files[key])

            model = None
            colors = None
            if expected_files["stat"].is_file():
                res = load_planar_results(output_ops)
                iscell = res["iscell"]
                spks = res["spks"][iscell]
                n_neurons = spks.shape[0]

                if iscell.ndim == 2:
                    iscell = iscell[:, 0]

                stat = res["stat"]
                f = res["F"][iscell]

                if f.shape[0] < 10:
                    print(f"Too few cells to plot traces for {plane_dir.stem}.")
                    return output_ops

                if expected_files["model"].is_file():
                    print("Loading cached rastermap model...")
                    model = np.load(expected_files["model"], allow_pickle=True).item()
                else:
                    if n_neurons < 200:
                        params = {
                            "n_clusters": None,
                            "n_PCs": min(64, n_neurons - 1),
                            "locality": 0.1,
                            "time_lag_window": 15,
                            "grid_upsample": 0,
                        }
                    else:
                        params = {
                            "n_clusters": 100,
                            "n_PCs": 128,
                            "locality": 0.0,
                            "grid_upsample": 10,
                        }

                    print("Computing rastermap model...")
                    model = Rastermap(**params).fit(spks)
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
                    print("Sorting neurons by rastermap model...")
                    isort = np.where(iscell == 1)[0][model.isort]
                    output_ops["isort"] = isort  # now global to stat, not local

                percentile = output_ops.get("dff_percentile", dff_percentile)
                win_size = output_ops.get("dff_window_size", dff_window_size)
                dff = dff_rolling_percentile(
                    f,
                    percentile=percentile,
                    window_size=win_size
                ) * 100  # convert to percentage
                dff_noise = dff_shot_noise(dff, output_ops["fs"])

                if n_neurons < 10:
                    print(f"Too few cells to plot traces for {plane_dir.stem}.")
                else:
                    print("Plotting traces...")
                    fig, colors = plot_traces(
                        dff,
                        save_path=expected_files["traces"],
                        num_neurons=n_neurons,
                        signal_units="dffp",
                        return_color=True
                    )
                    plot_traces_noise(
                        dff_noise,
                        ncells=n_neurons,
                        savepath=expected_files["traces_noise"]
                    )

                print("Plotting noise distribution...")
                plot_noise_distribution(dff_noise, save_path=expected_files["noise"])

                suite2p_roi_overlay(
                    output_ops,
                    stat,
                    iscell,
                    "max_proj",
                    plot_indices=None,
                    savepath=expected_files["segmentation"],
                )
                cell_indices = output_ops["isort"][:n_neurons]
                suite2p_roi_overlay(
                    output_ops,
                    stat,
                    iscell,
                    "max_proj",
                    plot_indices=cell_indices,
                    savepath=expected_files["segmentation_traces"],
                    color_mode="colormap",
                    colors=colors if colors is not None else None,
                )

            fig_label = kwargs.get("fig_label", plane_dir.stem)
            for key in ["meanImg", "max_proj", "meanImgE"]:
                if key not in output_ops:
                    continue
                plot_projection(
                    output_ops,
                    expected_files[key],
                    fig_label=fig_label,
                    display_masks=False,
                    add_scalebar=True,
                    proj=key,
                )
            print("Plots generated successfully.")
    except Exception:
        traceback.print_exc()
    return output_ops


def run_grid_search(
    base_ops: dict,
    grid_search_dict: dict,
    input_file: Path | str,
    save_root: Path | str,
):
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
    [suite2p parameters](http://suite2p.readthedocs.io/en/latest/parameters.html)

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
