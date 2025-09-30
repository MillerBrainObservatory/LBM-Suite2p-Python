import logging
import time
from pathlib import Path
import os
import traceback
from contextlib import nullcontext
from itertools import product
import copy
import gc

import numpy as np
import pandas as pd

import suite2p
from suite2p.io.binary import BinaryFile
from lbm_suite2p_python.merging import remake_plane_figures
from lbm_suite2p_python.utils import ops_to_json, load_planar_results, load_ops
from mbo_utilities.log import get as get_logger
import mbo_utilities as mbo  # noqa

logger = get_logger("run_lsp")

from lbm_suite2p_python._benchmarking import get_cpu_percent, get_ram_used
from lbm_suite2p_python.volume import (
    plot_volume_signal,
    plot_volume_neuron_counts,
    get_volume_stats,
    save_images_to_movie,
    plot_execution_time,
)

PIPELINE_TAGS = ("plane", "roi", "z", "plane_", "roi_", "z_")


def derive_tag_from_filename(path):
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
    >>> derive_tag_from_filename("plane_01.tif")
    'plane1'
    >>> derive_tag_from_filename("plane2.bin")
    'plane2'
    >>> derive_tag_from_filename("roi5.raw")
    'roi5'
    >>> derive_tag_from_filename("ROI_10.dat")
    'roi10'
    >>> derive_tag_from_filename("res-3.h5")
    'res3'
    >>> derive_tag_from_filename("assembled_data_1.tiff")
    'assembled_data_1'
    >>> derive_tag_from_filename("file_12.tif")
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


def summarize_regdx_metrics(root_dir: Path, subdir: str = "") -> pd.DataFrame:
    rows = []

    for ops_path in root_dir.rglob(f"*{subdir}*/ops.npy"):
        try:
            ops = load_ops(ops_path)
            regdx = np.array(ops.get("regDX"))
            if regdx is None or len(regdx) == 0 or regdx.ndim != 2:
                continue

            dx = regdx[:, 0]
            dy = regdx[:, 1]
            norm = regdx[:, 2]

            rows.append({
                "config": ops_path.parent.parent.name,
                "mean_rigid": np.mean(np.abs(dx)),
                "mean_avg_nr": np.mean(np.abs(dy)),
                "mean_max_nr": np.mean(np.abs(norm)),
            })
        except Exception as e:
            print(f"Error processing {ops_path}: {e}")
            continue
    return pd.DataFrame(rows)

def run_volume(
        input_files: list,
        save_path: str | Path = None,
        ops: dict | str | Path = None,
        keep_reg: bool = True,
        keep_raw: bool = True,
        force_reg: bool = False,
        force_detect: bool = False,
        dff_window_size: int = 500,
        dff_percentile: int = 20,
        save_json: bool = False,
        **kwargs,
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
    dff_window_size : int, default 500
        Number of frames to use for windowed dF/F₀ calculations.
    dff_percentile : int, default 20
        Percentile to use for baseline F₀ estimation in dF/F₀ calculations.
    save_json : bool, default False
        If true, saves ops as a JSON file in addition to npy.

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
    start = time.time()
    if save_path is None:
        save_path = Path(input_files[0]).parent

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    all_ops = []
    for file in input_files:
        start_file = time.time()
        subdir = derive_tag_from_filename(Path(file).stem)
        plane_save_path = Path(save_path).joinpath(subdir)
        plane_save_path.mkdir(exist_ok=True)
        ops_file = run_plane(
            input_path=file,
            save_path=plane_save_path,
            ops=ops,
            keep_reg=keep_reg,
            keep_raw=keep_raw,
            force_reg=force_reg,
            force_detect=force_detect,
            dff_window_size=dff_window_size,
            dff_percentile=dff_percentile,
            save_json=save_json,
        )
        end_file = time.time()
        print(f"Time for {file}: {(end_file - start_file)/60:0.1f} min")
        print(f"CPU {get_cpu_percent():4.1f}% | RAM {get_ram_used()/1024:5.2f} GB")
        all_ops.append(ops_file)
        del ops_file
        gc.collect()

    end = time.time()
    print(f"Total time for volume: {(end - start)/60:0.1f} min")

    if "roi" in Path(input_files[0]).stem.lower():
        print(f"Detected mROI data, merging ROIs for each z-plane...")
        from .merging import merge_rois_for_planes, remake_plane_figures
        base_dir = Path(input_files[0]).parent
        merged_dir = base_dir.joinpath(f"merged")
        merged_dir.mkdir(exist_ok=True)
        print(f"Merging ROIs for planes in {base_dir}...")
        merge_rois_for_planes(base_dir, merged_dir)
        print("Remaking plane figures...")
        for ops_path in all_ops:
            remake_plane_figures(ops_path.parent)

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
        # todo: why is suite2p not saving timings to ops.npy?
        plot_execution_time(zstats_file, os.path.join(save_path, "execution_time.png"))

        res_z = [
            load_planar_results(ops_path, z_plane=i)
            for i, ops_path in enumerate(all_ops)
        ]
        all_spks = np.concatenate([res["spks"] for res in res_z], axis=0)
        print(type(all_spks))
        try:
            from rastermap import Rastermap
            from lbm_suite2p_python.zplane import plot_rastermap
            HAS_RASTERMAP = True
        except ImportError:
            Rastermap = None
            HAS_RASTERMAP = False
            plot_rastermap = None
        if HAS_RASTERMAP:
            model = Rastermap(
                n_clusters=100,
                n_PCs=100,
                locality=0.75,
                time_lag_window=15,
            ).fit(all_spks)
            np.save(os.path.join(save_path, "model.npy"), model)
            title_kwargs = {"fontsize": 8, "y": 0.95}
            if plot_rastermap is not None:
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

def _should_write_bin(ops_path: Path, force: bool = False) -> bool:
    """
    Decide whether a data_raw.bin should be re-written.

    Conditions that trigger re-write:
      - force=True
      - bin file missing
      - ops.npy missing
      - mismatch between ops metadata (Ly, Lx, nframes) and bin file size
      - bin file cannot be read or has wrong shape
    """
    ops_path = Path(ops_path)
    if not ops_path.is_file():
        return True

    bin_path = ops_path.parent / "data_raw.bin"
    tiff_path = ops_path.parent / "reg_tif"

    if not bin_path.is_file() and not tiff_path.is_dir() or force:
        return True


    try:
        ops = np.load(ops_path, allow_pickle=True).item()
        Ly, Lx = ops.get("Ly"), ops.get("Lx")
        nframes = ops.get("nframes", ops.get("n_frames"))

        if None in (Ly, Lx, nframes):
            return True

        expected_size = nframes * Ly * Lx * np.dtype(np.int16).itemsize
        actual_size = bin_path.stat().st_size

        if actual_size != expected_size:
            return True

        # Try opening first few frames to verify integrity
        arr = np.memmap(bin_path, dtype=np.int16, mode="r",
                        shape=(nframes, Ly, Lx))
        _ = arr[0].sum()  # touch data
        del arr

        return False  # all checks passed
    except Exception as e:
        print(f"Bin validation failed: {e}")
        return True


def run_plane_bin(ops) -> bool:
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
    print(f"Saved ops to {ops['ops_path']}")

    # merge in any non-conflicting prior fields
    merged_ops = {**ops, **{k: v for k, v in prior_ops.items() if k not in ops}}
    np.save(ops["ops_path"], merged_ops)
    del merged_ops, f_reg, f_raw, ops

    return True

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
        save_json: bool = False,
        **kwargs,
) -> Path:
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
    save_json : bool, default True
        If true, saves ops as a JSON file in addition to npy.
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

    assert isinstance(input_path, (Path, str)), (
        f"input_path should be a pathlib.Path or string, not: {type(input_path)}"
    )
    input_path = Path(input_path)
    if not input_path.is_file():
        if input_path.suffix != ".zarr":
            raise ValueError(f"Input file does not exist: {input_path}")
    input_parent = input_path.parent

    assert isinstance(save_path, (Path, str, type(None))), (
        f"save_path should be a pathlib.Path or string, not: {type(save_path)}"
    )
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
    ops = {**ops_default, **ops_user, "data_path": str(input_path.resolve())}

    file = mbo.imread(input_path)
    if hasattr(file, "metadata"):
        metadata = file.metadata  # noqa
    else:
        metadata = mbo.get_metadata(input_path)

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
    ops["save_path"] = str(plane_dir.resolve())

    needs_detect = False
    if force_detect:
        print(f"Roi detection forced for plane {plane}.")
        needs_detect = True
    elif ops["roidetect"]:
        if (plane_dir / "stat.npy").is_file():
            # make sure this is a valid stat.npy file
            stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
            if stat is None or len(stat) == 0:
                print(f"Detected empty stat.npy, forcing roi detection for plane {plane}.")
                needs_detect = True
            else:
                print(f"Roi detection skipped, stat.npy already exists for plane {plane}.")
                needs_detect = False
    elif (plane_dir / "stat.npy").is_file():
        # check contents of stat.npy
        stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
        if stat is None or len(stat) == 0:
            print(f"Detected empty stat.npy, forcing roi detection for plane {plane}.")
            needs_detect = True
        else:
            print(f"Roi detection skipped, stat.npy already exists for plane {plane}.")
            needs_detect = True

    ops_file = plane_dir / "ops.npy"
    reg_data_file = plane_dir / "data.bin"
    reg_data_file_tiff = plane_dir / "reg_tif"

    if _should_write_bin(ops_file, force=kwargs.get("force_save", False)):
        md_combined = {**metadata, **ops}
        mbo.imwrite(
            file,
            plane_dir,
            ext=".bin",
            metadata=md_combined,
            register_z=False
        )
    else:
        print(f"Skipping data_raw.bin write, already exists and passes data validation checks.")

    ops_outpath = (
        np.load(ops_file, allow_pickle=True).item()
        if (plane_dir / "ops.npy").exists()
        else {}
    )

    exists = False
    if reg_data_file.exists():
        exists = True
    if reg_data_file_tiff.exists():
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
        "reg_file": str((plane_dir / "data.bin").resolve()),
    }

    if "nframes" not in ops:
        if "metadata" in ops and "shape" in ops["metadata"]:
            ops["nframes"] = ops["metadata"]["shape"][0]
        elif "num_frames" in metadata:
            ops["nframes"] = metadata["num_frames"]
        elif "nframes" in metadata:
            ops["nframes"] = metadata["nframes"]
        elif file is not None and hasattr(file, "shape") and len(file.shape) >= 1:
            ops["nframes"] = file.shape[0]
        elif "shape" in metadata:
            ops["nframes"] = metadata["shape"][0]
        else:
            raise KeyError(
                "missing frame count (nframes) in ops or metadata, and cannot infer from data"
            )

    processed = run_plane_bin(ops)

    if save_json:
        # convert ops dict to JSON serializable and save as ops.json
        ops_to_json(ops_file)

    if not processed:
        print(f"Skipping {ops_file.name}, processing was not completed.")
        return ops_file

    # cleanup ourselves
    if not keep_raw:
        (plane_dir / "data_raw.bin").unlink(missing_ok=True)
    if not keep_reg:
        (plane_dir / "data.bin").unlink(missing_ok=True)

    try:
        remake_plane_figures(
            plane_dir,
            dff_percentile=dff_percentile,
            dff_window_size=dff_window_size,
            run_rastermap=kwargs.get("run_rastermap", False)
        )
    except Exception:
        traceback.print_exc()
    return ops_file


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

        save_path = save_root / tag
        run_plane(
            input_path=input_file,
            save_path=save_path,
            ops=ops,
            keep_reg=True,
            keep_raw=True,
            force_reg=True,
            force_detect=True,
        )
