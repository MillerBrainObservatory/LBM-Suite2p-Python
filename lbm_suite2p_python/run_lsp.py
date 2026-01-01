import logging
import time
from datetime import datetime
from pathlib import Path
import os
import traceback
from contextlib import nullcontext
from itertools import product
import copy
import gc

import numpy as np

from lbm_suite2p_python.default_ops import default_ops
from lbm_suite2p_python.postprocessing import (
    ops_to_json,
    load_planar_results,
    load_ops,
    dff_rolling_percentile,
    apply_filters,
)

from importlib.metadata import version, PackageNotFoundError

def _get_version():
    try:
        return version("lbm_suite2p_python")
    except PackageNotFoundError:
        return "0.0.0"


from lbm_suite2p_python.zplane import (
    save_pc_panels_and_metrics,
    plot_zplane_figures,
    plot_filtered_cells,
    plot_filter_exclusions,
    plot_cell_filter_summary,
)

DEFAULT_CELL_FILTERS = [
    {"name": "max_diameter", "min_diameter_um": 4, "max_diameter_um": 35}
]
from mbo_utilities.log import get as get_logger
from mbo_utilities.metadata import (
    get_param,
    get_voxel_size,
)



logger = get_logger("run_lsp")

from lbm_suite2p_python._benchmarking import get_cpu_percent, get_ram_used
from lbm_suite2p_python.volume import (
    plot_volume_diagnostics,
    plot_orthoslices,
    plot_3d_roi_map,
    get_volume_stats,
)
from mbo_utilities.arrays import (
    iter_rois,
    supports_roi,
    _normalize_planes,
    _build_output_path,
)
from mbo_utilities._writers import _write_plane

PIPELINE_TAGS = ("plane", "roi", "z", "plane_", "roi_", "z_")

from lbm_suite2p_python.utils import _is_lazy_array, _get_num_planes


def _get_suite2p_version():
    """Get suite2p version string."""
    try:
        import suite2p
        return getattr(suite2p, "__version__", "unknown")
    except ImportError:
        return "not installed"


def _add_processing_step(ops, step_name, input_files=None, duration_seconds=None, extra=None):
    """
    Add a processing step to ops["processing_history"].

    Each step is appended to the history list, preserving previous runs.
    This allows tracking of re-runs and incremental processing.

    Parameters
    ----------
    ops : dict
        The ops dictionary to update.
    step_name : str
        Name of the processing step (e.g., "binary_write", "registration", "detection").
    input_files : list of str, optional
        List of input file paths for this step.
    duration_seconds : float, optional
        How long this step took.
    extra : dict, optional
        Additional metadata for this step.

    Returns
    -------
    dict
        The updated ops dictionary.
    """
    if "processing_history" not in ops:
        ops["processing_history"] = []

    step_record = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "lbm_suite2p_python_version": _get_version(),
        "suite2p_version": _get_suite2p_version(),
    }

    if input_files is not None:
        step_record["input_files"] = [str(f) for f in input_files] if not isinstance(input_files, str) else [input_files]

    if duration_seconds is not None:
        step_record["duration_seconds"] = round(duration_seconds, 2)

    if extra is not None:
        step_record.update(extra)

    ops["processing_history"].append(step_record)
    return ops


add_processing_step = _add_processing_step


def pipeline(
    input_data,
    save_path: str | Path = None,
    ops: dict = None,
    planes: list | int = None,
    roi_mode: int = None,
    keep_reg: bool = True,
    keep_raw: bool = False,
    force_reg: bool = False,
    force_detect: bool = False,
    num_timepoints: int = None,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    cell_filters: list = None,
    accept_all_cells: bool = False,
    save_json: bool = False,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
    # deprecated parameters
    roi: int = None,
    num_frames: int = None,
    **kwargs,
) -> list[Path]:
    """
    Unified Suite2p processing pipeline.

    Wrapper around run_volume (for 4D data) and run_plane (for 3D data).
    Automatically detects input type and delegates processing.

    Parameters
    ----------
    input_data : str, Path, list, or lazy array
        Input data source (file, directory, list of files, or array).
    save_path : str or Path, optional
        Output directory.
    ops : dict, optional
        Suite2p parameters.
    planes : int or list, optional
        Planes to process (1-based index).
    roi_mode : int, optional
        ROI mode for ScanImage data (None=stitch, 0=split, N=single).
    keep_reg, keep_raw : bool
        Keep binary files.
    force_reg, force_detect : bool
        Force re-processing.
    num_timepoints : int, optional
        Limit frames.
    dff_window_size, dff_percentile, dff_smooth_window : optional
        dF/F parameters.
    cell_filters : list, optional
        Filters to apply. Default is 4-35um diameter if None. Pass [] to disable.
    accept_all_cells : bool
        Mark all detected ROIs as accepted.
    **kwargs
        Additional args passed to sub-functions.

    Returns
    -------
    list[Path]
        List of paths to produced ops.npy files.
    """
    from mbo_utilities import imread
    from mbo_utilities.arrays import supports_roi

    # 1. Handle Deprecations
    if roi is not None:
        import warnings
        warnings.warn("'roi' is deprecated, use 'roi_mode'", DeprecationWarning, stacklevel=2)
        roi_mode = roi
    if num_frames is not None:
        import warnings
        warnings.warn("'num_frames' is deprecated, use 'num_timepoints'", DeprecationWarning, stacklevel=2)
        num_timepoints = num_frames

    # Normalize kwargs
    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}
    if num_timepoints is not None:
        writer_kwargs["num_frames"] = num_timepoints

    # 2. Load Input to Determine Dimensionality
    # We need to know if it's 3D (single plane) or 4D (volume)
    is_list = isinstance(input_data, (list, tuple))
    
    if is_list:
        # Check if list of files implies volume (files with plane tags)
        # We'll just assume list = volume for now, as run_volume handles lists
        is_volumetric = True
        arr = None
    else:
        # Load array to check dimensions
        # If input is already array, this is fast. If path, it loads lazy array.
        if _is_lazy_array(input_data):
            arr = input_data
        else:
            print(f"Loading input to determine dimensions: {input_data}")
            arr = imread(input_data, **reader_kwargs)
            
        # Apply ROI mode if applicable check dimensions
        if roi_mode is not None and supports_roi(arr):
             arr.roi = roi_mode
        
        # Check dims
        # TZYX (4D) or TYX (3D)
        if arr.ndim == 4:
            is_volumetric = True
        elif arr.ndim == 3:
            is_volumetric = False
            # Check if user asked for multiple planes on a 3D array? 
            # If array is 3D, it's one plane. 
            # Unless it's a stack of planes? But imread usually returns TYX for single plane tiff.
            # If it's a stack of planes (ZYX?), current pipeline assumes Time is first dim.
            # So 3D TYX is one plane.
        else:
             # handle odd cases
             is_volumetric = False

    # 3. Delegate
    if is_volumetric:
        print("Delegating to run_volume (4D input detected)...")
        if arr is not None:
             input_arg = arr
        else:
             input_arg = input_data
             
        return run_volume(
            input_data=input_arg,
            save_path=save_path,
            ops=ops,
            planes=planes,
            keep_reg=keep_reg,
            keep_raw=keep_raw,
            force_reg=force_reg,
            force_detect=force_detect,
            dff_window_size=dff_window_size,
            dff_percentile=dff_percentile,
            dff_smooth_window=dff_smooth_window,
            accept_all_cells=accept_all_cells,
            cell_filters=cell_filters,
            save_json=save_json,
            reader_kwargs=reader_kwargs,
            writer_kwargs=writer_kwargs,
            **kwargs
        )
    else:
        print("Delegating to run_plane (3D input detected)...")
        # run_plane returns a single Path, we wrap in list
        ops_path = run_plane(
            input_data=arr, # Pass the array we loaded (with ROI applied)
            save_path=save_path,
            ops=ops,
            # planes argument is ignored for single plane, or used to validate?
            # run_plane infers using 'plane' in ops/metadata.
            # If user passed 'planes', we should check if they asked for something valid?
            # For 3D input, 'planes' argument implies iterating? But it's 3D...
            # We'll rely on run_plane to extract metadata.
            keep_reg=keep_reg,
            keep_raw=keep_raw,
            force_reg=force_reg,
            force_detect=force_detect,
            dff_window_size=dff_window_size,
            dff_percentile=dff_percentile,
            dff_smooth_window=dff_smooth_window,
            accept_all_cells=accept_all_cells,
            cell_filters=cell_filters,
            save_json=save_json,
            reader_kwargs=reader_kwargs,
            writer_kwargs=writer_kwargs,
            **kwargs
        )
        return [ops_path]


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


def get_plane_num_from_tag(tag: str, fallback: int = None) -> int:
    """
    Extract the plane number from a tag string like "plane3" or "roi7".

    Parameters
    ----------
    tag : str
        A tag string (e.g., "plane3", "roi7", "z10") typically from derive_tag_from_filename.
    fallback : int, optional
        Value to return if no number can be extracted from the tag.

    Returns
    -------
    int
        The extracted plane number, or the fallback value if extraction fails.

    Examples
    --------
    >>> get_plane_num_from_tag("plane3")
    3
    >>> get_plane_num_from_tag("roi7")
    7
    >>> get_plane_num_from_tag("z10")
    10
    >>> get_plane_num_from_tag("assembled_data", fallback=0)
    0
    """
    import re

    match = re.search(r"(\d+)$", tag)
    if match:
        return int(match.group(1))
    return fallback


def run_volume(
    input_data,
    save_path: str | Path = None,
    ops: dict | str | Path = None,
    planes: list | int = None,
    keep_reg: bool = True,
    keep_raw: bool = False,
    force_reg: bool = False,
    force_detect: bool = False,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    accept_all_cells: bool = False,
    cell_filters: list = None,
    save_json: bool = False,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
    **kwargs,
):
    """
    Processes a full volumetric imaging dataset using Suite2p.

    Iterates over 3D planes (z-stacks) within the volume, calling run_plane() for each.
    Aggregates results into volume_stats.npy and generates volumetric plots.

    Parameters
    ----------
    input_data : list, Path, or lazy array
        Input data source.
        - List of paths: [plane1.tif, plane2.tif, ...]
        - Lazy array: 4D array (Time, Z, Y, X)
    save_path : str or Path, optional
        Base directory to save outputs.
    ops : dict, optional
        Suite2p parameters.
    planes : list or int, optional
        Specific planes to process (1-based index).
    keep_reg : bool, default True
        Keep registered binaries.
    keep_raw : bool, default False
        Keep raw binaries.
    force_reg : bool, default False
        Force re-registration.
    force_detect : bool, default False
        Force detection.
    dff_window_size, dff_percentile, dff_smooth_window : optional
        dF/F calculation parameters.
    accept_all_cells : bool, default False
        Mark all ROIs as accepted.
    cell_filters : list, optional
        Filters to apply (see run_plane).
    save_json : bool, default False
        Save ops as JSON.
    **kwargs
        Additional args passed to run_plane.

    Returns
    -------
    list[Path]
        List of paths to ops.npy files for processed planes.
    """
    from mbo_utilities.arrays import _normalize_planes
    from mbo_utilities import imread
    from lbm_suite2p_python.merging import merge_mrois

    # Handle input data
    input_arr = None
    input_paths = []

    if _is_lazy_array(input_data):
        input_arr = input_data
        if hasattr(input_arr, "filenames"):
             # For lazy arrays backed by files, use the first file's parent as default save_path
             if save_path is None and input_arr.filenames:
                 save_path = Path(input_arr.filenames[0]).parent / "suite2p_results"
    elif isinstance(input_data, (list, tuple)):
        input_paths = [Path(p) for p in input_data]
        if save_path is None and input_paths:
             save_path = input_paths[0].parent
    elif isinstance(input_data, (str, Path)):
        # Single path representing a volume (e.g. 4D tiff, zarr)
        # We'll load it as an array to iterate planes
        input_path = Path(input_data)
        if save_path is None:
             save_path = input_path.parent / (input_path.stem + "_results")
        
        # Load as array to determine planes
        input_arr = imread(input_path, **(reader_kwargs or {}))
    else:
        raise TypeError(f"Invalid input_data type: {type(input_data)}")

    if save_path is None:
         raise ValueError("save_path must be specified.")
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Determine num_planes
    if input_arr is not None:
        try:
             num_planes = _get_num_planes(input_arr)
        except:
             num_planes = 1
    else:
        num_planes = len(input_paths)

    # Normalize planes to process
    planes_indices = _normalize_planes(planes, num_planes)
    
    print(f"Processing {len(planes_indices)} planes in volume (Total planes: {num_planes})")
    print(f"Output: {save_path}")

    ops_files = []
    
    # Iterate
    for i, plane_idx in enumerate(planes_indices):
        plane_num = plane_idx + 1
        
        # Prepare input for run_plane
        if input_arr is not None:
            # Pass the whole array, run_plane handles extraction via ops['plane']
            current_input = input_arr
        else:
            # List of files - map plane_num to file index (assuming 1-to-1 if no explicit mapping)
            # If input_paths corresponds to ALL planes, then plane_idx indexes into it
            if plane_idx < len(input_paths):
                current_input = input_paths[plane_idx]
            else:
                # Fallback or error? Assuming input_files length matches num_planes
                 current_input = input_paths[0] # Should not happen if logic is correct
        
        # Prepare ops with plane number
        current_ops = load_ops(ops) if ops else default_ops()
        current_ops["plane"] = plane_num
        current_ops["num_zplanes"] = num_planes # useful info

        # Call run_plane
        try:
            print(f"\n--- Volume Step: Plane {plane_num} ---")
            ops_file = run_plane(
                input_data=current_input,
                save_path=save_path,
                ops=current_ops,
                keep_reg=keep_reg,
                keep_raw=keep_raw,
                force_reg=force_reg,
                force_detect=force_detect,
                dff_window_size=dff_window_size,
                dff_percentile=dff_percentile,
                dff_smooth_window=dff_smooth_window,
                accept_all_cells=accept_all_cells,
                cell_filters=cell_filters,
                save_json=save_json,
                reader_kwargs=reader_kwargs,
                writer_kwargs=writer_kwargs,
                **kwargs
            )
            ops_files.append(ops_file)
        except Exception as e:
            print(f"ERROR processing plane {plane_num}: {e}")
            traceback.print_exc()

    # Post-Loop: Merging and Volume Stats
    
    # Check for multi-ROI merging
    # Minimal check: if input filenames have "roi", try merging
    should_merge = False
    if input_paths:
         if any("roi" in p.stem.lower() for p in input_paths):
             should_merge = True
    elif input_arr is not None and hasattr(input_arr, "filenames"):
         if any("roi" in str(p).lower() for p in input_arr.filenames):
             should_merge = True
             
    if should_merge:
        print("Detected mROI data, attempting to merge...")
        merged_savepath = save_path / "merged_mrois"
        try:
             merge_mrois(save_path, merged_savepath)
             # Update ops_files to point to merged results
             # We need to find the new ops files
             # mbo_utilities check_and_merge_mrois returned list, but we don't have it.
             # We'll glob the new directory
             merged_ops = sorted(list(merged_savepath.glob("**/ops.npy")))
             if merged_ops:
                 ops_files = merged_ops
                 print(f"Merged {len(ops_files)} planes to {merged_savepath}")
                 save_path = merged_savepath # Update save_path for subsequent plots
        except Exception as e:
             print(f"Merging failed: {e}")

    # Generate volume statistics
    if ops_files:
        print("\nGenering volumetric statistics...")
        try:
             # run_plane already does individual calls, but we need aggregate stats
             stats_path = get_volume_stats(ops_files, overwrite=True)
             
             # Volumetric plots
             try:
                 plot_volume_diagnostics(ops_files, save_path / "volume_quality_diagnostics.png")
                 plot_orthoslices(ops_files, save_path / "orthoslices.png")
                 plot_3d_roi_map(ops_files, save_path / "roi_map_3d.png", color_by="snr")
                 plot_3d_roi_map(ops_files, save_path / "roi_map_3d_plane.png", color_by="plane")
             except Exception as e:
                  print(f"Warning: Volume plots failed: {e}")
                  traceback.print_exc()
                  
        except Exception as e:
             print(f"Warning: Volume statistics failed: {e}")
             traceback.print_exc()

    return ops_files


def _should_write_bin(ops_path: Path, force: bool = False, *, validate_chan2: bool | None = None, expected_dtype: np.dtype = np.int16) -> bool:
    if force:
        return True
    ops_path = Path(ops_path)
    if not ops_path.is_file():
        return True
    raw_path = ops_path.parent / "data_raw.bin"
    reg_path = ops_path.parent / "data.bin"
    chan2_path = ops_path.parent / "data_chan2.bin"

    # If neither raw nor registered binary exists, need to write
    if not raw_path.is_file() and not reg_path.is_file():
        return True

    # Use whichever binary exists for validation (prefer raw)
    binary_to_validate = raw_path if raw_path.is_file() else reg_path
    try:
        ops = np.load(ops_path, allow_pickle=True).item()
        if validate_chan2 is None:
            validate_chan2 = (ops.get("align_by_chan", 1) == 2)
        Ly = ops.get("Ly")
        Lx = ops.get("Lx")
        nframes_raw = ops.get("nframes_chan1") or ops.get("nframes") or ops.get("num_frames")
        if (None in (nframes_raw, Ly, Lx)) or (nframes_raw <= 0 or Ly <= 0 or Lx <= 0):
            return True
        expected_size_raw = int(nframes_raw) * int(Ly) * int(Lx) * np.dtype(expected_dtype).itemsize
        actual_size_raw = binary_to_validate.stat().st_size
        if actual_size_raw != expected_size_raw or actual_size_raw == 0:
            return True
        try:
            arr = np.memmap(binary_to_validate, dtype=expected_dtype, mode="r", shape=(int(nframes_raw), int(Ly), int(Lx)))
            _ = arr[0, 0, 0]
            del arr
        except Exception:
            return True
        if validate_chan2:
            nframes_chan2 = ops.get("nframes_chan2")
            if (not chan2_path.is_file()) or (nframes_chan2 is None) or (nframes_chan2 <= 0):
                return True
            expected_size_chan2 = int(nframes_chan2) * int(Ly) * int(Lx) * np.dtype(expected_dtype).itemsize
            actual_size_chan2 = chan2_path.stat().st_size
            if actual_size_chan2 != expected_size_chan2 or actual_size_chan2 == 0:
                return True
            try:
                arr2 = np.memmap(chan2_path, dtype=expected_dtype, mode="r", shape=(int(nframes_chan2), int(Ly), int(Lx)))
                _ = arr2[0, 0, 0]
                del arr2
            except Exception:
                return True
        return False
    except Exception as e:
        print(f"Bin validation failed for {ops_path.parent}: {e}")
        return True


def _should_register(ops_path: str | Path) -> bool:
    """
    Determine whether Suite2p registration still needs to be performed.

    Registration is considered complete if any of the following hold:
      - A reference image (refImg) exists and is a valid ndarray
      - meanImg exists (Suite2p always produces it post-registration)
      - Valid registration offsets (xoff/yoff) are present

    Returns True if registration *should* be run, False otherwise.
    """
    ops = load_ops(ops_path)

    has_ref = isinstance(ops.get("refImg"), np.ndarray)
    has_mean = isinstance(ops.get("meanImg"), np.ndarray)

    # Check for valid offsets - ensure they are actual arrays, not _NoValue or other sentinels
    def _has_valid_offsets(key):
        val = ops.get(key)
        if val is None or not isinstance(val, np.ndarray):
            return False
        try:
            return np.any(np.isfinite(val))
        except (TypeError, ValueError):
            return False

    has_offsets = _has_valid_offsets("xoff") or _has_valid_offsets("yoff")
    has_metrics = any(k in ops for k in ("regDX", "regPC", "regPC1", "regDX1"))

    # registration done if any of these are true
    registration_done = has_ref or has_mean or has_offsets or has_metrics
    return not registration_done


def run_plane_bin(ops) -> bool:
    """
    Run Suite2p pipeline on pre-written binary files.

    Executes registration, cell detection, and signal extraction on binary
    data referenced in ops. Requires data_raw.bin and ops.npy to exist.

    Parameters
    ----------
    ops : dict or str or Path
        Suite2p ops dictionary or path to ops.npy file.

    Returns
    -------
    bool
        True if pipeline completed successfully.
    """
    from contextlib import nullcontext
    from suite2p.io.binary import BinaryFile
    from suite2p.run_s2p import pipeline

    ops = load_ops(ops)

    # Get Ly and Lx with helpful error message if missing
    Ly = ops.get("Ly")
    Lx = ops.get("Lx")
    if Ly is None or Lx is None:
        raise KeyError(
            f"Missing required dimension keys in ops: Ly={Ly}, Lx={Lx}. "
            f"Ensure the binary was written with proper metadata. "
            f"Available keys: {list(ops.keys())}"
        )
    Ly, Lx = int(Ly), int(Lx)

    raw_file = ops.get("raw_file")
    n_func = ops.get("nframes_chan1") or ops.get("nframes") or ops.get("n_frames")
    if raw_file is None or n_func is None:
        raise KeyError("Missing raw_file or nframes_chan1")
    n_func = int(n_func)

    ops_parent = Path(ops["ops_path"]).parent
    ops["save_path"] = ops_parent

    reg_file = ops_parent / "data.bin"
    ops["reg_file"] = str(reg_file)

    chan2_file = ops.get("chan2_file", "")
    use_chan2 = bool(chan2_file) and Path(chan2_file).exists()
    n_chan2 = int(ops.get("nframes_chan2", 0)) if use_chan2 else 0

    n_align = n_func if not use_chan2 else min(n_func, n_chan2)
    if n_align <= 0:
        raise ValueError("Non-positive frame count after alignment selection.")
    if use_chan2 and (n_func != n_chan2):
        print(f"[run_plane_bin] Trimming to {n_align} frames (func={n_func}, chan2={n_chan2}).")

    ops["functional_chan"] = 1
    ops["align_by_chan"] = 2 if use_chan2 else 1
    ops["nchannels"] = 2 if use_chan2 else 1
    ops["nframes"] = n_align
    ops["nframes_chan1"] = n_align
    if use_chan2:
        ops["nframes_chan2"] = n_align

    if "diameter" in ops:
        # save user's input diameter before suite2p/cellpose overwrites it
        # cellpose estimates actual cell diameters and saves median to ops["diameter"]
        ops["diameter_user"] = ops["diameter"]
        if ops["diameter"] is not None and np.isnan(ops["diameter"]):
            ops["diameter"] = 8
            ops["diameter_user"] = 8
        if (ops["diameter"] in (None, 0)) and ops.get("anatomical_only", 0) > 0:
            ops["diameter"] = 8
            ops["diameter_user"] = 8
            print("Warning: diameter was not set, defaulting to 8.")

    # When running registration, reset detection-derived parameters so that
    # compute_enhanced_mean_image() will reinitialize them from diameter.
    # This fixes a bug where re-running registration on previously processed data
    # would inherit spatscale_pix=0 from a failed Cellpose detection, causing
    # meanImgE to be computed with a [1,1] filter (all 0.5 output).
    run_registration = bool(ops.get("do_registration", True))
    if run_registration:
        for key in ["spatscale_pix", "Vcorr", "Vmax", "Vmap", "Vsplit", "ihop"]:
            if key in ops:
                del ops[key]

    reg_file_chan2 = ops_parent / "data_chan2_reg.bin" if use_chan2 else None

    ops["anatomical_red"] = False
    ops["chan2_thres"] = 0.1

    # Memory estimation warning for large datasets
    if ops.get("roidetect", True) and ops.get("anatomical_only", 0) > 0:
        # Estimate memory usage for Cellpose detection
        estimated_gb = (Ly * Lx * n_align * 2) / 1e9  # Rough estimate
        spatial_scale = ops.get("spatial_scale", 0)
        if spatial_scale > 0:
            estimated_gb /= (spatial_scale ** 2)

        if estimated_gb > 50:  # Warn for datasets > 50GB
            print(f"Large dataset warning: {estimated_gb:.1f} GB estimated for detection")
            if spatial_scale == 0:
                print(f"  Consider adding 'spatial_scale': 2 to reduce memory usage by 4x")
            print(f"  Or reduce 'batch_size' (current: {ops.get('batch_size', 500)})")

    # When skipping registration, copy data_raw.bin to data.bin and detect valid region
    if not run_registration:
        print("Registration skipped - copying data_raw.bin to data.bin...")
        import shutil
        raw_file_path = Path(raw_file)
        reg_file_path = Path(reg_file)

        # Copy data_raw.bin to data.bin if it doesn't exist or is empty
        if raw_file_path.exists():
            if not reg_file_path.exists() or reg_file_path.stat().st_size == 0:
                print(f"  Copying {raw_file_path.name} -> {reg_file_path.name}")
                shutil.copy2(raw_file_path, reg_file_path)
            else:
                print(f"  {reg_file_path.name} already exists, skipping copy")

            # Detect valid region (exclude dead zones from Suite3D shifts)
            # This replicates what Suite2p's registration does via compute_crop()
            # IMPORTANT: Skip auto-detection for anatomical_only mode since Cellpose
            # returns masks in full image coordinates, not cropped coordinates
            use_anatomical = ops.get("anatomical_only", 0) > 0
            if "yrange" not in ops or "xrange" not in ops:
                if use_anatomical:
                    # For anatomical detection, always use full image to avoid coordinate mismatch
                    print("  Using full image dimensions for anatomical detection (avoids cropping issues)")
                    ops["yrange"] = [0, Ly]
                    ops["xrange"] = [0, Lx]
                else:
                    print("  Detecting valid region to exclude dead zones...")
                    with BinaryFile(Ly=Ly, Lx=Lx, filename=str(raw_file_path)) as f:
                        meanImg_full = f.sampled_mean().astype(np.float32)

                        # Find regions with valid data (threshold at 1% of max)
                        threshold = meanImg_full.max() * 0.01
                        valid_mask = meanImg_full > threshold
                        valid_rows = np.any(valid_mask, axis=1)
                        valid_cols = np.any(valid_mask, axis=0)

                        if valid_rows.sum() > 0 and valid_cols.sum() > 0:
                            y_indices = np.where(valid_rows)[0]
                            x_indices = np.where(valid_cols)[0]
                            yrange = [int(y_indices[0]), int(y_indices[-1] + 1)]
                            xrange = [int(x_indices[0]), int(x_indices[-1] + 1)]
                        else:
                            yrange = [0, Ly]
                            xrange = [0, Lx]

                        ops["yrange"] = yrange
                        ops["xrange"] = xrange
                        print(f"  Valid region: yrange={yrange}, xrange={xrange}")

            # Set registration outputs that detection expects
            if "badframes" not in ops:
                ops["badframes"] = np.zeros(n_align, dtype=bool)
            if "xoff" not in ops:
                ops["xoff"] = np.zeros(n_align, dtype=np.float32)
            if "yoff" not in ops:
                ops["yoff"] = np.zeros(n_align, dtype=np.float32)
            if "corrXY" not in ops:
                ops["corrXY"] = np.ones(n_align, dtype=np.float32)

        # Also copy channel 2 if it exists
        if use_chan2:
            chan2_path = Path(chan2_file)
            reg_chan2_path = Path(reg_file_chan2)
            if chan2_path.exists():
                if not reg_chan2_path.exists() or reg_chan2_path.stat().st_size == 0:
                    print(f"  Copying {chan2_path.name} -> {reg_chan2_path.name}")
                    shutil.copy2(chan2_path, reg_chan2_path)
                else:
                    print(f"  {reg_chan2_path.name} already exists, skipping copy")

    with (
        BinaryFile(Ly=Ly, Lx=Lx, filename=str(reg_file), n_frames=n_align) as f_reg,
        BinaryFile(Ly=Ly, Lx=Lx, filename=str(raw_file), n_frames=n_align) as f_raw,
        (BinaryFile(Ly=Ly, Lx=Lx, filename=str(reg_file_chan2), n_frames=n_align) if use_chan2 else nullcontext()) as f_reg_chan2,
        (BinaryFile(Ly=Ly, Lx=Lx, filename=str(chan2_file), n_frames=n_align) if use_chan2 else nullcontext()) as f_raw_chan2,
    ):
        ops = pipeline(
            f_reg=f_reg,
            f_raw=f_raw,
            f_reg_chan2=f_reg_chan2 if use_chan2 else None,
            f_raw_chan2=f_raw_chan2 if use_chan2 else None,
            run_registration=run_registration,
            ops=ops,
            stat=None,
        )

    if use_chan2:
        ops["reg_file_chan2"] = str(reg_file_chan2)
    np.save(ops["ops_path"], ops)
    return True


def run_plane(
    input_data,
    save_path: str | Path | None = None,
    ops: dict | str | Path = None,
    chan2_file: str | Path | None = None,
    keep_raw: bool = False,
    keep_reg: bool = True,
    force_reg: bool = False,
    force_detect: bool = False,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    accept_all_cells: bool = False,
    cell_filters: list = None,
    save_json: bool = False,
    plane_name: str | None = None,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
    **kwargs,
) -> Path:
    """
    Processes a single imaging plane using suite2p.

    Handles registration, segmentation, filtering, dF/F calculation, and plotting.
    Now accepts both file paths and lazy arrays.

    Parameters
    ----------
    input_data : str, Path, or lazy array
        Input data. Can be a file path or an mbo_utilities lazy array.
    save_path : str or Path, optional
        Root directory to save the results. A subdirectory will be created based on
        the input filename or `plane_name` parameter.
    ops : dict, str or Path, optional
        Path to or dict of user‐supplied ops.npy.
    chan2_file : str, optional
        Path to structural / anatomical data used for registration.
    keep_raw : bool, default False
        If True, do not delete the raw binary (`data_raw.bin`) after processing.
    keep_reg : bool, default True
        If True, keep the registered binary (`data.bin`) after processing.
    force_reg : bool, default False
        If True, force a new registration.
    force_detect : bool, default False
        If True, force ROI detection.
    dff_window_size : int, optional
        Frames for rolling percentile baseline. Default: auto-calculated (~10*tau*fs).
    dff_percentile : int, default 20
        Percentile for baseline F0.
    dff_smooth_window : int, optional
        Smoothing window for dF/F. Default: auto-calculated.
    accept_all_cells : bool, default False
        If True, mark all detected ROIs as accepted cells.
    cell_filters : list[dict], optional
        Filters to apply to detected ROIs (e.g. diameter, area).
        Default: [{"name": "max_diameter", "min_diameter_um": 4, "max_diameter_um": 35}]
        Pass [] to disable default filtering.
    save_json : bool, default False
        Save ops as JSON.
    plane_name : str, optional
        Custom name for the plane subdirectory.
    reader_kwargs : dict, optional
        Arguments for mbo_utilities.imread.
    writer_kwargs : dict, optional
        Arguments for binary writing.
    **kwargs : dict
        Additional arguments passed to Suite2p.

    Returns
    -------
    Path
        Path to the saved ops.npy file.
    """
    from mbo_utilities import imread, imwrite
    from mbo_utilities.metadata import get_metadata
    from mbo_utilities.arrays import ScanImageArray

    if "debug" in kwargs:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled.")

    # Validate cell_filters (use default if None)
    if cell_filters is None:
        cell_filters = DEFAULT_CELL_FILTERS

    # Handle input_data type
    input_path = None
    input_arr = None

    if _is_lazy_array(input_data):
        input_arr = input_data
        # Try to get a filename for naming purposes, otherwise require plane_name
        filenames = getattr(input_arr, "filenames", [])
        if filenames:
            input_path = Path(filenames[0])
        elif plane_name is None:
            raise ValueError("plane_name is required when input is an array without filenames.")
        else:
            # dummy path for internal logic compatibility
            input_path = Path(f"{plane_name}.tif") 
    elif isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
    else:
        raise TypeError(f"input_data must be path or lazy array, got {type(input_data)}")

    input_parent = input_path.parent

    # Save path handling
    if save_path is None:
        if input_arr is not None:
             raise ValueError("save_path is required when input is an array.")
        
        # binary inputs with ops.npy are processed in-place
        is_binary_input = input_path.suffix == ".bin"
        binary_with_ops = is_binary_input and (input_path.parent / "ops.npy").exists()
        if binary_with_ops:
            save_path = input_parent
        else:
            save_path = input_parent
    else:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    # Directory setup
    skip_imwrite = False
    is_binary_input = input_path.suffix == ".bin"
    binary_with_ops = is_binary_input and (input_path.parent / "ops.npy").exists()

    if binary_with_ops and (input_path.parent == save_path or save_path == input_parent):
         # Processing existing binary in-place
         print(f"Processing existing binary in-place: {input_path}")
         plane_dir = input_path.parent
         skip_imwrite = True
    else:
        # Create subdirectory
         if plane_name is not None:
             subdir_name = plane_name
         else:
             subdir_name = derive_tag_from_filename(input_path.name)
         plane_dir = save_path / subdir_name
         plane_dir.mkdir(exist_ok=True)

    ops_file = plane_dir / "ops.npy"
    ops_default = default_ops()
    ops_user = load_ops(ops) if ops else {}
    ops = {**ops_default, **ops_user, "data_path": str(input_path.resolve())}

    # Normalize kwargs
    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}

    # Handle binary writing vs loading
    if skip_imwrite:
        file = None
        existing_ops = np.load(ops_file, allow_pickle=True).item() if ops_file.exists() else {}
        metadata = {k: v for k, v in existing_ops.items() if k in ("plane", "fs", "dx", "dy", "Ly", "Lx", "nframes")}
    else:
        should_write = _should_write_bin(ops_file, force=force_reg)
        if should_write:
            if input_arr is not None:
                file = input_arr
            else:
                file = imread(input_path, **reader_kwargs)
                if isinstance(file, ScanImageArray):
                    if not kwargs.get("allow_raw_array", False):
                         # If it's a raw array (ScanImage), we usually process it via pipeline or run_volume
                         # but run_plane can handle it if it's already a single plane slice (3D)
                         pass
            
            if hasattr(file, "metadata"):
                metadata = file.metadata
            else:
                metadata = get_metadata(input_path)
        else:
            file = None
            metadata = {} # Metadata loading from ops.npy handled later if file is None

    # Plane number handling
    if "plane" in ops:
        plane = ops["plane"]
    elif "plane" in metadata:
        plane = metadata["plane"]
        ops["plane"] = plane
    else:
        tag = derive_tag_from_filename(input_path)
        plane = get_plane_num_from_tag(tag, fallback=ops.get("plane", None))
        ops["plane"] = plane
    
    metadata["plane"] = plane
    ops["save_path"] = str(plane_dir.resolve())

    # Write binary
    if not skip_imwrite and file is not None:
        md_combined = {**metadata, **ops}
        print(f"Writing binary to {plane_dir}...")
        bin_start = time.time()
        imwrite(
            file,
            plane_dir,
            ext=".bin",
            metadata=md_combined,
            register_z=False,
            output_name="data_raw.bin",
            overwrite=True,
            **writer_kwargs,
        )
        # Record binary write
        # Reload ops from disk to get Lx, Ly, and other metadata added by imwrite
        if ops_file.exists():
            ops = np.load(ops_file, allow_pickle=True).item()
        _add_processing_step(
             ops,
             "binary_write",
             input_files=[str(input_path)],
             duration_seconds=time.time() - bin_start,
             extra={"plane": plane, "shape": list(file.shape)}
        )
        np.save(ops_file, ops)

    # Determine processing needs
    needs_detect = False
    if force_detect:
        needs_detect = True
    elif ops["roidetect"]:
        stat_file = plane_dir / "stat.npy"
        if stat_file.exists():
             stat = np.load(stat_file, allow_pickle=True)
             if stat is None or len(stat) == 0:
                 needs_detect = True
             else:
                 needs_detect = False
        else:
             needs_detect = True
    
    # Check registration needs
    if force_reg:
        needs_reg = True
    elif not ops_file.exists():
        needs_reg = True
    else:
        needs_reg = _should_register(ops_file)
    
    # Update ops logic
    if force_reg:
        ops["do_registration"] = 1
    elif not needs_reg:
        ops["do_registration"] = 0
    elif "do_registration" not in ops_user:
        ops["do_registration"] = 1
        
    if force_detect:
        ops["roidetect"] = 1
    elif "roidetect" not in ops_user:
        ops["roidetect"] = int(needs_detect)

    # Channel 2 handling
    if chan2_file is not None:
        chan2_path = Path(chan2_file)
        if chan2_path.exists():
             chan2_data = imread(chan2_path, **reader_kwargs)
             chan2_md = getattr(chan2_data, "metadata", {})
             imwrite(chan2_data, plane_dir, ext=".bin", metadata=chan2_md, register_z=False, structural=True)
             ops["chan2_file"] = str((plane_dir / "data_chan2.bin").resolve())
             ops["nframes_chan2"] = chan2_data.shape[0] if hasattr(chan2_data, "shape") else 0
             ops["nchannels"] = 2
             ops["align_by_chan"] = 2

    # Run Suite2p
    try:
        s2p_start = time.time()
        processed = run_plane_bin(ops) # This updates ops in-place and saves it
        
        if processed:
             updated_ops = load_ops(ops_file)
             _add_processing_step(
                 updated_ops,
                 "suite2p_pipeline",
                 duration_seconds=time.time() - s2p_start,
                 extra={
                     "do_registration": updated_ops.get("do_registration", 1),
                     "n_cells": len(np.load(plane_dir / "stat.npy", allow_pickle=True)) if (plane_dir / "stat.npy").exists() else 0
                 }
             )
             np.save(ops_file, updated_ops)
             
    except Exception as e:
        print(f"Error in run_plane_bin: {e}")
        traceback.print_exc()
        # Re-raise so caller knows processing failed
        raise

    if not processed:
         return ops_file

    # --- Post-Processing ---

    # 1. Accept All Cells
    if accept_all_cells:
        iscell_file = plane_dir / "iscell.npy"
        if iscell_file.exists():
            iscell = np.load(iscell_file, allow_pickle=True)
            np.save(plane_dir / "iscell_suite2p.npy", iscell)
            iscell[:, 0] = 1
            np.save(iscell_file, iscell)
            print("  Marked all ROIs as accepted.")

    # 2. Cell Filtering
    if cell_filters:
        print(f"  Applying cell filters: {[f['name'] for f in cell_filters]}")
        filter_start = time.time()
        try:
            iscell_original = np.load(plane_dir / "iscell.npy", allow_pickle=True)
            iscell_filtered, removed_mask, filter_results = apply_filters(
                plane_dir=plane_dir,
                filters=cell_filters,
                save=True,
            )
            updated_ops = load_ops(ops_file)
            _add_processing_step(
                updated_ops, 
                "cell_filtering", 
                duration_seconds=time.time() - filter_start,
                extra={"n_removed": int(removed_mask.sum())}
            )
            updated_ops["filter_metadata"] = filter_results
            np.save(ops_file, updated_ops)
            
            # Plots
            try:
                 fig = plot_filtered_cells(
                     plane_dir, 
                     iscell_original, 
                     iscell_filtered, 
                     save_path=plane_dir / "13_filtered_cells.png"
                 )
                 import matplotlib.pyplot as plt
                 plt.close(fig)
                 plot_filter_exclusions(plane_dir, iscell_filtered, filter_results, save_dir=plane_dir)
                 plot_cell_filter_summary(plane_dir, save_path=plane_dir / "15_filter_summary.png")
            except Exception as e:
                 print(f"  Warning: Filter plots failed: {e}")

        except Exception as e:
            print(f"  Warning: Cell filtering failed: {e}")

    # 3. dF/F Calculation
    F_file = plane_dir / "F.npy"
    Fneu_file = plane_dir / "Fneu.npy"
    if F_file.exists() and Fneu_file.exists():
        print("  Computing dF/F...")
        dff_start = time.time()
        F = np.load(F_file)
        Fneu = np.load(Fneu_file)
        F_corr = F - 0.7 * Fneu # Fixed neucoeff for now, could be parameter
        
        current_ops = load_ops(ops_file)
        dff = dff_rolling_percentile(
            F_corr,
            window_size=dff_window_size,
            percentile=dff_percentile,
            smooth_window=dff_smooth_window,
            fs=current_ops.get("fs", 30.0),
            tau=current_ops.get("tau", 1.0)
        )
        np.save(plane_dir / "dff.npy", dff)
        
        _add_processing_step(
             current_ops, 
             "dff_calculation", 
             duration_seconds=time.time() - dff_start,
             extra={"percentile": dff_percentile}
        )
        np.save(ops_file, current_ops)

    # 4. Plots and Cleanup
    try:
        plot_zplane_figures(
             plane_dir, 
             dff_percentile=dff_percentile,
             dff_window_size=dff_window_size, 
             dff_smooth_window=dff_smooth_window
        )
    except Exception as e:
        print(f"  Warning: Plot generation failed: {e}")

    if save_json:
        ops_to_json(ops_file)

    if not keep_raw:
         (plane_dir / "data_raw.bin").unlink(missing_ok=True)
    if not keep_reg:
         (plane_dir / "data.bin").unlink(missing_ok=True)

    # PC metrics
    save_pc_panels_and_metrics(ops_file, plane_dir / "pc_metrics")

    return ops_file


def grid_search(*args, **kwargs):
    """Run a grid search over Suite2p parameters. See lbm_suite2p_python.grid_search module."""
    from lbm_suite2p_python.grid_search import grid_search as _grid_search
    return _grid_search(*args, **kwargs)
