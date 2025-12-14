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

from lbm_suite2p_python import default_ops
import lbm_suite2p_python as lsp
from lbm_suite2p_python.postprocessing import (
    ops_to_json,
    load_planar_results,
    load_ops,
    dff_rolling_percentile,
)
from mbo_utilities.log import get as get_logger
from mbo_utilities.metadata import get_param, get_voxel_size

from lbm_suite2p_python.zplane import save_pc_panels_and_metrics, plot_zplane_figures

logger = get_logger("run_lsp")

from lbm_suite2p_python._benchmarking import get_cpu_percent, get_ram_used
from lbm_suite2p_python.volume import (
    plot_volume_signal,
    plot_volume_neuron_counts,
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

# Supported array types from mbo_utilities
_LAZY_ARRAY_TYPES = (
    "MboRawArray",
    "Suite2pArray",
    "MBOTiffArray",
    "TiffArray",
    "ZarrArray",
    "H5Array",
    "NumpyArray",
    "BinArray",
)


def _is_lazy_array(obj):
    """Check if obj is an mbo_utilities lazy array type."""
    return type(obj).__name__ in _LAZY_ARRAY_TYPES


def _get_num_planes_from_array(arr):
    """Get number of z-planes from a lazy array."""
    # Arrays are in TZYX format: (frames, planes, height, width)
    if hasattr(arr, "num_planes"):
        return arr.num_planes
    if hasattr(arr, "num_channels"):
        return arr.num_channels
    shape = arr.shape
    if len(shape) == 4:
        return shape[1]  # Z dimension
    return 1


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
        "lbm_suite2p_python_version": lsp.__version__,
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


def pipeline(
    input_data,
    save_path: str | Path = None,
    ops: dict = None,
    planes: list | int = None,
    roi: int = None,
    keep_reg: bool = True,
    keep_raw: bool = False,
    force_reg: bool = False,
    force_detect: bool = False,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    cell_filters: list = None,
    save_json: bool = False,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
    **kwargs,
) -> list[Path]:
    """
    Unified Suite2p processing pipeline for any input type.

    Uses mbo_utilities.imread() to handle all supported input formats:
    - Raw ScanImage TIFFs (with phase correction and ROI stitching)
    - Processed TIFFs, Zarr, HDF5 files
    - Existing Suite2p binaries (.bin + ops.npy)
    - Directories containing supported files
    - Pre-loaded lazy arrays from mbo_utilities

    Parameters
    ----------
    input_data : str, Path, list, or lazy array
        Input data source. Can be:
        - Path to a file (TIFF, Zarr, HDF5, .bin)
        - Path to a directory containing supported files
        - List of file paths (one per plane for volumetric data)
        - An mbo_utilities lazy array (MboRawArray, Suite2pArray, etc.)
    save_path : str or Path, optional
        Output directory for results. If None:
        - For file inputs: uses parent directory of input
        - For array inputs: raises ValueError (must be specified)
    ops : dict, optional
        Suite2p parameters. If None, uses default_ops() with metadata
        auto-populated from the input data (frame rate, pixel size, etc.).
    planes : int or list, optional
        Which z-planes to process (1-indexed). Options:
        - None: Process all planes (default)
        - int: Process single plane (e.g., planes=7)
        - list: Process specific planes (e.g., planes=[1, 5, 10])
    roi : int, optional
        ROI handling for multi-ROI ScanImage data:
        - None: Stitch all ROIs horizontally into single FOV (default)
        - 0: Process each ROI separately (creates separate outputs)
        - N > 0: Process only ROI N (1-indexed)
    keep_reg : bool, default True
        Keep registered binary (data.bin) after processing.
    keep_raw : bool, default False
        Keep raw binary (data_raw.bin) after processing.
    force_reg : bool, default False
        Force re-registration even if already complete.
    force_detect : bool, default False
        Force ROI detection even if stat.npy exists.
    dff_window_size : int, optional
        Window size for rolling percentile ΔF/F baseline (in frames).
        If None, auto-calculated as ~10 × tau × fs.
    dff_percentile : int, default 20
        Percentile for baseline F₀ estimation.
    dff_smooth_window : int, optional
        Temporal smoothing window for dF/F traces (in frames).
        If None, auto-calculated as ~0.5 × tau × fs to ensure the window
        spans the calcium indicator decay time. Set to 1 to disable.
    cell_filters : list of dict, optional
        List of cell filters to apply after detection. Each dict must have:
        - 'name': str - filter name ('max_diameter', 'area', 'eccentricity', 'diameter')
        - Additional keys are passed as kwargs to the filter function.

        Available filters and their parameters:
        - 'max_diameter': max_diameter_um, max_diameter_px, min_diameter_um, min_diameter_px
        - 'area': min_area_px, max_area_px, min_mult, max_mult
        - 'eccentricity': max_ratio, min_ratio
        - 'diameter': min_mult, max_mult (relative to ops['diameter'])

        **Default behavior**: If cell_filters is None and pixel resolution is available
        in the metadata, a default filter of ``min_diameter_um=4, max_diameter_um=35`` is
        applied automatically. To disable this, pass ``cell_filters=[]`` (empty list).

        Example::

            cell_filters=[
                {"name": "max_diameter", "max_diameter_um": 22},
                {"name": "eccentricity", "max_ratio": 5.0},
            ]

        To disable default filtering::

            cell_filters=[]

    save_json : bool, default False
        Save ops as JSON in addition to .npy.
    reader_kwargs : dict, optional
        Keyword arguments passed to mbo_utilities.imread() when loading data.
        Useful for controlling how raw ScanImage TIFFs are read. Common options:

        - ``fix_phase`` : bool, default True
            Apply phase correction for bidirectional scanning.
        - ``phasecorr_method`` : str, default 'mean'
            Phase correction method ('mean', 'mode', 'median').
        - ``border`` : int, default 3
            Border pixels to ignore during phase estimation.
        - ``use_fft`` : bool, default False
            Use FFT-based subpixel phase correction.
        - ``fft_method`` : str, default '2d'
            FFT method ('1d' or '2d').
        - ``upsample`` : int, default 5
            Upsampling factor for subpixel precision.
        - ``max_offset`` : int, default 4
            Maximum phase offset to search.

    writer_kwargs : dict, optional
        Keyword arguments passed to mbo_utilities when writing binary files.
        Common options:

        - ``output_suffix`` : str, default ""
            Append a string to the output filename.
        - ``target_chunk_mb`` : int, default 100
            Target chunk size in MB for streaming writes.
        - ``progress_callback`` : Callable, optional
            Callback function for progress updates.

    **kwargs
        Additional arguments passed to Suite2p.

    Returns
    -------
    list[Path]
        List of paths to ops.npy files for each processed plane.

    Examples
    --------
    Process a directory of raw ScanImage TIFFs:

    >>> import lbm_suite2p_python as lsp
    >>> results = lsp.pipeline("D:/data/raw_tiffs", save_path="D:/results")

    Process specific planes from a file:

    >>> results = lsp.pipeline("D:/data/volume.zarr", planes=[1, 5, 10])

    Process a pre-loaded array from mbo_utilities (e.g., from GUI):

    >>> import mbo_utilities as mbo
    >>> arr = mbo.imread("D:/data/raw")
    >>> results = lsp.pipeline(arr, save_path="D:/results", roi=0)  # Split ROIs

    Process with custom ops:

    >>> ops = {"diameter": 8, "threshold_scaling": 0.8}
    >>> results = lsp.pipeline("D:/data", ops=ops)

    Control phase correction for raw ScanImage TIFFs:

    >>> results = lsp.pipeline(
    ...     "D:/data/raw",
    ...     reader_kwargs={"fix_phase": True, "use_fft": True},
    ... )

    Disable phase correction (for already-corrected data):

    >>> results = lsp.pipeline(
    ...     "D:/data/raw",
    ...     reader_kwargs={"fix_phase": False},
    ... )

    Notes
    -----
    **Input Type Detection:**

    The function automatically detects input type and handles it appropriately:

    - Raw ScanImage TIFFs: Phase correction applied, multi-ROI stitched/split
    - Processed files: Loaded directly without modification
    - Suite2p binaries: Processed in-place if ops.npy exists
    - Directories: Scanned for supported files

    **Output Structure:**

    For volumetric data (multiple planes)::

        save_path/
        ├── plane01/
        │   ├── ops.npy, stat.npy, F.npy, ...
        │   └── data.bin (if keep_reg=True)
        ├── plane02/
        │   └── ...
        └── volume_stats.npy

    For multi-ROI data with roi=0::

        save_path/
        ├── plane01_roi1/
        ├── plane01_roi2/
        └── merged_mrois/
            └── plane01/

    **Metadata Flow:**

    When ops=None, metadata from the input is used to populate:
    - fs (frame rate)
    - dx, dy (pixel resolution)
    - Ly, Lx (frame dimensions)

    **Parameter Override Precedence:**

    The ``force_reg`` and ``force_detect`` arguments take precedence over
    ``do_registration`` and ``roidetect`` values in the ops dict:

    - ``force_reg=True`` → always register, ignoring ``ops["do_registration"]``
    - ``force_detect=True`` → always detect, ignoring ``ops["roidetect"]``
    - ``force_reg=False`` (default) → skip registration if already complete,
      even if ``ops["do_registration"]=1``

    This allows users to focus on detection parameters without worrying about
    the registration/detection flags in their ops dict.

    See Also
    --------
    run_plane : Lower-level single-plane processing
    run_volume : Process list of files (legacy API)
    grid_search : Parameter optimization
    """
    from mbo_utilities import imread

    start_time = time.time()

    # Normalize kwargs dicts
    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}

    print(f"Loading input data...")

    if _is_lazy_array(input_data):
        arr = input_data
        filenames = getattr(arr, "filenames", [])
        print(f"  Input: {type(arr).__name__} (pre-loaded array)")
        if save_path is None:
            if filenames:
                save_path = Path(filenames[0]).parent / "suite2p_results"
            else:
                raise ValueError(
                    "save_path is required when input_data is a lazy array "
                    "without filenames attribute."
                )
    elif isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
        print(f"  Input: {input_path}")
        arr = imread(input_path, **reader_kwargs)
        print(f"  Loaded as: {type(arr).__name__}")
        filenames = getattr(arr, "filenames", [input_path])
        if save_path is None:
            save_path = input_path.parent if input_path.is_file() else input_path
    elif isinstance(input_data, (list, tuple)):
        # List of paths - could be multiple planes or multiple files
        paths = [Path(p) for p in input_data]
        print(f"  Input: {len(paths)} files")

        # Check if these are per-plane files (volumetric) or files to concatenate
        # If filenames contain plane indicators, treat as volumetric
        has_plane_tags = any(
            any(tag in p.stem.lower() for tag in PIPELINE_TAGS)
            for p in paths
        )

        if has_plane_tags:
            # Volumetric: process each file as a separate plane
            # Fall back to run_volume behavior
            print(f"  Detected {len(paths)} plane files, processing as volume...")
            return run_volume(
                input_files=paths,
                save_path=save_path,
                ops=ops,
                keep_reg=keep_reg,
                keep_raw=keep_raw,
                force_reg=force_reg,
                force_detect=force_detect,
                dff_window_size=dff_window_size,
                dff_percentile=dff_percentile,
                save_json=save_json,
                reader_kwargs=reader_kwargs,
                writer_kwargs=writer_kwargs,
                **kwargs,
            )
        else:
            # Try to load as a single dataset
            arr = imread(paths, **reader_kwargs)
            print(f"  Loaded as: {type(arr).__name__}")
            filenames = paths
            if save_path is None:
                save_path = paths[0].parent
    else:
        raise TypeError(
            f"input_data must be a path, list of paths, or lazy array. "
            f"Got: {type(input_data)}"
        )

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    # === STEP 2: Configure ROI on the lazy array ===
    # This lets the array handle stitching/splitting internally
    if roi is not None and supports_roi(arr):
        arr.roi = roi
        roi_desc = {None: "stitch all", 0: "split all"}.get(roi, f"ROI {roi} only")
        print(f"  ROI mode: {roi_desc}")

    # === STEP 3: Extract metadata and configure ops ===
    metadata = dict(getattr(arr, "metadata", {}) or {})

    # Get dimensions from the array (which now reflects ROI setting)
    num_planes = _get_num_planes_from_array(arr)
    num_frames = arr.shape[0]
    Ly, Lx = arr.shape[-2], arr.shape[-1]

    print(f"\nDataset info:")
    print(f"  Shape: {arr.shape}")
    print(f"  Frames: {num_frames}")
    print(f"  Planes: {num_planes}")
    print(f"  Dimensions: {Ly} x {Lx}")

    # Show MboRawArray-specific settings
    if hasattr(arr, "fix_phase"):
        print(f"  Phase correction: {arr.fix_phase}")
    if hasattr(arr, "use_fft"):
        print(f"  FFT subpixel: {arr.use_fft}")

    fs = get_param(metadata, "fs")
    if fs:
        print(f"  Frame rate: {fs:.2f} Hz")
    if supports_roi(arr):
        print(f"  ROIs: {getattr(arr, 'num_rois', 1)}")

    # Build ops from defaults + metadata
    if ops is None:
        ops = default_ops()
    else:
        base_ops = default_ops()
        base_ops.update(ops)
        ops = base_ops

    # auto-populate from metadata
    if fs:
        ops["fs"] = fs

    # get pixel resolution from metadata using mbo_utilities
    voxel = get_voxel_size(metadata)
    # only set if we got real values (not defaults of 1.0)
    if voxel.dx != 1.0 or voxel.dy != 1.0:
        ops["dx"] = voxel.dx
        ops["dy"] = voxel.dy
        ops["pixel_resolution"] = [voxel.dx, voxel.dy]

    ops["Ly"] = Ly
    ops["Lx"] = Lx
    ops["nframes"] = num_frames

    # Normalize planes to 0-indexed list using mbo_utilities helper
    planes_to_process = _normalize_planes(planes, num_planes)

    print(f"\nProcessing plan:")
    print(f"  Planes: {[p+1 for p in planes_to_process]}")
    print(f"  Output: {save_path}")

    # === STEP 4: Process each plane and ROI combination ===
    all_ops_files = []
    has_multiple_rois = supports_roi(arr) and getattr(arr, "num_rois", 1) > 1

    for plane_idx in planes_to_process:
        plane_num = plane_idx + 1  # 1-indexed for display
        plane_start = time.time()

        print(f"\n{'='*60}")
        print(f"Processing plane {plane_num}/{num_planes}")
        print(f"{'='*60}")

        # Iterate over ROIs using the array's built-in semantics
        for current_roi in iter_rois(arr):
            # Set ROI on array - this changes what __getitem__ returns
            if current_roi is not None and supports_roi(arr):
                arr.roi = current_roi

            # Build output directory using mbo_utilities convention
            bin_file = _build_output_path(
                save_path,
                plane_idx,
                current_roi,
                ext="bin",
                structural=False,
                has_multiple_rois=has_multiple_rois,
                output_suffix=writer_kwargs.get("output_suffix"),
            )
            plane_dir = bin_file.parent
            plane_tag = plane_dir.name

            # Check if already processed
            ops_file = plane_dir / "ops.npy"
            stat_file = plane_dir / "stat.npy"

            if ops_file.exists() and stat_file.exists() and not force_reg and not force_detect:
                print(f"  Skipping {plane_tag}: already complete")
                all_ops_files.append(ops_file)
                # Still regenerate plots for existing results
                try:
                    plot_zplane_figures(
                        plane_dir,
                        dff_percentile=dff_percentile,
                        dff_window_size=dff_window_size,
                        dff_smooth_window=dff_smooth_window,
                    )
                except Exception as e:
                    print(f"  Warning: Plot generation failed: {e}")
                continue

            # Get dimensions for this specific ROI/plane combination
            # The array's __getitem__ will handle phase correction, ROI stitching, etc.
            if num_planes > 1:
                sample_frame = arr[0, plane_idx]
            else:
                sample_frame = arr[0] if arr.ndim == 3 else arr[0, 0]

            plane_Ly, plane_Lx = sample_frame.shape[-2], sample_frame.shape[-1]
            plane_nframes = num_frames

            # Build ops for this plane
            plane_ops = copy.deepcopy(ops)
            plane_ops.update({
                "Ly": plane_Ly,
                "Lx": plane_Lx,
                "nframes": plane_nframes,
                "nframes_chan1": plane_nframes,
                "plane": plane_num,
                "data_path": str(plane_dir),
                "save_path": str(plane_dir),
                "ops_path": str(ops_file),
                "raw_file": str(bin_file),
                "shape": (plane_nframes, plane_Ly, plane_Lx),
            })

            # Write binary using _write_plane - it handles plane extraction internally
            if not bin_file.exists() or force_reg:
                print(f"  Writing binary ({plane_nframes} frames, {plane_Ly}x{plane_Lx})...")
                bin_write_start = time.time()

                # _write_plane uses plane_index to extract the correct z-plane
                # The array's __getitem__ handles ROI stitching and phase correction
                _write_plane(
                    arr,
                    bin_file,
                    overwrite=True,
                    metadata=plane_ops,
                    plane_index=plane_idx if num_planes > 1 else None,
                    **writer_kwargs,
                )

                # Record binary write step
                _add_processing_step(
                    plane_ops,
                    "binary_write",
                    input_files=filenames,
                    duration_seconds=time.time() - bin_write_start,
                    extra={"plane": plane_num, "shape": list(plane_ops["shape"])},
                )
            else:
                print(f"  Binary exists, skipping write")
                # Ensure ops.npy exists even if binary was skipped
                if not ops_file.exists():
                    np.save(ops_file, plane_ops)

            # Run Suite2p pipeline on binary
            try:
                print(f"  Running Suite2p pipeline...")
                s2p_start = time.time()
                run_plane_bin(ops_file)
                all_ops_files.append(ops_file)

                # Reload ops to get updated values from Suite2p, then add history
                updated_ops = load_ops(ops_file)
                _add_processing_step(
                    updated_ops,
                    "suite2p_pipeline",
                    duration_seconds=time.time() - s2p_start,
                    extra={
                        "do_registration": updated_ops.get("do_registration", True),
                        "anatomical_only": updated_ops.get("anatomical_only", 0),
                        "n_cells_detected": len(np.load(plane_dir / "stat.npy", allow_pickle=True)) if (plane_dir / "stat.npy").exists() else 0,
                    },
                )
                np.save(ops_file, updated_ops)

                # apply cell filters
                # default: filter by 4-35µm diameter if pixel resolution available
                from lbm_suite2p_python.postprocessing import apply_filters

                filters_to_apply = cell_filters
                if filters_to_apply is None:
                    # check if pixel resolution is available for default filter
                    current_ops = load_ops(ops_file)
                    voxel = get_voxel_size(current_ops)
                    pixel_size = (voxel.dx + voxel.dy) / 2 if voxel.dx != 1.0 or voxel.dy != 1.0 else None
                    if pixel_size is not None:
                        filters_to_apply = [{"name": "max_diameter", "min_diameter_um": 4, "max_diameter_um": 35}]
                        print(f"  Applying default diameter filter (4-35 µm, pixel_size={pixel_size:.2f} µm/px)")

                if filters_to_apply:
                    print(f"  Applying cell filters...")
                    filter_start = time.time()
                    try:
                        # Load original iscell for comparison plot
                        iscell_original = np.load(plane_dir / "iscell.npy", allow_pickle=True)

                        iscell_filtered, removed_mask, filter_results = apply_filters(
                            plane_dir=plane_dir,
                            filters=filters_to_apply,
                            save=True,
                        )
                        # Record filtering step
                        updated_ops = load_ops(ops_file)
                        _add_processing_step(
                            updated_ops,
                            "cell_filtering",
                            duration_seconds=time.time() - filter_start,
                            extra={
                                "filters": [f["name"] for f in filters_to_apply],
                                "n_removed": int(removed_mask.sum()),
                                "n_remaining": int(iscell_filtered.sum()),
                            },
                        )
                        np.save(ops_file, updated_ops)

                        # Generate filter comparison plot
                        if removed_mask.sum() > 0:
                            try:
                                from lbm_suite2p_python.zplane import plot_filtered_cells
                                fig = plot_filtered_cells(
                                    plane_dir,
                                    iscell_original=iscell_original,
                                    iscell_filtered=iscell_filtered,
                                    save_path=plane_dir / "13_filtered_cells.png",
                                )
                                import matplotlib.pyplot as plt
                                plt.close(fig)
                            except Exception as e:
                                print(f"  Warning: Filter plot failed: {e}")
                    except Exception as e:
                        print(f"  Warning: Cell filtering failed: {e}")

                # Post-processing: dF/F calculation
                print(f"  Computing dF/F...")
                dff_start = time.time()
                F_file = plane_dir / "F.npy"
                Fneu_file = plane_dir / "Fneu.npy"
                dff_file = plane_dir / "dff.npy"

                if F_file.exists() and Fneu_file.exists():
                    F = np.load(F_file)
                    Fneu = np.load(Fneu_file)
                    F_corr = F - 0.7 * Fneu

                    fs_val = ops.get("fs", 30.0)
                    tau = ops.get("tau", 1.0)
                    dff = dff_rolling_percentile(
                        F_corr,
                        window_size=dff_window_size,
                        percentile=dff_percentile,
                        smooth_window=dff_smooth_window,
                        fs=fs_val,
                        tau=tau,
                    )
                    np.save(dff_file, dff)

                    # Record dF/F calculation step
                    updated_ops = load_ops(ops_file)
                    _add_processing_step(
                        updated_ops,
                        "dff_calculation",
                        duration_seconds=time.time() - dff_start,
                        extra={
                            "dff_percentile": dff_percentile,
                            "dff_window_size": dff_window_size,
                            "dff_smooth_window": dff_smooth_window,
                            "neucoeff": 0.7,
                        },
                    )
                    np.save(ops_file, updated_ops)

                # Generate plots
                try:
                    print(f"  Generating plots...")
                    plot_zplane_figures(
                        plane_dir,
                        dff_percentile=dff_percentile,
                        dff_window_size=dff_window_size,
                        dff_smooth_window=dff_smooth_window,
                    )
                except Exception as e:
                    print(f"  Warning: Plot generation failed: {e}")

                # Cleanup binaries if requested
                if not keep_raw and bin_file.exists():
                    bin_file.unlink()
                if not keep_reg:
                    reg_file = plane_dir / "data.bin"
                    if reg_file.exists():
                        reg_file.unlink()

            except Exception as e:
                print(f"  ERROR processing {plane_tag}: {e}")
                traceback.print_exc()

            plane_elapsed = time.time() - plane_start
            print(f"  Completed {plane_tag} in {plane_elapsed:.1f}s")

    # === STEP 5: Generate volumetric outputs if multiple planes ===
    if len(planes_to_process) > 1 and all_ops_files:
        print(f"\n{'='*60}")
        print("Generating volumetric statistics...")
        print(f"{'='*60}")
        try:
            # Use existing functions from volume.py
            volume_stats = get_volume_stats(all_ops_files)
            np.save(save_path / "volume_stats.npy", volume_stats)

            try:
                plot_volume_signal(volume_stats, save_path)
            except Exception as e:
                print(f"  Warning: plot_volume_signal failed: {e}")

            try:
                plot_volume_neuron_counts(volume_stats, save_path)
            except Exception as e:
                print(f"  Warning: plot_volume_neuron_counts failed: {e}")

            try:
                plot_volume_diagnostics(all_ops_files, save_path)
            except Exception as e:
                print(f"  Warning: plot_volume_diagnostics failed: {e}")

        except Exception as e:
            print(f"Warning: Volume output generation failed: {e}")

    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Processed: {len(all_ops_files)} planes")
    print(f"  Time: {total_elapsed:.1f}s")
    print(f"  Output: {save_path}")
    print(f"{'='*60}")

    return all_ops_files


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
    input_files: list,
    save_path: str | Path = None,
    ops: dict | str | Path = None,
    keep_reg: bool = True,
    keep_raw: bool = False,
    force_reg: bool = False,
    force_detect: bool = False,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    save_json: bool = False,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
    **kwargs,
):
    """
    Processes a full volumetric imaging dataset using Suite2p, handling plane-wise registration,
    segmentation, plotting, and aggregation of volumetric statistics and visualizations.

    Supports planar, contiguous .zarr, tiff, suite2p .bin and automatically merges multi-ROI datasets
    acquired with ScanImage's multi-ROI mode.

    Parameters
    ----------
    input_files : list of str or Path
        List of file paths, each representing a single imaging plane. Supported formats:
        - .tif files (e.g., "plane01.tif", "plane02.tif")
        - .bin files from mbo.imwrite (e.g., "plane01_stitched/data_raw.bin")
        - .zarr files (e.g., "plane01_roi01.zarr", "plane01_roi02.zarr")
        For binary inputs, must have accompanying ops.npy in parent directory.
    save_path : str or Path, optional
        Base directory to save all outputs.
        If None, creates a "volume" directory in the parent of the first input file.
        For binary inputs with ops.npy, processing occurs in-place at the parent directory.
    ops : dict or str or Path, optional
        Suite2p parameters to use for each imaging plane. Can be:
        - Dictionary of parameters
        - Path to ops.npy file
        - None (uses defaults from default_ops())
    keep_raw : bool, default False
        If True, do not delete the raw binary (data_raw.bin) after processing.
    keep_reg : bool, default True
        If True, keep the registered binary (data.bin) after processing.
    force_reg : bool, default False
        If True, force re-registration even if refImg/meanImg/xoff exist in ops.npy.
    force_detect : bool, default False
        If True, force ROI detection even if stat.npy exists and is non-empty.
    dff_window_size : int, optional
        Window size for rolling percentile ΔF/F baseline (in frames).
        If None, auto-calculated as ~10 × tau × fs.
    dff_percentile : int, default 20
        Percentile to use for baseline F₀ estimation (e.g., 20 = 20th percentile).
    dff_smooth_window : int, optional
        Temporal smoothing window for dF/F traces (in frames).
        If None, auto-calculated as ~0.5 × tau × fs to ensure the window
        spans the calcium indicator decay time. Set to 1 to disable.
    save_json : bool, default False
        If True, saves ops as JSON in addition to .npy format.
    **kwargs
        Additional keyword arguments passed to run_plane().

    Returns
    -------
    list of Path
        List of paths to ops.npy files for each plane (or merged plane if mROI).

    Notes
    -----
    **Directory Structure:**

    For standard single-ROI data::

        save_path/
        ├── plane01/
        │   ├── ops.npy, stat.npy, F.npy, Fneu.npy, spks.npy, iscell.npy
        │   ├── data.bin (registered binary, if keep_reg=True)
        │   └── [visualization PNGs]
        ├── plane02/
        │   └── ...
        ├── volume_stats.npy          # Per-plane statistics
        ├── mean_volume_signal.png    # Signal strength across planes
        └── rastermap.png             # Clustered activity (if rastermap installed)

    **Multi-ROI Merging:**

    When input filenames contain "roi" (case-insensitive), e.g., "plane01_roi01.tif",
    "plane01_roi02.tif", the pipeline automatically detects multi-ROI acquisition and
    performs horizontal stitching after planar processing::

        save_path/
        ├── plane01_roi01/           # Individual ROI results
        │   └── [Suite2p outputs]
        ├── plane01_roi02/
        │   └── [Suite2p outputs]
        ├── merged_mrois/            # Merged results (used for volumetric stats)
        │   ├── plane01/
        │   │   ├── ops.npy          # Merged ops with Lx = sum of ROI widths
        │   │   ├── stat.npy         # Concatenated ROIs with xpix offsets applied
        │   │   ├── F.npy, spks.npy  # Concatenated traces
        │   │   ├── data.bin         # Horizontally stitched binary
        │   │   └── [merged visualizations]
        │   └── plane02/
        │       └── ...
        └── [volumetric outputs as above]

    The merging process:
    - Groups directories by plane number (e.g., "plane01_roi01", "plane01_roi02" → "plane01")
    - Horizontally concatenates images (refImg, meanImg, meanImgE, max_proj)
    - Adjusts stat["xpix"] and stat["med"] coordinates to account for horizontal offset
    - Concatenates fluorescence traces (F, Fneu, spks) and cell classifications (iscell)
    - Creates stitched binary files by horizontally stacking frames

    **Supported Input Scenarios:**

    1. TIFF files (standard workflow)::

        input_files = ["plane01.tif", "plane02.tif", "plane03.tif"]
        lsp.run_volume(input_files, save_path="outputs")

    2. Binary files from interrupted processing::

        input_files = [
            "plane01_stitched/data_raw.bin",
            "plane02_stitched/data_raw.bin",
        ]
        lsp.run_volume(input_files)  # Processes in-place

    3. Multi-ROI TIFF files (automatic merging)::

        input_files = [
            "plane01_roi01.tif", "plane01_roi02.tif",
            "plane02_roi01.tif", "plane02_roi02.tif",
        ]
        lsp.run_volume(input_files, save_path="outputs")

    4. Mixed input types::

        input_files = [
            "plane01.tif",                      # New TIFF
            "plane02_stitched/data_raw.bin",    # Existing binary
        ]
        lsp.run_volume(input_files, save_path="outputs")

    See Also
    --------
    run_plane : Process a single imaging plane
    run_plane_bin : Process an existing binary file through Suite2p pipeline
    merge_mrois : Manual multi-ROI merging function
    """
    from mbo_utilities.file_io import get_files

    if not input_files:
        raise Exception("No input files given.")
    if isinstance(input_files, (str, Path)):
        input_files = [input_files]

    start = time.time()
    if save_path is None:
        save_path = Path(input_files[0]).parent

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    all_ops = []
    for z, file in enumerate(input_files):
        file_path = Path(file)
        start_file = time.time()

        # Create a fresh kwargs dict for each iteration to avoid cross-contamination
        call_kwargs = dict(kwargs)

        # determine plane name and input file based on input type
        # for binary inputs with ops.npy, run_plane will process in-place
        # for other formats, run_plane creates subdirectory based on plane_name
        if file_path.suffix == ".bin" and file_path.parent.joinpath("ops.npy").exists():
            # binary from mbo.imwrite or previous processing - process in-place
            print(f"Detected existing binary with ops.npy: {file_path}")
            plane_name = file_path.parent.name

            # prefer data_raw.bin if it exists
            if (file_path.parent / "data_raw.bin").exists():
                input_to_process = file_path.parent / "data_raw.bin"
            else:
                input_to_process = file_path

            # for binary inputs, pass the parent as save_path so run_plane processes in-place
            plane_save_path = file_path.parent
        else:
            # tiff, zarr, or other format - derive plane_name from filename
            plane_name = derive_tag_from_filename(file_path.name)
            plane_num = get_plane_num_from_tag(plane_name, fallback=len(all_ops))
            input_to_process = file_path
            plane_save_path = save_path
            call_kwargs["plane"] = plane_num

        # run_plane handles subdirectory creation via plane_name
        try:
            ops_file = run_plane(
                input_path=input_to_process,
                save_path=plane_save_path,
                ops=ops,
                keep_reg=keep_reg,
                keep_raw=keep_raw,
                force_reg=force_reg,
                force_detect=force_detect,
                dff_window_size=dff_window_size,
                dff_percentile=dff_percentile,
                dff_smooth_window=dff_smooth_window,
                save_json=save_json,
                plane_name=plane_name,
                reader_kwargs=reader_kwargs,
                writer_kwargs=writer_kwargs,
                **call_kwargs,
            )
            all_ops.append(ops_file)
            print(f"Completed {file_path.name} -> {ops_file}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
            # Continue with next file rather than failing entire volume
            continue
        finally:
            end_file = time.time()
            print(f"Time for {file}: {(end_file - start_file) / 60:0.1f} min")
            print(f"CPU {get_cpu_percent():4.1f}% | RAM {get_ram_used() / 1024:5.2f} GB")
            gc.collect()

    end = time.time()
    print(f"Total time for volume: {(end - start) / 60:0.1f} min")

    if "roi" in Path(input_files[0]).stem.lower():
        print("Detected mROI data, merging ROIs for each z-plane...")
        from .merging import merge_mrois

        merged_savepath = save_path.joinpath("merged_mrois")
        merge_mrois(save_path, merged_savepath)
        save_path = merged_savepath

        all_ops = sorted(get_files(merged_savepath, "ops.npy", 2))
        print(f"Planes found after merge: {len(all_ops)}")
    else:
        all_ops = sorted(get_files(save_path, "ops.npy", 2))
        print(f"No mROI data detected, planes found: {len(all_ops)}")

    try:
        zstats_file = get_volume_stats(all_ops, overwrite=True)

        if zstats_file is not None:
            # Generate comprehensive volume diagnostics figure (replaces individual plots)
            plot_volume_diagnostics(
                all_ops, os.path.join(save_path, "volume_quality_diagnostics.png")
            )
            # Generate 3D visualizations
            plot_orthoslices(
                all_ops, os.path.join(save_path, "orthoslices.png")
            )
            plot_3d_roi_map(
                all_ops, os.path.join(save_path, "roi_map_3d.png"), color_by="plane"
            )
            plot_3d_roi_map(
                all_ops, os.path.join(save_path, "roi_map_3d_snr.png"), color_by="snr"
            )
        else:
            print("  Skipping volume plots due to missing statistics")

        # Load planar results with error handling
        res_z = []
        for i, ops_path in enumerate(all_ops):
            try:
                res = load_planar_results(ops_path, z_plane=i)
                res_z.append(res)
            except FileNotFoundError as e:
                print(f"Skipping plane {i}: {e}")
            except Exception as e:
                print(f"Error loading plane {i}: {e}")

        if not res_z:
            print("No valid planar results - skipping rastermap")
            raise ValueError("No valid planar results available for rastermap")

        # Check for frame count mismatches across planes
        frame_counts = [res["spks"].shape[1] for res in res_z]
        min_frames = min(frame_counts)
        max_frames = max(frame_counts)

        frames_cropped = 0
        if min_frames != max_frames:
            frames_cropped = max_frames - min_frames
            print(f"WARNING: Frame count mismatch across planes!")
            print(f"  Frame counts range from {min_frames} to {max_frames}")
            print(f"  Cropping all planes to {min_frames} frames ({frames_cropped} frames dropped)")

            # Crop all arrays to minimum frame count
            for res in res_z:
                res["spks"] = res["spks"][:, :min_frames]
                res["F"] = res["F"][:, :min_frames]
                res["Fneu"] = res["Fneu"][:, :min_frames]

        all_spks = np.concatenate([res["spks"] for res in res_z], axis=0)

        # Save frame cropping metadata
        volume_meta = {
            "n_planes": len(res_z),
            "n_frames": min_frames,
            "frames_cropped": frames_cropped,
            "original_frame_counts": frame_counts,
        }
        np.save(os.path.join(save_path, "volume_meta.npy"), volume_meta)

        try:
            from rastermap import Rastermap
            from lbm_suite2p_python.zplane import plot_rastermap
            HAS_RASTERMAP = True
        except ImportError:
            Rastermap = None
            HAS_RASTERMAP = False
            plot_rastermap = None
            print("rastermap not found. Install via: pip install rastermap")
            print("  or: pip install mbo_utilities[rastermap]")
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

        # Generate volumetric quality figures
        print("Generating diagnostic figures...")
        from lbm_suite2p_python.zplane import (
            plot_multiplane_masks,
            plot_plane_quality_metrics,
            plot_trace_analysis,
            create_volume_summary_table,
        )

        # Consolidate data from all planes
        # iscell is (n_rois, 2): [:, 0] is 0/1, [:, 1] is classifier probability
        # Ensure each stat dict has 'iplane' set from the z_plane array
        all_stat_list = []
        for res in res_z:
            z_plane_arr = res["z_plane"]
            for i, s in enumerate(res["stat"]):
                # Add iplane to stat dict if not present
                if "iplane" not in s:
                    s["iplane"] = int(z_plane_arr[i]) if i < len(z_plane_arr) else 0
                all_stat_list.append(s)
        all_stat = np.array(all_stat_list, dtype=object)
        all_iscell = np.vstack([res["iscell"] for res in res_z])  # (total_rois, 2)
        all_F = np.concatenate([res["F"] for res in res_z], axis=0)
        all_Fneu = np.concatenate([res["Fneu"] for res in res_z], axis=0)

        # Get ops from first plane for frame rate
        first_ops = load_ops(all_ops[0])

        # Multi-plane masks overview (only if multiple planes)
        if len(res_z) > 1:
            try:
                plot_multiplane_masks(
                    suite2p_path=save_path,
                    stat=all_stat,
                    iscell=all_iscell,
                    save_path=save_path / "all_planes_masks.png",
                )
                print(f"  Saved: all_planes_masks.png")
            except Exception as e:
                print(f"  Failed to generate multiplane masks: {e}")

        try:
            # Quality metrics summary
            plot_plane_quality_metrics(
                stat=all_stat,
                iscell=all_iscell,
                save_path=save_path / "volume_quality_metrics.png",
            )
            print(f"  Saved: volume_quality_metrics.png")
        except Exception as e:
            print(f"  Failed to generate quality metrics: {e}")

        try:
            # Trace analysis
            plot_trace_analysis(
                F=all_F,
                Fneu=all_Fneu,
                stat=all_stat,
                iscell=all_iscell,
                ops=first_ops,
                save_path=save_path / "volume_trace_analysis.png",
            )
            print(f"  Saved: volume_trace_analysis.png")
        except Exception as e:
            print(f"  Failed to generate trace analysis: {e}")

        try:
            # Summary table
            create_volume_summary_table(
                stat=all_stat,
                iscell=all_iscell,
                F=all_F,
                Fneu=all_Fneu,
                ops=first_ops,
                save_path=save_path / "volume_summary.csv",
            )
            print(f"  Saved: volume_summary.csv")
        except Exception as e:
            print(f"  Failed to generate summary table: {e}")

    except Exception:
        print("Volume statistics failed.")
        print("Traceback: ", traceback.format_exc())

    print(f"Processing completed for {len(input_files)} files.")
    return all_ops


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
    from contextlib import nullcontext
    from suite2p.io.binary import BinaryFile
    from suite2p.run_s2p import pipeline

    ops = load_ops(ops)
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])

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
        if ops["diameter"] is not None and np.isnan(ops["diameter"]):
            ops["diameter"] = 8
        if (ops["diameter"] in (None, 0)) and ops.get("anatomical_only", 0) > 0:
            ops["diameter"] = 8
            print("Warning: diameter was not set, defaulting to 8.")

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
    run_registration = bool(ops.get("do_registration", True))
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
    input_path: str | Path,
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
    save_json: bool = False,
    plane_name: str | None = None,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
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
        Root directory to save the results. A subdirectory will be created based on
        the input filename or `plane_name` parameter.
    ops : dict, str or Path, optional
        Path to or dict of user‐supplied ops.npy. If given, it overrides any existing or generated ops.
    chan2_file : str, optional
        Path to structural / anatomical data used for registration.
    keep_raw : bool, default False
        If True, do not delete the raw binary (`data_raw.bin`) after processing.
    keep_reg : bool, default True
        If True, keep the registered binary (`data.bin`) after processing.
    force_reg : bool, default False
        If True, force a new registration even if existing shifts are found in ops.npy.
    force_detect : bool, default False
        If True, force ROI detection even if an existing stat.npy is present.
    dff_window_size : int, optional
        Number of frames for rolling percentile baseline in ΔF/F₀ calculation.
        If None (default), auto-calculated as ~10 × tau × fs based on ops values.
        This ensures the window spans multiple calcium transients so the percentile
        filter can find the baseline between events.
    dff_percentile : int, default 20
        Percentile to use for baseline F₀ estimation in dF/F calculation.
    dff_smooth_window : int, optional
        Temporal smoothing window for dF/F traces (in frames).
        If None (default), auto-calculated as ~0.5 × tau × fs to emphasize
        transients while reducing noise. Set to 1 to disable smoothing.
    save_json : bool, default False
        If True, saves ops as a JSON file in addition to npy.
    plane_name : str, optional
        Custom name for the plane subdirectory. If None, derived from input filename.
        Used by run_volume() to control output directory naming.
    **kwargs : dict, optional

    Returns
    -------
    Path
        Path to the saved ops.npy file.

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
    - Results are saved to `save_path/{plane_name}/` subdirectory to keep outputs organized.

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
    from mbo_utilities.array_types import MboRawArray
    from mbo_utilities.lazy_array import imread, imwrite
    from mbo_utilities.metadata import get_metadata

    if "debug" in kwargs:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled.")

    assert isinstance(
        input_path, (Path, str)
    ), f"input_path should be a pathlib.Path or string, not: {type(input_path)}"
    input_path = Path(input_path)
    input_parent = input_path.parent

    assert isinstance(
        save_path, (Path, str, type(None))
    ), f"save_path should be a pathlib.Path or string, not: {type(save_path)}"

    # determine if input is a binary that should be processed in-place
    is_binary_input = input_path.suffix == ".bin"
    binary_with_ops = is_binary_input and (input_path.parent / "ops.npy").exists()

    if save_path is None:
        # no save_path provided - use input's parent directory
        if binary_with_ops:
            # binary inputs with ops.npy are processed in-place
            save_path = input_parent
        else:
            # for other inputs, create subdirectory next to input
            save_path = input_parent
    else:
        save_path = Path(save_path)
        if not save_path.parent.is_dir():
            raise ValueError(
                f"save_path does not have a valid parent directory: {save_path}"
            )
        save_path.mkdir(exist_ok=True)

    # create plane subdirectory unless processing binary in-place
    # this keeps planar outputs separate from volumetric outputs
    skip_imwrite = False
    if binary_with_ops and input_path.parent == save_path:
        # binary at target location - process in-place
        print(f"Input is already a binary at target location: {input_path}")
        plane_dir = save_path
        skip_imwrite = True
    else:
        # create subdirectory for this plane's outputs
        if plane_name is not None:
            subdir_name = plane_name
        else:
            subdir_name = derive_tag_from_filename(input_path.name)
        plane_dir = save_path / subdir_name
        plane_dir.mkdir(exist_ok=True)

    # check for existing ops.npy in the target directory
    ops_file = plane_dir / "ops.npy"

    ops_default = default_ops()
    ops_user = load_ops(ops) if ops else {}
    ops = {**ops_default, **ops_user, "data_path": str(input_path.resolve())}

    # suite2p diameter handling
    if (
        isinstance(ops["diameter"], list)
        and len(ops["diameter"]) > 1
        and ops["aspect"] == 1.0
    ):
        ops["aspect"] = ops["diameter"][0] / ops["diameter"][1]  # noqa

    # Skip imread if we're using existing binary OR if binary exists and passes validation
    should_write = skip_imwrite is False and _should_write_bin(ops_file, force=force_reg)

    # Normalize kwargs dicts
    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}

    if skip_imwrite or not should_write:
        file = None
        # Load metadata from existing ops.npy
        existing_ops = np.load(ops_file, allow_pickle=True).item() if ops_file.exists() else {}
        metadata = {k: v for k, v in existing_ops.items() if k in ("plane", "fs", "dx", "dy", "Ly", "Lx", "nframes")}
    else:
        # Only call imread if we're actually going to write the binary
        file = imread(input_path, **reader_kwargs)
        if isinstance(file, MboRawArray):
            raise TypeError(
                "Input file appears to be a raw array. Please provide a planar input file."
            )
        if hasattr(file, "metadata"):
            metadata = file.metadata  # noqa
        else:
            metadata = get_metadata(input_path)

    if "plane" in ops:
        plane = ops["plane"]
        metadata["plane"] = plane
    elif "plane" in metadata:
        plane = metadata["plane"]
        ops["plane"] = plane
    else:
        # get the plane from the filename
        tag = derive_tag_from_filename(input_path)
        plane = get_plane_num_from_tag(tag, fallback=ops.get("plane", None))
        ops["plane"] = plane
        metadata["plane"] = plane

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
                print(
                    f"Detected empty stat.npy, forcing roi detection for plane {plane}."
                )
                needs_detect = True
            else:
                print(
                    f"Roi detection skipped, stat.npy already exists for plane {plane}."
                )
                needs_detect = False
        else:
            print(
                f"ops['roidetect'] is True with no stat.npy file present, "
                f"proceeding with segmentation/detection for plane {plane}."
            )
            needs_detect = True
    elif (plane_dir / "stat.npy").is_file():
        # check contents of stat.npy
        stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
        if stat is None or len(stat) == 0:
            print(f"Detected empty stat.npy, forcing roi detection for plane {plane}.")
            needs_detect = True
        else:
            print(f"Roi detection skipped, stat.npy already exists for plane {plane}.")
            needs_detect = False

    # Write binary if needed (already determined should_write above)
    if skip_imwrite:
        print(f"Skipping binary write, using existing binary at {input_path}")
    elif should_write:
        md_combined = {**metadata, **ops}
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
    else:
        print(
            f"Skipping data_raw.bin write, already exists and passes data validation checks."
        )

    ops_outpath = (
        np.load(ops_file, allow_pickle=True).item()
        if (plane_dir / "ops.npy").exists()
        else {}
    )

    if force_reg:
        needs_reg = True
    else:
        if not ops_file.exists():
            needs_reg = True
        else:
            needs_reg = _should_register(ops_file)

    # Build ops dict - user settings should not be overwritten
    # Preserve the plane number that was determined earlier (line 769-779)
    correct_plane = plane

    # Only preserve processing results from existing ops, not file paths
    # This prevents cross-contamination when re-running with different parameters
    preserved_keys = {
        'refImg', 'meanImg', 'meanImgE', 'max_proj', 'Vcorr', 'Vmap',
        'xoff', 'yoff', 'corrXY', 'badframes', 'yrange', 'xrange',
        'regDX', 'regPC', 'regPC1', 'regDX1', 'tPC', 'tPC1',
        'Ly', 'Lx', 'nframes', 'nframes_chan1', 'nframes_chan2'
    }
    ops_preserved = {k: v for k, v in ops_outpath.items() if k in preserved_keys}

    ops = {
        **ops_default,
        **ops_preserved,  # Only preserved results, not paths
        **ops_user,
        "ops_path": str(ops_file),
        "save_path": str(plane_dir),
        "raw_file": str((plane_dir / "data_raw.bin").resolve()),
        "reg_file": str((plane_dir / "data.bin").resolve()),
        "plane": correct_plane,  # Override with correct plane number
    }

    # Set do_registration/roidetect based on force flags and needs analysis
    # force_reg/force_detect ALWAYS override user ops values
    if force_reg:
        ops["do_registration"] = 1
    elif force_reg is False and not needs_reg:
        # Skip registration if already done, regardless of user ops
        ops["do_registration"] = 0
        if ops_user.get("do_registration", 0) == 1:
            print(f"Registration already complete, skipping despite do_registration=1 in ops")
    elif "do_registration" not in ops_user:
        ops["do_registration"] = int(needs_reg)

    if force_detect:
        ops["roidetect"] = 1
    elif "roidetect" not in ops_user:
        ops["roidetect"] = int(needs_detect)

    # optional structural (channel 2) input
    if chan2_file is not None:
        chan2_file = Path(chan2_file)
        if not chan2_file.exists():
            raise FileNotFoundError(f"chan2_path not found: {chan2_file}")

        chan2_data = imread(chan2_file)
        chan2_md = chan2_data.metadata if hasattr(chan2_data, "metadata") else {}
        chan2_frames = chan2_md.get("num_frames") or chan2_md.get("nframes") or chan2_data.shape[0]

        # write channel 2 binary automatically
        imwrite(chan2_data, plane_dir, ext=".bin", metadata=chan2_md, register_z=False, structural=True)
        ops["chan2_file"] = str((plane_dir / "data_chan2.bin").resolve())
        ops["nframes_chan2"] = int(chan2_frames)
        ops["nchannels"] = 2
        ops["align_by_chan"] = 2

    if "nframes" not in ops:
        # try ops["metadata"] first, then metadata dict
        nframes = None
        if "metadata" in ops and "shape" in ops["metadata"]:
            nframes = ops["metadata"]["shape"][0]
        if nframes is None:
            nframes = get_param(metadata, "nframes")
        if nframes is None and file is not None and hasattr(file, "shape") and len(file.shape) >= 1:
            # warning: may trigger lazy loading
            print(f"Warning: nframes not found in metadata, loading file to determine shape (plane {plane})...")
            nframes = file.shape[0]
        if nframes is None:
            raise KeyError(
                "missing frame count (nframes) in ops or metadata, and cannot infer from data"
            )
        ops["nframes"] = nframes

    try:
        processed = run_plane_bin(ops)
    except Exception as e:
        print(f"Error in run_plane_bin for plane {plane}: {e}")
        traceback.print_exc()
        processed = False

    if not processed:
        print(f"Skipping {ops_file.name}, processing was not completed.")
        return ops_file

    if save_json:
        # convert ops dict to JSON serializable and save as ops.json
        ops_to_json(ops_file)

    raw_file = Path(ops.get("raw_file", plane_dir / "data_raw.bin"))
    reg_file = Path(ops.get("reg_file", plane_dir / "data.bin"))

    try:
        if not keep_raw and raw_file.exists():
            raw_file.unlink(missing_ok=True)
        if not keep_reg and reg_file.exists():
            reg_file.unlink(missing_ok=True)
    except Exception as e:
        print(e)

    # Only generate PC metrics if they don't already exist
    pc_panels = plane_dir / "pc_metrics_panels.tif"
    pc_csv = plane_dir / "pc_metrics.csv"
    if not pc_panels.exists() or not pc_csv.exists():
        save_pc_panels_and_metrics(ops_file, plane_dir / "pc_metrics")

    try:
        plot_zplane_figures(
            plane_dir,
            dff_percentile=dff_percentile,
            dff_window_size=dff_window_size,
            dff_smooth_window=dff_smooth_window,
            run_rastermap=kwargs.get("run_rastermap", False),
        )
    except Exception:
        traceback.print_exc()
    return ops_file


def grid_search(
    input_file: Path | str,
    save_path: Path | str,
    grid_params: dict,
    ops: dict = None,
    force_reg: bool = False,
    force_detect: bool = True,
):
    """
    Run a grid search over all combinations of Suite2p parameters.

    Tests all combinations of parameters in `grid_params`, running `run_plane`
    for each combination and saving results to separate subdirectories.

    Parameters
    ----------
    input_file : str or Path
        Path to the input data file (TIFF, Zarr, HDF5, etc.).
    save_path : str or Path
        Root directory where results will be saved. Each parameter combination
        gets its own subdirectory named by parameter values.
    grid_params : dict
        Dictionary mapping parameter names to lists of values to test.
        All combinations will be tested (Cartesian product).
    ops : dict, optional
        Base ops dictionary. If None, uses `default_ops()`. Grid parameters
        override values in this dictionary for each combination.
    force_reg : bool, default False
        If True, force registration even if already done.
    force_detect : bool, default True
        If True, force ROI detection for each combination.

    Notes
    -----
    - ``force_reg`` and ``force_detect`` override any ``do_registration`` or
      ``roidetect`` values in the ops dict. Users don't need to set those.
    - Subfolder names use abbreviated parameter keys (first 3 chars) and values.
    - Registration is shared across combinations when `force_reg=False`.
    - For Suite2p parameters, see: https://suite2p.readthedocs.io/en/latest/settings.html

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>>
    >>> # Search detection parameters
    >>> lsp.grid_search(
    ...     input_file="data/plane07.zarr",
    ...     save_path="results/grid_search",
    ...     grid_params={
    ...         "threshold_scaling": [0.8, 1.0, 1.2],
    ...         "diameter": [6, 8],
    ...     },
    ... )
    >>>
    >>> # Search Cellpose parameters
    >>> lsp.grid_search(
    ...     input_file="data/plane07.zarr",
    ...     save_path="results/cellpose_search",
    ...     grid_params={
    ...         "anatomical_only": [2, 3],
    ...         "spatial_hp_cp": [0, 0.5],
    ...         "diameter": [6, 8],
    ...     },
    ...     ops={"sparse_mode": False},  # Required for Cellpose
    ... )

    Output structure::

        results/grid_search/
        ├── thr0.80_dia6/
        │   ├── ops.npy, stat.npy, F.npy, ...
        ├── thr0.80_dia8/
        ├── thr1.00_dia6/
        └── ...

    """
    from lbm_suite2p_python.default_ops import default_ops

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    # Use default ops if not provided
    if ops is None:
        ops = default_ops()

    print(f"Grid search: {save_path}")
    print(f"Parameters: {list(grid_params.keys())}")

    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    param_combos = list(product(*param_values))

    n_total = len(param_combos)
    print(f"Total combinations: {n_total}")

    for i, combo in enumerate(param_combos, 1):
        combo_ops = copy.deepcopy(ops)
        combo_dict = dict(zip(param_names, combo))
        combo_ops.update(combo_dict)

        # Create readable folder name from parameters
        tag_parts = [
            f"{k[:3]}{v:.2f}" if isinstance(v, float) else f"{k[:3]}{v}"
            for k, v in combo_dict.items()
        ]
        tag = "_".join(tag_parts)
        combo_save_path = save_path / tag

        print(f"\n[{i}/{n_total}] {tag}")

        # Use plane_name to put results directly in combo folder (no extra subdir)
        ops_file = combo_save_path / "ops.npy"

        # Skip if already processed (unless forcing)
        if ops_file.exists() and not force_reg and not force_detect:
            # Check if detection is complete
            stat_file = combo_save_path / "stat.npy"
            if stat_file.exists():
                print(f"  Skipping: already complete")
                continue

        run_plane(
            input_path=input_file,
            save_path=save_path,
            ops=combo_ops,
            keep_reg=True,
            keep_raw=False,
            force_reg=force_reg,
            force_detect=force_detect,
            plane_name=tag,  # Use tag as plane_name to control output dir
        )

    print(f"\nGrid search complete: {n_total} combinations")
    print(f"Results in: {save_path}")
