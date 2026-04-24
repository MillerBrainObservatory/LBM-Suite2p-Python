import logging
import time
from datetime import datetime
from pathlib import Path
import traceback
import copy

import numpy as np

from lbm_suite2p_python.default_ops import default_ops
from lbm_suite2p_python.db_settings import save_ops_db_settings
from lbm_suite2p_python.postprocessing import (
    ops_to_json,
    load_ops,
    dff_rolling_percentile,
    apply_filters,
)

from importlib.metadata import version, PackageNotFoundError


def _call_upstream_pipeline(ops, f_reg, f_raw, f_reg_chan2, f_raw_chan2,
                            run_registration, stat=None):
    """Compat wrapper around upstream `suite2p.run_s2p.pipeline` that
    preserves the fork's `ops`-in / `ops`-out contract.

    Upstream's pipeline now takes a nested `settings` dict, requires
    `save_path` positionally, and returns a 12-tuple of discrete outputs
    instead of a merged ops. This helper:
      1. Splits the flat fork ops into (db, settings) via db_settings helpers.
      2. Resolves torch_device from ops if set.
      3. Calls pipeline with the upstream signature.
      4. Folds reg_outputs / detect_outputs dicts back into the flat ops
         dict so downstream fork code (which reads e.g. ops['meanImg'],
         ops['yrange']) keeps working.
    """
    from suite2p.run_s2p import pipeline as _upstream_pipeline
    from suite2p import default_settings as _us_default_settings
    from lbm_suite2p_python.db_settings import ops_to_db_settings

    save_path = ops.get("save_path") or ops.get("ops_path")
    if save_path is None:
        raise ValueError("ops must carry 'save_path' or 'ops_path' to call upstream pipeline")
    save_path = str(Path(save_path).parent) if str(save_path).endswith("ops.npy") else str(save_path)

    # build settings by overlaying ops-derived values onto the full upstream
    # default schema. upstream frequently grows new required keys
    # (upsample_meanImg, batch_size, ...) that won't exist in an ops dict
    # built by the fork — falling back to upstream defaults keeps us safe
    # as upstream evolves without a db_settings.py patch on every release.
    _, derived = ops_to_db_settings(ops)
    settings = _us_default_settings()
    def _deep_merge(base, overlay):
        for k, v in overlay.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _deep_merge(base[k], v)
            else:
                base[k] = v
    _deep_merge(settings, derived)

    # resolve torch device (default cuda, fall back to cpu on allocation)
    try:
        import torch
        dev_name = ops.get("torch_device") or settings.get("torch_device") or "cuda"
        try:
            device = torch.device(dev_name)
            # probe a tiny allocation to validate cuda is actually usable
            if device.type == "cuda":
                torch.zeros(1, device=device)
        except Exception:
            device = torch.device("cpu")
    except ImportError:
        device = None

    # extraction leak: upstream's create_neuropil_masks treats pixels
    # outside ops["yrange"]/ops["xrange"] as valid neuropil candidates
    # because cell_pix is zero there. for axially-padded LBM data that
    # pulls zero-valued padding into Fneu and corrupts dF/F at boundary
    # ROIs. the shim monkeypatches create_cell_pix for this call only so
    # the donut grower never walks into padding.
    from lbm_suite2p_python._padding_shim import padding_aware_extraction
    with padding_aware_extraction(ops):
        (reg_outputs, detect_outputs, stat, F, Fneu, F_chan2, Fneu_chan2,
         spks, iscell, redcell, zcorr, plane_times) = _upstream_pipeline(
            save_path=save_path,
            f_reg=f_reg,
            f_raw=f_raw,
            f_reg_chan2=f_reg_chan2,
            f_raw_chan2=f_raw_chan2,
            run_registration=run_registration,
            settings=settings,
            badframes=ops.get("badframes"),
            stat=stat,
            device=device if device is not None else None,
        )

    # fold outputs back into flat ops for fork consumers
    if isinstance(reg_outputs, dict):
        for k, v in reg_outputs.items():
            ops[k] = v
    if isinstance(detect_outputs, dict):
        for k, v in detect_outputs.items():
            # don't clobber diameter — keep whatever was passed in unless upstream
            # actually refined it (and collapse list to scalar for fork consumers)
            if k == "diameter" and v is not None:
                if isinstance(v, (list, tuple)) and len(v) >= 1:
                    v = float(v[0]) if len(v) < 2 or v[0] == v[1] else (float(v[0]) + float(v[1])) / 2
                ops[k] = v
            elif k != "diameter":
                ops[k] = v
    ops["plane_times"] = plane_times
    return ops


def _compute_enhanced_mean_image(img, ops):
    """Compat shim for upstream's removal of `compute_enhanced_mean_image`.

    Upstream suite2p replaced `compute_enhanced_mean_image(img, ops)` with
    `highpass_mean_image(I, aspect=1.0)` — the ops dict is no longer read
    (aspect is now the only knob, pulled from ops by the caller). Some
    fork call sites pass `None` for `img` and rely on ops["meanImg"]; we
    reproduce that fallback here.
    """
    from suite2p.registration import highpass_mean_image
    if img is None:
        img = ops["meanImg"]
    aspect = ops.get("aspect", 1.0) or 1.0
    return highpass_mean_image(np.asarray(img).astype(np.float32), aspect=float(aspect))


def _get_version():
    try:
        return version("lbm_suite2p_python")
    except PackageNotFoundError:
        return "0.0.0"


# canonical set of frame-count alias keys that mbo_utilities writes to
# ops.npy. lbm_suite2p_python and mbo_utilities must keep these in
# lockstep — when one updates the count (e.g. dual-channel trim), the
# others must follow or downstream readers see contradictory values.
_FRAME_COUNT_ALIASES = (
    "nframes",
    "num_frames",
    "num_timepoints",
    "n_frames",
    "T",
    "nt",
    "timepoints",
)


def _set_frame_count_aliases(ops: dict, n: int) -> None:
    """Set every frame-count alias in `ops` to `n`.

    Centralizes the fan-out so adding a new alias only requires
    updating `_FRAME_COUNT_ALIASES`. Channel-specific keys
    (`nframes_chan1`, `nframes_chan2`) are NOT touched here — those
    have their own semantics and must be set explicitly by the caller.
    """
    n = int(n)
    for key in _FRAME_COUNT_ALIASES:
        ops[key] = n


# detection outputs produced by suite2p in the plane dir. copied into a
# new plane_dir alongside ops.npy so figure regeneration works when the
# user reruns against a different save_path.
_DETECTION_OUTPUT_FILES = (
    "stat.npy", "iscell.npy", "F.npy", "Fneu.npy",
    "spks.npy", "dff.npy", "redcell.npy", "zcorr.npy",
    "reg_outputs.npy", "detect_outputs.npy",
    "F_chan2.npy", "Fneu_chan2.npy",
    "roi_stats.npy",
)


def _resolve_source_plane_dir(input_arr, input_path, target_plane, source_plane_num=None):
    """Find the source directory for a specific plane.

    Raises instead of silently falling back to ``input_path.parent`` when
    a match can't be found uniquely. Previously a missed lookup caused
    plane 1's files to be copied into plane N's output dir (see the
    'Copying data.bin from zplane01 -> zplane02' bug).

    Parameters
    ----------
    input_arr : lazy array or None
        Lazy array from ``mbo.imread`` — expected to carry a ``filenames``
        attribute for volumetric Suite2p arrays.
    input_path : Path
        Fallback path (used only for single-plane inputs).
    target_plane : int
        1-based plane number we're trying to process.
    source_plane_num : int, optional
        1-based source-plane identity when the loaded array represents a
        plane subset (e.g. every-other). Takes precedence over
        ``target_plane`` for the zplane-name match.

    Returns
    -------
    Path
        Source plane directory.
    """
    search_plane = source_plane_num if source_plane_num is not None else target_plane

    if input_arr is not None and hasattr(input_arr, "filenames") and input_arr.filenames:
        matches = []
        for fname in input_arr.filenames:
            fpath = Path(fname)
            if f"zplane{search_plane:02d}" in fpath.parent.name:
                matches.append(fpath.parent)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous source directory for plane {search_plane}: "
                f"multiple matches in input_arr.filenames: {matches}. "
                "Narrow the input to a single plane or clean up the source tree."
            )
        # zero matches — but if the input only contains a single plane
        # whose name doesn't carry a zplane number (e.g. processed
        # single-plane), and that's the plane we're looking for, fall
        # back to that single filename's parent.
        if len(input_arr.filenames) == 1:
            return Path(input_arr.filenames[0]).parent
        raise FileNotFoundError(
            f"No source directory for plane {search_plane} in input_arr. "
            f"Searched filenames: {[str(p) for p in input_arr.filenames]}. "
            "The per-plane lookup scans for 'zplane{NN:02d}' in each "
            "filename's parent dir name."
        )

    # no lazy array — input was a direct path. only safe when the input
    # itself is already a plane dir.
    return input_path.parent


def _copy_if_needed(src: Path, dst: Path, *, overwrite_empty: bool = True) -> bool:
    """Copy src -> dst when dst is missing (or empty, if overwrite_empty).

    Returns True when a copy actually happened. Never clobbers a
    non-empty destination.
    """
    import shutil
    if not src.exists():
        return False
    if dst.exists():
        if not (overwrite_empty and dst.stat().st_size == 0):
            return False
    print(f"  Copying {src.name} from {src.parent} -> {dst.parent}")
    shutil.copy2(src, dst)
    return True


def _stage_source_into_plane_dir(
    src_dir: Path,
    plane_dir: Path,
    source_mode: str,
    ops: dict,
) -> None:
    """Stage files from a source plane dir into ``plane_dir`` per ``source_mode``.

    Modes:

    - ``"copy"``: copy everything — raw + registered binaries + detection
      outputs + ops.npy. Self-contained output dir; matches the pre-fix
      behavior for datasets that still have ``data_raw.bin``.
    - ``"copy_reg"``: skip ``data_raw.bin``; copy ``data.bin``,
      ``data_chan2.bin``, detection outputs, ops.npy. Right default when
      you want to re-run detection settings and the raw binary was
      already discarded.
    - ``"link"``: don't copy any binary. Copy detection outputs and
      ops.npy. Point ``ops['raw_file']`` / ``ops['reg_file']`` /
      ``ops['chan2_file']`` at the source paths. Source binaries are
      read-only from this plane's perspective — registration must be off
      (the caller should enforce).
    - ``"auto"``: ``copy`` if source has ``data_raw.bin``, otherwise
      ``copy_reg``. Never picks ``link`` implicitly — that's an opt-in.

    ``ops`` is mutated in place for the link paths.
    """
    src_dir = Path(src_dir)
    plane_dir = Path(plane_dir)
    plane_dir.mkdir(parents=True, exist_ok=True)

    # auto resolution
    if source_mode == "auto":
        source_mode = "copy" if (src_dir / "data_raw.bin").exists() else "copy_reg"

    valid_modes = {"copy", "copy_reg", "link"}
    if source_mode not in valid_modes:
        raise ValueError(
            f"Invalid source_mode={source_mode!r}. "
            f"Expected one of {sorted(valid_modes)}."
        )

    # ops.npy + detection outputs are always staged into plane_dir so
    # figures/post-processing can read them locally.
    _copy_if_needed(src_dir / "ops.npy", plane_dir / "ops.npy")
    for fname in _DETECTION_OUTPUT_FILES:
        _copy_if_needed(src_dir / fname, plane_dir / fname)

    if source_mode == "copy":
        for fname in ("data_raw.bin", "data.bin", "data_chan2.bin"):
            _copy_if_needed(src_dir / fname, plane_dir / fname)
    elif source_mode == "copy_reg":
        for fname in ("data.bin", "data_chan2.bin"):
            _copy_if_needed(src_dir / fname, plane_dir / fname)
        # don't carry a stale raw_file pointer — there's no raw on disk
        # in plane_dir and we don't want it to point back at source.
        ops.pop("raw_file", None)
    else:  # link
        for ops_key, fname in (
            ("raw_file", "data_raw.bin"),
            ("reg_file", "data.bin"),
            ("chan2_file", "data_chan2.bin"),
        ):
            src_bin = src_dir / fname
            if src_bin.exists():
                ops[ops_key] = str(src_bin.resolve())


def _apply_reactive_metadata(
    ops: dict,
    source_metadata: dict | None,
    source_shape: tuple | None = None,
    frame_indices: list[int] | None = None,
    plane_indices: list[int] | None = None,
    logger=None,
) -> None:
    """Reactively scale fs/dz in `ops` from a source metadata dict and
    the user's stride selections, in place.

    Uses `mbo_utilities.metadata.OutputMetadata` so the scaling rules
    stay aligned with the writer side. Without this, every-other-
    timepoint produces ops.npy with raw source `fs` (factor of 2
    wrong), and every-other-plane produces ops.npy with raw source `dz`.

    Parameters
    ----------
    ops : dict
        ops dict to mutate. Existing keys are NOT overwritten unless
        the reactive value differs (so user-set explicit overrides win
        over the reactive computation).
    source_metadata : dict | None
        Source acquisition metadata (must contain `fs`, `dz` etc. for
        scaling to work). If None, function is a no-op.
    source_shape : tuple | None
        Source shape tuple (T, C, Z, Y, X). If None, OutputMetadata
        falls back to whatever it can infer.
    frame_indices : list[int] | None
        0-based timepoint indices the user selected. The implicit
        stride scales `fs`. None means no T-stride scaling.
    plane_indices : list[int] | None
        0-based z-plane indices the user selected (FULL list, not just
        the current plane). The implicit stride scales `dz`. None
        means no Z-stride scaling.
    logger : optional
        If provided, a missing `fs` produces a `.warning(...)` call.
    """
    if not source_metadata:
        return

    try:
        from mbo_utilities.metadata import OutputMetadata, get_param
    except ImportError:
        # mbo_utilities not available — caller's choice to handle this
        return

    if logger is not None and get_param(source_metadata, "fs") is None:
        logger.warning(
            "Source metadata has no 'fs' field. Frame rate in ops.npy "
            "may not reflect the acquisition rate. Set fs via the "
            "metadata editor or fix the source TIFF metadata."
        )

    selections: dict = {}
    if frame_indices is not None:
        selections["T"] = list(frame_indices)
    if plane_indices is not None:
        selections["Z"] = list(plane_indices)

    if not selections:
        return

    # OutputMetadata's accessors crash on `source_shape=None`. Default
    # to an empty tuple — the reactive scaling for fs/dz only needs the
    # selections, not the absolute shape.
    out_meta = OutputMetadata(
        source=dict(source_metadata),
        source_shape=source_shape if source_shape is not None else (),
        source_dims=("T", "C", "Z", "Y", "X"),
        selections=selections,
    )

    scaled = out_meta.to_dict()
    # Split-responsibility scaling:
    #   - `fs` is owned by mbo's writer (`_imwrite_base` builds its own
    #     OutputMetadata from `frames=` and scales fs there). If we
    #     also scaled fs here, the writer would re-scale our already-
    #     scaled value and end up with fs/4 instead of fs/2.
    #   - `dz` (and dx/dy) are owned by lbm because mbo's writer only
    #     sees the per-plane data, not the FULL plane selection list,
    #     so it cannot compute the implicit z-stride.
    # We also propagate `_metadata_provenance` so the next pass (a
    # re-run of this ops through OutputMetadata) reads the stamped
    # base+stride and doesn't double-scale dz.
    if plane_indices is not None:
        for key in ("dz", "dx", "dy", "z_step",
                    "umPerPixZ", "umPerPixX", "umPerPixY",
                    "_metadata_provenance"):
            if key in scaled and scaled[key] is not None:
                ops[key] = scaled[key]


from lbm_suite2p_python.zplane import (
    save_pc_panels_and_metrics,
    plot_zplane_figures,
    plot_filtered_cells,
    plot_filter_exclusions,
    plot_cell_filter_summary,
)

DEFAULT_CELL_FILTERS = []
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


def _add_processing_step(
    ops, step_name, input_files=None, duration_seconds=None, extra=None
):
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
        step_record["input_files"] = (
            [str(f) for f in input_files]
            if not isinstance(input_files, str)
            else [input_files]
        )

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
    db: dict | None = None,
    settings: dict | None = None,
    planes: list | int = None,
    roi_mode: int = None,
    keep_reg: bool = True,
    keep_raw: bool = False,
    force_reg: bool = False,
    force_detect: bool = False,
    num_timepoints: int = None,
    frame_indices: list | None = None,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    cell_filters: list = None,
    accept_all_cells: bool = False,
    save_json: bool = False,
    source_mode: str = "auto",
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
        Input data source. Can be a file path, directory, list of files,
        or a lazy array from mbo_utilities.
    save_path : str or Path, optional
        Output directory for results. If None, saves next to input file.
        Required when input_data is an array without filenames.
    ops : dict, optional
        Suite2p parameters to override defaults. If None, uses optimized
        defaults with metadata auto-populated from input.
    planes : int or list, optional
        Which z-planes to process (1-indexed). None processes all planes.
    roi_mode : int, optional
        ROI handling for multi-ROI ScanImage data:
        - None: stitch all ROIs into single FOV (default)
        - 0: process each ROI separately
        - N > 0: process only ROI N (1-indexed)
    keep_reg : bool, default True
        Keep registered binary (data.bin) after processing.
    keep_raw : bool, default False
        Keep raw binary (data_raw.bin) after processing.
    force_reg : bool, default False
        Force re-registration even if already complete.
    force_detect : bool, default False
        Force ROI detection even if stat.npy exists.
    num_timepoints : int, optional
        Limit processing to first N frames (truncation only). For an
        explicit set of frames or a strided selection, use
        ``frame_indices`` instead.
    frame_indices : list[int], optional
        Explicit 0-based frame indices to process. Supports stride
        (e.g. ``list(range(0, 1574, 2))`` for every other frame).
        When provided, the implicit stride is used by `OutputMetadata`
        to reactively scale `fs` (e.g. stride of 2 → `fs / 2` in the
        output ops.npy). Takes precedence over ``num_timepoints``.
    dff_window_size : int, optional
        Window size for rolling percentile dF/F baseline (frames).
        If None, auto-calculated as ~10 * tau * fs.
    dff_percentile : int, default 20
        Percentile for baseline F0 estimation.
    dff_smooth_window : int, optional
        Temporal smoothing window for dF/F traces (frames).
        If None, auto-calculated. Set to 1 to disable.
    cell_filters : list, optional
        Filters to apply to detected ROIs. Default is no filters.
        Example: [{"name": "max_diameter", "min_diameter_um": 4, "max_diameter_um": 35}]
    accept_all_cells : bool, default False
        If True, mark all detected ROIs as accepted, overriding
        Suite2p's built-in classifier results.
    save_json : bool, default False
        Save ops as JSON in addition to .npy format.
    reader_kwargs : dict, optional
        Arguments passed to mbo_utilities.imread().
    writer_kwargs : dict, optional
        Arguments passed to binary writer (e.g., output_format).
    plane_name : str, optional
        Name for output directory when input is an array without
        filenames. Passed via kwargs.
    roi : int, optional
        Deprecated. Use roi_mode instead.
    num_frames : int, optional
        Deprecated. Use num_timepoints instead.
    **kwargs
        Additional arguments passed to run_plane or run_volume.

    Returns
    -------
    list[Path]
        Paths to ops.npy files for each processed plane.

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>> ops = {"anatomical_only": 4} # max-projection
    >>> results = lsp.pipeline("D:/data/volume.zarr", planes=[1, 5, 10], ops=ops)

    >>> # process array, accept all ROIs without filtering
    >>> results = lsp.pipeline(
    ...     arr, save_path="D:/results", plane_name="plane0",
    ...     accept_all_cells=True, cell_filters=[], ops=ops
    ... )

    See Also
    --------
    run_plane : lower-level single-plane processing
    run_volume : process list of files
    """
    from mbo_utilities import imread
    from mbo_utilities.arrays import supports_roi

    # 1. Handle Deprecations
    if roi is not None:
        import warnings

        warnings.warn(
            "'roi' is deprecated, use 'roi_mode'", DeprecationWarning, stacklevel=2
        )
        roi_mode = roi

    if num_frames is not None:
        import warnings

        warnings.warn(
            "'num_frames' is deprecated, use 'num_timepoints'",
            DeprecationWarning,
            stacklevel=2,
        )
        num_timepoints = num_frames

    # flatten (db, settings) into ops so downstream run_volume / run_plane
    # don't each need to forward the pair. explicit ops keys still win.
    if db is not None or settings is not None:
        from lbm_suite2p_python.db_settings import merge_db_settings_into_ops
        ops = merge_db_settings_into_ops(ops if isinstance(ops, dict) else None, db, settings)

    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}
    if num_timepoints is not None:
        writer_kwargs["num_frames"] = num_timepoints

    # 1-based frame numbers.
    if frame_indices is not None:
        writer_kwargs["frames"] = [int(i) + 1 for i in frame_indices]
        # don't double-pass num_frames; len(frame_indices) is implicit
        writer_kwargs.pop("num_frames", None)

    # Always load array to check dimensions and ensure downstream functions have the array shape
    # If input is already array, this is fast. If path or list of paths, it loads lazy array.
    if _is_lazy_array(input_data):
        arr = input_data
    else:
        print(f"Loading input to determine dimensions...")
        arr = imread(input_data, **reader_kwargs)

    # Apply ROI mode if applicable check dimensions
    if roi_mode is not None and supports_roi(arr):
        arr.roi = roi_mode

    # check if data is volumetric (multiple Z planes)
    # mbo_utilities arrays expose nz; legacy arrays use num_planes or shape inference
    is_volumetric = _get_num_planes(arr) > 1

    # 3. Delegate
    if is_volumetric:
        print("Delegating to run_volume (volumetric input detected)...")
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
            frame_indices=frame_indices,
            dff_window_size=dff_window_size,
            dff_percentile=dff_percentile,
            dff_smooth_window=dff_smooth_window,
            accept_all_cells=accept_all_cells,
            cell_filters=cell_filters,
            save_json=save_json,
            source_mode=source_mode,
            reader_kwargs=reader_kwargs,
            writer_kwargs=writer_kwargs,
            **kwargs,
        )
    else:
        # run_plane returns a single Path, we wrap in list
        ops_path = run_plane(
            input_data=arr,  # Pass the array we loaded (with ROI applied)
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
            frame_indices=frame_indices,
            dff_window_size=dff_window_size,
            dff_percentile=dff_percentile,
            dff_smooth_window=dff_smooth_window,
            accept_all_cells=accept_all_cells,
            cell_filters=cell_filters,
            save_json=save_json,
            source_mode=source_mode,
            reader_kwargs=reader_kwargs,
            writer_kwargs=writer_kwargs,
            **kwargs,
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


def generate_plane_dirname(
    plane: int,
    nframes: int = None,
    frame_start: int = 1,
    frame_stop: int = None,
    suffix: str = None,
) -> str:
    """
    generate a descriptive directory name for a plane's outputs.

    uses mbo_utilities tag conventions for consistent, self-documenting names.
    format: zplaneNN[_tpSTART-STOP][_suffix]

    parameters
    ----------
    plane : int
        z-plane number (1-based)
    nframes : int, optional
        total number of frames. if provided and > 1, adds timepoint range.
    frame_start : int, default 1
        first frame (1-based)
    frame_stop : int, optional
        last frame (1-based). defaults to nframes if not provided.
    suffix : str, optional
        additional suffix (e.g., "stitched", "roi1")

    returns
    -------
    str
        directory name like "zplane01", "zplane03_tp00001-05000", etc.

    examples
    --------
    >>> generate_plane_dirname(3)
    'zplane03'
    >>> generate_plane_dirname(3, nframes=5000)
    'zplane03_tp00001-05000'
    >>> generate_plane_dirname(3, nframes=5000, suffix="stitched")
    'zplane03_tp00001-05000_stitched'
    >>> generate_plane_dirname(1, frame_start=100, frame_stop=500)
    'zplane01_tp00100-00500'
    """
    from mbo_utilities.arrays.features._dim_tags import TAG_REGISTRY, DimensionTag

    parts = []

    # z-plane tag (always present)
    z_def = TAG_REGISTRY["Z"]
    z_tag = DimensionTag(z_def, start=plane, stop=None, step=1)
    parts.append(z_tag.to_string())

    # timepoint tag (if multiple frames)
    if nframes is not None and nframes > 1:
        t_def = TAG_REGISTRY["T"]
        stop = frame_stop if frame_stop is not None else nframes
        t_tag = DimensionTag(t_def, start=frame_start, stop=stop, step=1)
        parts.append(t_tag.to_string())

    # optional suffix
    if suffix:
        parts.append(suffix)

    return "_".join(parts)


def run_volume(
    input_data,
    save_path: str | Path = None,
    ops: dict | str | Path = None,
    planes: list | int = None,
    keep_reg: bool = True,
    keep_raw: bool = False,
    force_reg: bool = False,
    force_detect: bool = False,
    frame_indices: list | None = None,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    accept_all_cells: bool = False,
    cell_filters: list = None,
    save_json: bool = False,
    source_mode: str = "auto",
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
    frame_indices : list, default None
        List of frame indices to process.
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
        # load as array to get actual plane count (files may contain interleaved planes)
        input_arr = imread(input_paths, **(reader_kwargs or {}))
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

    # Extract voxel size from input array metadata for propagation to per-plane ops
    _volume_voxel = None
    if input_arr is not None and hasattr(input_arr, "metadata"):
        _volume_voxel = get_voxel_size(input_arr.metadata)

    # Normalize planes to process
    planes_indices = _normalize_planes(planes, num_planes)

    # Build the FULL Z selection (0-based) — we pass this to every
    # run_plane call so its OutputMetadata layer can compute the
    # implicit z-stride and reactively scale dz. Without the full list
    # in scope, each per-plane ops.npy would carry the unscaled source
    # dz even though only every Nth plane was being processed.
    _volume_z_selection = list(planes_indices) if planes_indices else None
    # Snapshot the source acquisition metadata once — used by the
    # reactive helper inside the per-plane loop to scale dz/fs.
    _volume_source_meta = (
        dict(input_arr.metadata)
        if input_arr is not None and hasattr(input_arr, "metadata")
        else None
    )
    _volume_source_shape = (
        tuple(input_arr.shape5d)
        if input_arr is not None and hasattr(input_arr, "shape5d")
        else None
    )

    print(
        f"Processing {len(planes_indices)} planes in volume (Total planes: {num_planes})"
    )
    print(f"Output: {save_path}")

    progress_callback = kwargs.pop("progress_callback", None)

    ops_files = []

    # Iterate
    for i, plane_idx in enumerate(planes_indices):
        plane_num = plane_idx + 1

        if progress_callback:
            progress_callback(
                plane=i,
                total_planes=len(planes_indices),
                step="plane_start",
                message=f"Plane {plane_num}",
            )

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
                current_input = input_paths[0]  # Should not happen if logic is correct

        # Prepare ops with plane number — deepcopy so suite2p's in-place
        # mutations (badframes, meanImg, Ly/Lx, etc.) don't leak across planes
        current_ops = copy.deepcopy(load_ops(ops)) if ops else default_ops()
        current_ops["plane"] = plane_num
        current_ops["num_zplanes"] = num_planes  # useful info

        # if the source was saved as a plane-subset (e.g. every-other-plane),
        # mbo stamps `selected_planes_0based` into the file metadata. use
        # that to translate the local plane index back to source-plane
        # identity so output dirs read `zplane07_...` rather than
        # `zplane02_...`. `ops["plane"]` stays as the local index because
        # downstream slicing (`planes=[ops["plane"]]` in run_plane) must
        # index into the *loaded* array, which only has the saved subset.
        if input_arr is not None and hasattr(input_arr, "metadata"):
            _source_idx = (input_arr.metadata or {}).get("selected_planes_0based")
            if _source_idx is not None and plane_idx < len(_source_idx):
                current_ops["source_plane_num"] = int(_source_idx[plane_idx]) + 1

        # Propagate voxel size from volume metadata into per-plane ops
        if _volume_voxel is not None:
            if _volume_voxel.dz is not None:
                current_ops.setdefault("dz", _volume_voxel.dz)
                current_ops.setdefault("z_step", _volume_voxel.dz)
            if _volume_voxel.dx != 1.0:
                current_ops.setdefault("dx", _volume_voxel.dx)
            if _volume_voxel.dy != 1.0:
                current_ops.setdefault("dy", _volume_voxel.dy)

        # Reactively scale dz (and fs if frame_indices are provided)
        # using the FULL plane selection. This MUST happen before
        # run_plane fires, because run_plane sees only its own plane
        # number and can't compute the implicit z-stride on its own.
        # The helper overwrites dz/dx/dy/fs in current_ops so the
        # `setdefault` calls above are effectively replaced when the
        # reactive value differs.
        _apply_reactive_metadata(
            ops=current_ops,
            source_metadata=_volume_source_meta,
            source_shape=_volume_source_shape,
            frame_indices=frame_indices,
            plane_indices=_volume_z_selection,
            logger=logger,
        )

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
                frame_indices=frame_indices,
                dff_window_size=dff_window_size,
                dff_percentile=dff_percentile,
                dff_smooth_window=dff_smooth_window,
                accept_all_cells=accept_all_cells,
                cell_filters=cell_filters,
                save_json=save_json,
                source_mode=source_mode,
                reader_kwargs=reader_kwargs,
                writer_kwargs=writer_kwargs,
                **kwargs,
            )
            ops_files.append(ops_file)

            if progress_callback:
                progress_callback(
                    plane=i,
                    total_planes=len(planes_indices),
                    step="plane_done",
                    message=f"Plane {plane_num} complete",
                )
        except Exception as e:
            print(f"ERROR processing plane {plane_num}: {e}")
            traceback.print_exc()

    if not ops_files:
        raise RuntimeError(
            "run_volume failed: All planes resulted in exceptions during processing."
        )

    # Post-Loop: Merging and Volume Stats

    # Check for multi-ROI merging using metadata (not filename heuristics)
    should_merge = False
    if input_arr is not None and hasattr(input_arr, "metadata"):
        md = input_arr.metadata
        roi_mode = md.get("roi_mode")
        num_rois = md.get("num_rois") or md.get("num_mrois") or 1
        # only merge if roi_mode is "separate" (files written as separate rois)
        if roi_mode == "separate" and num_rois > 1:
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
                save_path = merged_savepath  # Update save_path for subsequent plots
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
                plot_volume_diagnostics(
                    ops_files, save_path / "volume_quality_diagnostics.png"
                )
                plot_orthoslices(ops_files, save_path / "orthoslices.png")
                plot_3d_roi_map(ops_files, save_path / "roi_map_3d.png", color_by="snr")
                plot_3d_roi_map(
                    ops_files, save_path / "roi_map_3d_plane.png", color_by="plane"
                )
            except Exception as e:
                print(f"Warning: Volume plots failed: {e}")
                traceback.print_exc()

        except Exception as e:
            print(f"Warning: Volume statistics failed: {e}")
            traceback.print_exc()

    return ops_files


def _should_write_bin(
    ops_path: Path,
    force: bool = False,
    *,
    validate_chan2: bool | None = None,
    expected_dtype: np.dtype = np.int16,
    expected_nframes: int | None = None,
    expected_ly: int | None = None,
    expected_lx: int | None = None,
) -> bool:
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
            validate_chan2 = ops.get("align_by_chan", 1) == 2
        Ly = ops.get("Ly")
        Lx = ops.get("Lx")
        nframes_raw = (
            ops.get("nframes_chan1") or ops.get("nframes") or ops.get("num_frames")
        )
        if (None in (nframes_raw, Ly, Lx)) or (nframes_raw <= 0 or Ly <= 0 or Lx <= 0):
            return True

        # invalidate cache if requested dimensions differ from cached
        if expected_nframes is not None and int(expected_nframes) != int(nframes_raw):
            print(
                f"Cache invalidated: cached nframes={nframes_raw}, requested={expected_nframes}"
            )
            return True
        if expected_ly is not None and int(expected_ly) != int(Ly):
            print(f"Cache invalidated: cached Ly={Ly}, requested={expected_ly}")
            return True
        if expected_lx is not None and int(expected_lx) != int(Lx):
            print(f"Cache invalidated: cached Lx={Lx}, requested={expected_lx}")
            return True

        expected_size_raw = (
            int(nframes_raw) * int(Ly) * int(Lx) * np.dtype(expected_dtype).itemsize
        )
        actual_size_raw = binary_to_validate.stat().st_size
        if actual_size_raw != expected_size_raw or actual_size_raw == 0:
            return True
        try:
            arr = np.memmap(
                binary_to_validate,
                dtype=expected_dtype,
                mode="r",
                shape=(int(nframes_raw), int(Ly), int(Lx)),
            )
            _ = arr[0, 0, 0]
            del arr
        except Exception:
            return True
        if validate_chan2:
            nframes_chan2 = ops.get("nframes_chan2")
            if (
                (not chan2_path.is_file())
                or (nframes_chan2 is None)
                or (nframes_chan2 <= 0)
            ):
                return True
            expected_size_chan2 = (
                int(nframes_chan2)
                * int(Ly)
                * int(Lx)
                * np.dtype(expected_dtype).itemsize
            )
            actual_size_chan2 = chan2_path.stat().st_size
            if actual_size_chan2 != expected_size_chan2 or actual_size_chan2 == 0:
                return True
            try:
                arr2 = np.memmap(
                    chan2_path,
                    dtype=expected_dtype,
                    mode="r",
                    shape=(int(nframes_chan2), int(Ly), int(Lx)),
                )
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

    # fixup stale absolute paths when data has been moved to a new location.
    # ops_path is the ground truth; derive all other paths from its parent.
    ops_path_stored = Path(ops.get("ops_path", ""))
    if ops_path_stored.name == "ops.npy" and ops_path_stored.parent.exists():
        ops_parent = ops_path_stored.parent
    elif "save_path" in ops and Path(ops["save_path"]).exists():
        ops_parent = Path(ops["save_path"])
        ops["ops_path"] = str(ops_parent / "ops.npy")
    else:
        raise FileNotFoundError(
            f"Cannot locate plane directory. ops_path={ops.get('ops_path')}, "
            f"save_path={ops.get('save_path')}"
        )

    # resolve path keys. rewrite stale absolutes (ops was moved) but
    # preserve an already-valid absolute path (source_mode="link" points
    # raw_file/reg_file/chan2_file at the source dir — clobbering those
    # would either break the link or force a redundant copy).
    ops["save_path"] = str(ops_parent)
    ops["ops_path"] = str(ops_parent / "ops.npy")

    def _repoint_if_stale(ops_key: str, default_name: str) -> None:
        current = ops.get(ops_key)
        if not current:
            return
        current_path = Path(current)
        if current_path.exists():
            return  # live pointer (local or linked source) — keep it
        local = ops_parent / current_path.name
        if local.exists():
            ops[ops_key] = str(local)
            return
        fallback = ops_parent / default_name
        if fallback.exists():
            ops[ops_key] = str(fallback)
            return
        # no live path available — clear the stale pointer so downstream
        # `if raw_file is None` / `reg_file.exists()` checks fire cleanly
        # instead of marching a bad path into suite2p.
        ops.pop(ops_key, None)

    _repoint_if_stale("raw_file", "data_raw.bin")
    _repoint_if_stale("chan2_file", "data_chan2.bin")
    _repoint_if_stale("reg_file", "data.bin")

    # ensure image arrays are numpy ndarrays (can become lists after dict merges)
    for key in ("meanImg", "meanImgE", "refImg", "max_proj", "Vcorr"):
        val = ops.get(key)
        if val is not None and not isinstance(val, np.ndarray):
            ops[key] = np.array(val)

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

    # run_registration governs whether we actually need data_raw.bin.
    # compute it here (up from the block that previously defined it) so
    # the raw_file check below can be relaxed when registration is off.
    run_registration = bool(ops.get("do_registration", True))
    run_detection = bool(ops.get("roidetect", True))

    raw_file = ops.get("raw_file")
    n_func = ops.get("nframes_chan1") or ops.get("nframes") or ops.get("n_frames")
    if n_func is None:
        raise KeyError("Missing nframes_chan1 / nframes / n_frames in ops")
    if run_registration and raw_file is None:
        raise KeyError(
            "Missing raw_file in ops — required when do_registration=1. "
            "Set do_registration=0 to run detection-only against an "
            "existing data.bin."
        )
    n_func = int(n_func)

    # reg_file may already point at a linked source binary; only pin it
    # to plane_dir/data.bin when ops has no pointer yet.
    existing_reg = ops.get("reg_file")
    if existing_reg and Path(existing_reg).exists():
        reg_file = Path(existing_reg)
    else:
        reg_file = ops_parent / "data.bin"
        ops["reg_file"] = str(reg_file)

    chan2_file = ops.get("chan2_file", "")
    use_chan2 = bool(chan2_file) and Path(chan2_file).exists()
    n_chan2 = int(ops.get("nframes_chan2", 0)) if use_chan2 else 0

    n_align = n_func if not use_chan2 else min(n_func, n_chan2)
    if n_align <= 0:
        raise ValueError("Non-positive frame count after alignment selection.")
    if use_chan2 and (n_func != n_chan2):
        print(
            f"[run_plane_bin] Trimming to {n_align} frames (func={n_func}, chan2={n_chan2})."
        )

    ops["functional_chan"] = 1
    ops["align_by_chan"] = 2 if use_chan2 else 1
    ops["nchannels"] = 2 if use_chan2 else 1
    # Fan out the trimmed frame count to every alias mbo_utilities also
    # writes (num_timepoints, num_frames, n_frames, T, nt, timepoints).
    # Without this, dual-channel trimming leaves nframes=n_align but the
    # other aliases stuck at the chan1 source value, and downstream
    # readers get inconsistent answers from the same ops dict.
    _set_frame_count_aliases(ops, n_align)
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

    # reset detection-derived parameters when re-running registration or detection
    # so compute_enhanced_mean_image() reinitializes them from diameter.
    # prevents inheriting spatscale_pix=0 from a failed run, which causes
    # meanImgE to be computed with a [1,1] filter (all 0.5 output).
    # (run_registration / run_detection were resolved earlier, near the
    # raw_file check — keep them in scope here.)
    if run_registration:
        # full reset: clear both registration and detection intermediates
        for key in ["spatscale_pix", "Vcorr", "Vmax", "Vmap", "Vsplit", "ihop"]:
            if key in ops:
                del ops[key]
    elif run_detection:
        # detection only: clear detection intermediates but keep registration outputs
        for key in ["spatscale_pix", "Vmax", "Vmap", "Vsplit", "ihop"]:
            if key in ops:
                del ops[key]

    reg_file_chan2 = ops_parent / "data_chan2_reg.bin" if use_chan2 else None

    ops["anatomical_red"] = False
    ops["chan2_thres"] = 0.1

    if ops.get("roidetect", True) and ops.get("anatomical_only", 0) > 0:
        # Estimate memory usage for Cellpose detection
        estimated_gb = (Ly * Lx * n_align * 2) / 1e9  # Rough estimate
        spatial_scale = ops.get("spatial_scale", 0)
        if spatial_scale > 0:
            estimated_gb /= spatial_scale**2

        if estimated_gb > 50:  # Warn for datasets > 50GB
            print(
                f"Large dataset warning: {estimated_gb:.1f} GB estimated for detection"
            )
            if spatial_scale == 0:
                print(
                    f"  Consider adding 'spatial_scale': 2 to reduce memory usage by 4x"
                )
            print(f"  Or reduce 'batch_size' (current: {ops.get('batch_size', 500)})")

    # When skipping registration, make sure data.bin is available and
    # synthesize the reg-output fields detection expects. The registered
    # binary may already exist locally, be linked to a source dir, or
    # need to be materialized from data_raw.bin.
    if not run_registration:
        import shutil

        reg_file_path = Path(reg_file)
        raw_file_path = Path(raw_file) if raw_file else None

        # materialize data.bin from data_raw.bin only when necessary
        if raw_file_path and raw_file_path.exists() and (
            not reg_file_path.exists() or reg_file_path.stat().st_size == 0
        ):
            print(f"  Materializing {reg_file_path.name} from {raw_file_path.name}")
            shutil.copy2(raw_file_path, reg_file_path)

        if not reg_file_path.exists():
            raise FileNotFoundError(
                f"Registration was skipped but no usable registered binary is "
                f"available. reg_file={reg_file_path}, raw_file={raw_file}. "
                "Either turn registration back on, point source_mode at a "
                "directory containing data.bin, or use source_mode='link' "
                "to reference an existing source binary."
            )

        # yrange/xrange: prefer exact padding bounds when the writer set
        # them; otherwise infer from whichever binary is available.
        if "_pad_yrange" in ops and "_pad_xrange" in ops:
            ops["yrange"] = list(ops["_pad_yrange"])
            ops["xrange"] = list(ops["_pad_xrange"])
            print(f"  Valid region from padding: yrange={ops['yrange']}, xrange={ops['xrange']}")
        elif "yrange" not in ops or "xrange" not in ops:
            use_anatomical = ops.get("anatomical_only", 0) > 0
            if use_anatomical:
                print(
                    "  Using full image dimensions for anatomical detection (avoids cropping issues)"
                )
                ops["yrange"] = [0, Ly]
                ops["xrange"] = [0, Lx]
            else:
                # prefer data_raw.bin for detection of valid region (it's
                # closer to acquisition), fall back to data.bin.
                detect_from = (
                    raw_file_path
                    if raw_file_path and raw_file_path.exists()
                    else reg_file_path
                )
                print(f"  Detecting valid region from {detect_from.name}...")
                with BinaryFile(Ly=Ly, Lx=Lx, filename=str(detect_from)) as f:
                    meanImg_full = f.sampled_mean().astype(np.float32)

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

        # registration outputs that detection expects
        if "badframes" not in ops:
            ops["badframes"] = np.zeros(n_align, dtype=bool)
        if "xoff" not in ops:
            ops["xoff"] = np.zeros(n_align, dtype=np.float32)
        if "yoff" not in ops:
            ops["yoff"] = np.zeros(n_align, dtype=np.float32)
        if "corrXY" not in ops:
            ops["corrXY"] = np.ones(n_align, dtype=np.float32)

        # Also materialize channel 2 registered binary if source chan2 exists
        if use_chan2:
            chan2_path = Path(chan2_file)
            reg_chan2_path = Path(reg_file_chan2)
            if chan2_path.exists():
                if not reg_chan2_path.exists() or reg_chan2_path.stat().st_size == 0:
                    print(f"  Copying {chan2_path.name} -> {reg_chan2_path.name}")
                    shutil.copy2(chan2_path, reg_chan2_path)
                else:
                    print(f"  {reg_chan2_path.name} already exists, skipping copy")

    # ensure summary images exist even when registration was skipped.
    # meanImg is needed by detection; meanImgE is only computed during
    # registration so we derive it here when missing.
    if not run_registration:
        if "meanImg" not in ops:
            with BinaryFile(
                Ly=Ly, Lx=Lx, filename=str(reg_file), n_frames=n_align
            ) as f_mean:
                ops["meanImg"] = f_mean.sampled_mean().astype(np.float32)
                print("  Computed meanImg from binary")

        if "meanImgE" not in ops and "meanImg" in ops:
            # compute on valid region only to avoid edge artifacts
            # from zero-padding border
            if "_pad_yrange" in ops and "_pad_xrange" in ops:
                yr, xr = ops["yrange"], ops["xrange"]
                full_mean = ops["meanImg"]
                ops["meanImg"] = full_mean[yr[0]:yr[1], xr[0]:xr[1]]
                enhanced_crop = _compute_enhanced_mean_image(None, ops)
                ops["meanImg"] = full_mean
                meanImgE = np.zeros_like(full_mean, dtype=np.float32)
                meanImgE[yr[0]:yr[1], xr[0]:xr[1]] = enhanced_crop
                ops["meanImgE"] = meanImgE
            else:
                ops["meanImgE"] = _compute_enhanced_mean_image(
                    ops["meanImg"].astype(np.float32), ops
                )
            print("  Computed meanImgE from meanImg")

    # for very short recordings, suite2p's bin_movie crashes when
    # bin_size > nframes (produces empty array). pre-compute meanImg so
    # detection_wrapper skips the failing reduction, and clamp tau so
    # bin_size <= nframes.
    tau_fs = np.round(ops.get("tau", 0.7) * ops.get("fs", 30))
    if n_align < max(tau_fs, 2):
        with BinaryFile(
            Ly=Ly, Lx=Lx, filename=str(reg_file), n_frames=n_align
        ) as f_pre:
            all_frames = f_pre.data[:n_align]
            ops["meanImg"] = all_frames.mean(axis=0).astype(np.float32)
            ops["max_proj"] = all_frames.max(axis=0).astype(np.float32)
        # set tau so bin_size = max(1, n//nbinned, round(tau*fs)) <= nframes
        ops["tau"] = 0

    has_pad_range = "_pad_yrange" in ops and "_pad_xrange" in ops

    # upstream's BinaryFile defaults to write=False (read-only) and errors on
    # missing files. reg_file / reg_file_chan2 are registration DESTINATIONS
    # — they may not exist yet on a fresh run — so we pass write=True to pick
    # up the "r+ if exists else w+" behavior the fork used to rely on.
    # raw_file / chan2_file are input sources (already produced by mbo's
    # imwrite or upstream copy) and stay read-only.
    # f_raw is only needed when registration runs; open it conditionally
    # so detection-only runs don't require data_raw.bin at all.
    _open_raw = (
        raw_file is not None
        and Path(raw_file).exists()
        and run_registration
    )
    with (
        BinaryFile(Ly=Ly, Lx=Lx, filename=str(reg_file), n_frames=n_align, write=True) as f_reg,
        (
            BinaryFile(Ly=Ly, Lx=Lx, filename=str(raw_file), n_frames=n_align)
            if _open_raw
            else nullcontext()
        ) as f_raw,
        (
            BinaryFile(Ly=Ly, Lx=Lx, filename=str(reg_file_chan2), n_frames=n_align, write=True)
            if use_chan2
            else nullcontext()
        ) as f_reg_chan2,
        (
            BinaryFile(Ly=Ly, Lx=Lx, filename=str(chan2_file), n_frames=n_align)
            if use_chan2
            else nullcontext()
        ) as f_raw_chan2,
    ):
        if has_pad_range and run_registration:
            # when data was padded for axial alignment, suite2p's
            # compute_crop only accounts for registration motion and
            # ignores the zero-padded border.  split pipeline into
            # registration-only then detection-only so we can clamp
            # xrange/yrange to the valid (non-padded) region in between.
            ops["roidetect"] = False
            print("NOTE: running registration-only pass (detection deferred)")
            ops = _call_upstream_pipeline(
                ops,
                f_reg, f_raw,
                f_reg_chan2 if use_chan2 else None,
                f_raw_chan2 if use_chan2 else None,
                run_registration=True,
                stat=None,
            )

            # clamp yrange/xrange so the valid window excludes ALL sources
            # of zero contamination:
            #   1. axial padding at the canvas borders (`_pad_yrange` /
            #      `_pad_xrange` = intersection of plane-shift bounds),
            #   2. per-frame registration motion trimming.
            #
            # upstream's `reg_yr` is computed assuming the full canvas is
            # valid pre-registration — it tells us how many rows at top
            # (`reg_yr[0]`) and bottom (`Ly - reg_yr[1]`) are unusable due
            # to motion alone. on a padded canvas those same motion trims
            # apply INSIDE the pad window, so we ADD them to pad_yr rather
            # than taking the intersection. naive `max(pad_yr[0], reg_yr[0])`
            # leaves a thin zero-contaminated strip that leaks into
            # max_proj / correlation images and into cellpose's input near
            # the edges. same logic for x.
            pad_yr = ops["_pad_yrange"]
            pad_xr = ops["_pad_xrange"]
            reg_yr = ops.get("yrange", [0, Ly])
            reg_xr = ops.get("xrange", [0, Lx])
            top_trim = int(max(0, reg_yr[0]))
            bot_trim = int(max(0, Ly - reg_yr[1]))
            left_trim = int(max(0, reg_xr[0]))
            right_trim = int(max(0, Lx - reg_xr[1]))
            ops["yrange"] = [
                int(pad_yr[0]) + top_trim,
                int(pad_yr[1]) - bot_trim,
            ]
            ops["xrange"] = [
                int(pad_xr[0]) + left_trim,
                int(pad_xr[1]) - right_trim,
            ]
            # defensive: ensure at least a 1-pixel window (extreme motion
            # on thin pads could collapse the window to zero size).
            ops["yrange"][1] = max(ops["yrange"][0] + 1, ops["yrange"][1])
            ops["xrange"][1] = max(ops["xrange"][0] + 1, ops["xrange"][1])
            print(
                f"  Clamped valid region: yrange={ops['yrange']} "
                f"xrange={ops['xrange']} "
                f"(pad={list(pad_yr)},{list(pad_xr)}; "
                f"motion_trim=T{top_trim}/B{bot_trim}/L{left_trim}/R{right_trim})"
            )

            # recompute meanImgE on the valid region only so the
            # high-pass filter doesn't create edge artifacts at the
            # zero-padding boundary.  compute_enhanced_mean_image reads
            # ops["meanImg"] internally, so swap it temporarily.
            yr, xr = ops["yrange"], ops["xrange"]
            full_mean = ops["meanImg"]
            ops["meanImg"] = full_mean[yr[0]:yr[1], xr[0]:xr[1]]
            enhanced_crop = _compute_enhanced_mean_image(None, ops)
            ops["meanImg"] = full_mean  # restore
            meanImgE = np.zeros_like(full_mean, dtype=np.float32)
            meanImgE[yr[0]:yr[1], xr[0]:xr[1]] = enhanced_crop
            ops["meanImgE"] = meanImgE

            # upstream's pipeline(run_registration=False) reloads reg_outputs.npy
            # from disk and uses ITS yrange/xrange for detection — not whatever
            # we just clamped into the in-memory ops dict. if we don't update
            # the on-disk file, detection sees the full (pre-clamp) valid
            # region and cellpose happily picks up ROIs in the zero-padded
            # corners. patch reg_outputs.npy in place with the clamped values
            # and the clamped meanImgE before the detection-only second call.
            _reg_save_dir = Path(ops["save_path"])
            _reg_outputs_path = _reg_save_dir / "reg_outputs.npy"
            if _reg_outputs_path.exists():
                _reg_disk = np.load(_reg_outputs_path, allow_pickle=True).item()
                _reg_disk["yrange"] = ops["yrange"]
                _reg_disk["xrange"] = ops["xrange"]
                _reg_disk["meanImgE"] = meanImgE
                np.save(_reg_outputs_path, _reg_disk, allow_pickle=True)
                print(f"  Updated {_reg_outputs_path.name} with clamped valid region")

            if ops.get("ops_path"):
                save_ops_db_settings(ops["ops_path"], ops)

            # now run detection + extraction + classification
            ops["roidetect"] = True
            ops["do_registration"] = 0
            ops = _call_upstream_pipeline(
                ops,
                f_reg, f_raw,
                f_reg_chan2 if use_chan2 else None,
                f_raw_chan2 if use_chan2 else None,
                run_registration=False,
                stat=None,
            )
        else:
            ops = _call_upstream_pipeline(
                ops,
                f_reg, f_raw,
                f_reg_chan2 if use_chan2 else None,
                f_raw_chan2 if use_chan2 else None,
                run_registration=run_registration,
                stat=None,
            )

    # ensure meanImgE is always present in final ops (safety net)
    if "meanImgE" not in ops and "meanImg" in ops:

        if "_pad_yrange" in ops and "_pad_xrange" in ops:
            yr, xr = ops["yrange"], ops["xrange"]
            full_mean = ops["meanImg"]
            ops["meanImg"] = full_mean[yr[0]:yr[1], xr[0]:xr[1]]
            enhanced_crop = _compute_enhanced_mean_image(None, ops)
            ops["meanImg"] = full_mean
            meanImgE = np.zeros_like(full_mean, dtype=np.float32)
            meanImgE[yr[0]:yr[1], xr[0]:xr[1]] = enhanced_crop
            ops["meanImgE"] = meanImgE
        else:
            ops["meanImgE"] = _compute_enhanced_mean_image(
                ops["meanImg"].astype(np.float32), ops
            )

    if use_chan2:
        ops["reg_file_chan2"] = str(reg_file_chan2)
    save_ops_db_settings(ops["ops_path"], ops)
    return True


def _cleanup_bin_files(plane_dir, keep_raw, keep_reg):
    """delete bin files if user doesn't want them, swallowing OS errors."""
    if not keep_raw:
        try:
            (plane_dir / "data_raw.bin").unlink(missing_ok=True)
        except OSError:
            pass
    if not keep_reg:
        try:
            (plane_dir / "data.bin").unlink(missing_ok=True)
        except OSError:
            pass


def run_plane(
    input_data,
    save_path: str | Path | None = None,
    ops: dict | str | Path = None,
    db: dict | None = None,
    settings: dict | None = None,
    chan2_file: str | Path | None = None,
    keep_raw: bool = False,
    keep_reg: bool = True,
    force_reg: bool = False,
    force_detect: bool = False,
    frame_indices: list | None = None,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    accept_all_cells: bool = False,
    cell_filters: list = None,
    save_json: bool = False,
    plane_name: str | None = None,
    source_mode: str = "auto",
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
        Default is no filters.
        Example: [{"name": "max_diameter", "min_diameter_um": 4, "max_diameter_um": 35}]
    save_json : bool, default False
        Save ops as JSON.
    frame_indices : list[int], optional
        Explicit 0-based timepoint indices. Supports stride
        (e.g. ``list(range(0, 1574, 2))`` for every other frame).
        When provided, the binary on disk contains exactly these
        frames, and `OutputMetadata` reactively scales `fs` in ops.npy
        based on the implicit stride. Takes precedence over any
        ``num_frames`` in ``writer_kwargs``.
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

    progress_callback = kwargs.pop("progress_callback", None)

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
            raise ValueError(
                "plane_name is required when input is an array without filenames."
            )
        else:
            # dummy path for internal logic compatibility
            input_path = Path(f"{plane_name}.tif")
    elif isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
    else:
        raise TypeError(
            f"input_data must be path or lazy array, got {type(input_data)}"
        )

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

    # load ops defaults and user settings first (needed for plane number).
    # accepts EITHER ops=... (legacy flat dict) OR db=... / settings=... (new
    # upstream-shaped dicts). when both are supplied, db/settings are
    # flattened first and any explicit ops keys win the merge.
    from lbm_suite2p_python.db_settings import merge_db_settings_into_ops
    ops_default = default_ops()
    ops_user = load_ops(ops) if ops else {}
    if db is not None or settings is not None:
        ops_user = merge_db_settings_into_ops(ops_user, db, settings)
    ops = {**ops_default, **ops_user, "data_path": str(input_path.resolve())}

    # normalize kwargs
    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}

    # determine if we're processing existing binary in-place
    if binary_with_ops and (
        input_path.parent == save_path or save_path == input_parent
    ):
        # resolve the plane-specific source dir. _resolve_source_plane_dir
        # raises on ambiguity / no match instead of silently using input_path.parent.
        target_plane = ops.get("plane", 1)
        source_plane_num = ops.get("source_plane_num")
        plane_dir = _resolve_source_plane_dir(
            input_arr, input_path, target_plane, source_plane_num=source_plane_num
        )
        print(f"Processing existing binary in-place: {plane_dir}")
        skip_imwrite = True
        ops_file = plane_dir / "ops.npy"
        existing_ops = (
            np.load(ops_file, allow_pickle=True).item() if ops_file.exists() else {}
        )
        metadata = {
            k: v
            for k, v in existing_ops.items()
            if k in ("plane", "fs", "dx", "dy", "dz", "z_step", "Ly", "Lx", "nframes")
        }
        file = None
        # merge: defaults < existing (registration results, images, etc.) < user overrides
        ops = {
            **ops_default,
            **existing_ops,
            **ops_user,
            "data_path": str(input_path.resolve()),
        }
    elif binary_with_ops:
        # binary input with ops but different save_path: stage files into
        # the new plane_dir per source_mode, instead of re-encoding.
        skip_imwrite = True
        file = None

        # resolve the plane-specific source dir (raises on ambiguity / miss).
        target_plane = ops.get("plane", 1)
        source_plane_num = ops.get("source_plane_num")
        src_dir = _resolve_source_plane_dir(
            input_arr, input_path, target_plane, source_plane_num=source_plane_num
        )

        existing_ops = np.load(src_dir / "ops.npy", allow_pickle=True).item()
        metadata = {
            k: v
            for k, v in existing_ops.items()
            if k in ("plane", "fs", "dx", "dy", "dz", "z_step", "Ly", "Lx", "nframes")
        }

        # determine plane number for directory naming
        if "plane" in ops:
            plane = ops["plane"]
        elif "plane" in existing_ops:
            plane = existing_ops["plane"]
        else:
            tag = derive_tag_from_filename(input_path)
            plane = get_plane_num_from_tag(tag, fallback=1)

        # source_plane_num (set by run_volume when the loaded array came
        # from a plane-subset save) overrides `plane` for dir naming only.
        # keeps slicing (which uses ops["plane"] as a local index) intact.
        naming_plane = ops.get("source_plane_num", plane)

        if plane_name is not None:
            subdir_name = plane_name
        else:
            nframes_hint = existing_ops.get("nframes_chan1") or existing_ops.get(
                "nframes"
            )
            subdir_name = generate_plane_dirname(plane=naming_plane, nframes=nframes_hint)

        plane_dir = save_path / subdir_name
        plane_dir.mkdir(exist_ok=True)
        ops_file = plane_dir / "ops.npy"

        # merge existing ops with user overrides BEFORE staging, so the
        # stage call can mutate link-mode path pointers on the final ops
        # dict without them being wiped by a later merge.
        ops = {
            **ops_default,
            **existing_ops,
            **ops_user,
            "data_path": str(input_path.resolve()),
        }

        # link mode can't safely co-exist with do_registration=1 because
        # registration would need to write the source data.bin. flip to
        # copy_reg and warn instead of corrupting the source binary.
        _effective_mode = source_mode
        if _effective_mode == "link" and ops_user.get("do_registration", 1):
            logger.warning(
                "source_mode='link' requires do_registration=0 (link mode "
                "is read-only from the source binary). Falling back to "
                "'copy_reg' so registration can proceed safely."
            )
            _effective_mode = "copy_reg"

        # stage ops.npy + detection outputs + binaries per mode. ops is
        # mutated in place for link mode to point raw_file/reg_file at
        # the source paths; for copy_reg mode a stale raw_file pointer
        # is popped so the later plane_dir path-pinning doesn't resurrect
        # it from a non-existent plane_dir/data_raw.bin.
        _stage_source_into_plane_dir(
            src_dir, plane_dir, _effective_mode, ops,
        )
    else:
        skip_imwrite = False

        # determine plane number for directory naming (from ops or input filename)
        if "plane" in ops:
            plane = ops["plane"]
        else:
            tag = derive_tag_from_filename(input_path)
            plane = get_plane_num_from_tag(tag, fallback=1)

        # source_plane_num (set by run_volume when the loaded array came
        # from a plane-subset save) overrides `plane` for dir naming only.
        naming_plane = ops.get("source_plane_num", plane)

        # compute plane_dir early so we can check if binaries already exist
        if plane_name is not None:
            subdir_name = plane_name
        else:
            # prefer the user-specified frame limit over raw array shape
            nframes_hint = (
                writer_kwargs.get("num_frames")
                or ops.get("nframes")
            )
            if not nframes_hint and input_arr is not None and hasattr(input_arr, "shape"):
                nframes_hint = input_arr.shape[0]
            # if input was a path, peek at metadata for frame count
            if not nframes_hint and input_arr is None and input_path is not None:
                try:
                    from mbo_utilities import get_metadata
                    md = get_metadata(input_path)
                    nframes_hint = (
                        md.get("num_timepoints")
                        or md.get("nframes")
                        or md.get("num_frames")
                    )
                except Exception:
                    nframes_hint = None
            subdir_name = generate_plane_dirname(
                plane=naming_plane,
                nframes=nframes_hint,
            )

        plane_dir = save_path / subdir_name
        plane_dir.mkdir(exist_ok=True)
        ops_file = plane_dir / "ops.npy"

        # extract expected dims from input for cache validation
        # honors writer_kwargs["num_frames"] limit if user requested fewer frames
        exp_nframes = writer_kwargs.get("num_frames")
        exp_ly = exp_lx = None
        if input_arr is not None and hasattr(input_arr, "shape"):
            if exp_nframes is None:
                exp_nframes = input_arr.shape[0]
            exp_ly = input_arr.shape[-2]
            exp_lx = input_arr.shape[-1]

        # check if binaries already exist in the actual output directory
        should_write = _should_write_bin(
            ops_file,
            force=force_reg,
            expected_nframes=exp_nframes,
            expected_ly=exp_ly,
            expected_lx=exp_lx,
        )
        if should_write:
            if input_arr is not None:
                file = input_arr
            else:
                file = imread(input_path, **reader_kwargs)

            if hasattr(file, "metadata"):
                metadata = file.metadata
            else:
                metadata = get_metadata(input_path)
        else:
            print(
                f"Skipping data_raw.bin write, already exists and passes data validation checks."
            )
            file = None
            # load metadata from existing ops if available
            if ops_file.exists():
                existing_ops = np.load(ops_file, allow_pickle=True).item()
                metadata = {
                    k: v
                    for k, v in existing_ops.items()
                    if k in ("plane", "fs", "dx", "dy", "dz", "z_step", "Ly", "Lx", "nframes")
                }
                # ensure registration metadata from previous runs is preserved
                ops = {
                    **ops_default,
                    **existing_ops,
                    **ops_user,
                    "data_path": str(input_path.resolve()),
                }
            else:
                metadata = {}

    # plane number handling (finalize)
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

    # propagate voxel size from metadata into ops (user ops take precedence via setdefault)
    vs = get_voxel_size(metadata)
    if vs.dz is not None:
        ops.setdefault("dz", vs.dz)
        ops.setdefault("z_step", vs.dz)
    if vs.dx != 1.0:
        ops.setdefault("dx", vs.dx)
    if vs.dy != 1.0:
        ops.setdefault("dy", vs.dy)

    # Propagate fs from source metadata into ops the same way dz/dx/dy
    # are propagated above. WITHOUT this, lbm's `default_ops()` ships
    # `fs=10.0` and that hardcoded default wins over the actual source
    # acquisition fs when the metadata kwarg merge happens inside
    # mbo's `imwrite` (the kwarg takes precedence over arr.metadata).
    # Result: any downstream reactive scaling — including the writer's
    # OutputMetadata fs/stride scaling — divides 10 instead of the
    # real fs, producing nonsense like fs=5 from a 30Hz source +
    # every-other-frame stride.
    src_fs = metadata.get("fs") if metadata else None
    if src_fs is None and file is not None and hasattr(file, "metadata"):
        src_fs = (file.metadata or {}).get("fs")
    if src_fs is not None:
        # only override the lbm default — don't clobber a value the
        # user explicitly set in their ops_user dict.
        if ops.get("fs") in (None, 10.0) or "fs" not in ops_user:
            ops["fs"] = float(src_fs)

    # When called directly (not via run_volume), the caller may pass
    # frame_indices for stride scaling. run_volume already invoked the
    # reactive helper before reaching us, so re-running it here is a
    # no-op when there's no T stride. When there IS a T stride from
    # this entry point, scale fs from the source metadata via the
    # reactive helper. plane_indices is None here because we only see
    # one plane at a time at this layer.
    if frame_indices is not None and file is not None:
        _src_meta = (
            dict(file.metadata) if hasattr(file, "metadata") else None
        )
        _src_shape = (
            tuple(file.shape5d) if hasattr(file, "shape5d") else None
        )
        _apply_reactive_metadata(
            ops=ops,
            source_metadata=_src_meta,
            source_shape=_src_shape,
            frame_indices=frame_indices,
            plane_indices=None,  # this layer only sees one plane
            logger=logger,
        )

    # store source filename info in ops for traceability
    ops["source_dirname"] = plane_dir.name
    ops["source_input"] = str(input_path.name)

    # Write binary
    if progress_callback:
        progress_callback(step="writing_binary", message="Writing binary...")
    if not skip_imwrite and file is not None:
        md_combined = {**metadata, **ops}
        print(f"Writing binary to {plane_dir}...")
        bin_start = time.time()
        # extract single plane if data is volumetric
        write_planes = [plane] if _get_num_planes(file) > 1 else None

        # apply per-plane axial shift if plane_shifts is present in metadata
        write_kw = dict(writer_kwargs)
        # If the caller gave us explicit frame indices, pass them as
        # `frames=` (1-based) to imwrite. This wins over any stale
        # `num_frames` truncation in writer_kwargs — strided semantics
        # require an explicit index list, not a count.
        if frame_indices is not None:
            write_kw["frames"] = [int(i) + 1 for i in frame_indices]
            write_kw.pop("num_frames", None)
        if md_combined.get("apply_shift") and md_combined.get("plane_shifts") is not None:
            from mbo_utilities._writers import load_registration_shifts, compute_pad_from_shifts

            _apply, _plane_shifts, _ = load_registration_shifts(md_combined)
            if _apply and _plane_shifts is not None:
                plane_0idx = plane - 1
                if plane_0idx < len(_plane_shifts):
                    sv = _plane_shifts[plane_0idx]
                    write_kw["shift_vector"] = sv
                    write_kw["all_plane_shifts"] = _plane_shifts
                    # use global padding so all planes share the same output dims
                    pt, pb, pl, pr = compute_pad_from_shifts(_plane_shifts)
                    if hasattr(file, "shape"):
                        Ly_orig = file.shape[-2]
                        Lx_orig = file.shape[-1]
                        md_combined["Ly"] = Ly_orig + pt + pb
                        md_combined["Lx"] = Lx_orig + pl + pr
                    print(f"  Applying axial shift for plane {plane}: {sv}")

        imwrite(
            file,
            plane_dir,
            ext=".bin",
            metadata=md_combined,
            register_z=False,
            output_name="data_raw.bin",
            overwrite=True,
            planes=write_planes,
            show_progress=False,
            **write_kw,
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
            extra={"plane": plane, "shape": list(file.shape)},
        )
        save_ops_db_settings(ops_file, ops)

    # Determine processing needs.
    # explicit user toggles win over file-state inference: when the caller
    # turned off registration AND detection (and didn't pass a force flag),
    # honor the intent and skip suite2p entirely regardless of what's on
    # disk. post-processing below still runs and guards its own inputs.
    user_skip_reg = (not ops.get("do_registration", 1)) and not force_reg
    user_skip_detect = (not ops.get("roidetect", 1)) and not force_detect

    stat_file = plane_dir / "stat.npy"

    if user_skip_reg and user_skip_detect:
        needs_reg = False
        needs_detect = False
    else:
        # detection: needed when stat.npy is missing or empty, unless the
        # user explicitly disabled roidetect (file state can't fabricate
        # detection results).
        needs_detect = False
        if force_detect:
            needs_detect = True
        elif not stat_file.exists():
            needs_detect = not user_skip_detect
        elif ops.get("roidetect", 1):
            stat = np.load(stat_file, allow_pickle=True)
            if stat is None or len(stat) == 0:
                needs_detect = True

        # registration: needed when ops is missing or _should_register says
        # so, unless the user explicitly disabled do_registration.
        if force_reg:
            needs_reg = True
        elif user_skip_reg:
            needs_reg = False
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
            imwrite(
                chan2_data,
                plane_dir,
                ext=".bin",
                metadata=chan2_md,
                register_z=False,
                structural=True,
                show_progress=False,
            )
            ops["chan2_file"] = str((plane_dir / "data_chan2.bin").resolve())
            # Use shape5d (the always-5D contract) so natural-rank arrays
            # (e.g. a 4D TZYX TiffArray with C=1 squeezed) report the
            # real T dimension. shape[0] picks Y on a natural 4D array
            # and crashes/mis-counts on a 2D one — same class of bug
            # mbo_utilities just fixed in `_imwrite_base`.
            if hasattr(chan2_data, "shape5d"):
                ops["nframes_chan2"] = int(chan2_data.shape5d[0])
            elif hasattr(chan2_data, "shape"):
                ops["nframes_chan2"] = int(chan2_data.shape[0])
            else:
                ops["nframes_chan2"] = 0
            ops["nchannels"] = 2
            ops["align_by_chan"] = 2

    # ensure ops paths point to current plane_dir before running suite2p.
    # handles cases where data was moved from its original location.
    ops["ops_path"] = str(ops_file)
    ops["save_path"] = str(plane_dir.resolve())
    raw_bin = plane_dir / "data_raw.bin"
    if raw_bin.exists():
        ops["raw_file"] = str(raw_bin)
    reg_bin = plane_dir / "data.bin"
    if reg_bin.exists():
        ops["reg_file"] = str(reg_bin)
    save_ops_db_settings(ops_file, ops)

    # Run Suite2p (skip entirely when nothing needs to be done)
    skip_suite2p = not needs_reg and not needs_detect
    if skip_suite2p:
        if user_skip_reg and user_skip_detect:
            print("  Suite2p disabled by user toggles; regenerating figures only.")
        else:
            print("  Registration and detection already complete, skipping suite2p.")
        print("  Re-generating post-processing and figures...")
        # when the user toggled everything off and there's nothing to
        # regenerate from, bail cleanly instead of letting each
        # post-processing step emit its own "file not found" warning.
        if user_skip_reg and user_skip_detect and not stat_file.exists():
            print(
                f"  Skipping plane: {plane_dir.name} has no stat.npy — "
                "nothing to regenerate with registration and detection off."
            )
            return ops_file
    else:
        if progress_callback:
            progress_callback(step="suite2p", message="Running suite2p...")
        try:
            s2p_start = time.time()
            processed = run_plane_bin(ops)

            if processed:
                updated_ops = load_ops(ops_file)
                _add_processing_step(
                    updated_ops,
                    "suite2p_pipeline",
                    duration_seconds=time.time() - s2p_start,
                    extra={
                        "do_registration": updated_ops.get("do_registration", 1),
                        "n_cells": len(
                            np.load(plane_dir / "stat.npy", allow_pickle=True)
                        )
                        if (plane_dir / "stat.npy").exists()
                        else 0,
                    },
                )
                save_ops_db_settings(ops_file, updated_ops)

        except Exception as e:
            print(f"Error in run_plane_bin: {e}")
            traceback.print_exc()
            _cleanup_bin_files(plane_dir, keep_raw, keep_reg)
            raise

        if not processed:
            _cleanup_bin_files(plane_dir, keep_raw, keep_reg)
            return ops_file

    # --- Post-Processing ---
    if progress_callback:
        progress_callback(step="postprocessing", message="Post-processing...")

    # Load original iscell first so we have the true suite2p classification
    iscell_original = None
    iscell_file = plane_dir / "iscell.npy"
    if iscell_file.exists():
        iscell_original = np.load(iscell_file, allow_pickle=True)

    # 1. Accept All Cells
    if accept_all_cells:
        if iscell_original is not None:
            np.save(plane_dir / "iscell_suite2p.npy", iscell_original.copy())
            iscell = iscell_original.copy()
            iscell[:, 0] = 1
            np.save(iscell_file, iscell)
            print("  Marked all ROIs as accepted.")

    # 2. Cell Filtering
    if cell_filters:
        logger.warning("Cell filtering is not ready yet. Filters will be ignored.")
        # The following block is kept for reference as requested, but we skip executing the actual filters
        if False:
            print(f"  Applying cell filters: {[f['name'] for f in cell_filters]}")
            filter_start = time.time()
            try:
                # Bug fixed: iscell_original is already loaded above, before accept_all_cells mutation
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
                    extra={"n_removed": int(removed_mask.sum())},
                )
                # convert filter_results list to dict keyed by filter name
                filter_metadata = {}
                for r in filter_results:
                    name = r["name"]
                    config = r.get("config", {})
                    info = r.get("info", {})
                    removed = r.get("removed_mask", np.zeros(0, dtype=bool))
                    # build params from config (user-specified) or info (computed)
                    params = {}
                    for key in [
                        "min_diameter_um",
                        "max_diameter_um",
                        "min_diameter_px",
                        "max_diameter_px",
                        "min_area_px",
                        "max_area_px",
                        "min_mult",
                        "max_mult",
                        "max_ratio",
                    ]:
                        if key in config and config[key] is not None:
                            val = config[key]
                            params[key] = (
                                round(val, 1) if isinstance(val, float) else val
                            )
                    if not params:
                        for key in [
                            "min_px",
                            "max_px",
                            "min_ratio",
                            "max_ratio",
                            "lower_px",
                            "upper_px",
                        ]:
                            if key in info and info[key] is not None:
                                params[key] = round(info[key], 1)
                    filter_metadata[name] = {
                        "params": params,
                        "n_rejected": int(removed.sum()),
                    }
                updated_ops["filter_metadata"] = filter_metadata
                save_ops_db_settings(ops_file, updated_ops)

                # Plots
                try:
                    fig = plot_filtered_cells(
                        plane_dir,
                        iscell_original,
                        iscell_filtered,
                        save_path=plane_dir / "13_filtered_cells.png",
                    )
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                    plot_filter_exclusions(
                        plane_dir, iscell_filtered, filter_results, save_dir=plane_dir
                    )
                    plot_cell_filter_summary(
                        plane_dir, save_path=plane_dir / "15_filter_summary.png"
                    )
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
        F_corr = F - 0.7 * Fneu  # Fixed neucoeff for now, could be parameter

        current_ops = load_ops(ops_file)
        dff = dff_rolling_percentile(
            F_corr,
            window_size=dff_window_size,
            percentile=dff_percentile,
            smooth_window=dff_smooth_window,
            fs=current_ops.get("fs", 30.0),
            tau=current_ops.get("tau", 1.0),
        )
        np.save(plane_dir / "dff.npy", dff)

        _add_processing_step(
            current_ops,
            "dff_calculation",
            duration_seconds=time.time() - dff_start,
            extra={"percentile": dff_percentile},
        )
        save_ops_db_settings(ops_file, current_ops)

    # 3b. ROI statistics
    try:
        from lbm_suite2p_python.postprocessing import compute_roi_stats

        print("  Computing ROI statistics...")
        compute_roi_stats(plane_dir)
    except Exception as e:
        print(f"  Warning: ROI stats computation failed: {e}")

    # 4. Plots and Cleanup
    try:
        plot_zplane_figures(
            plane_dir,
            dff_percentile=dff_percentile,
            dff_window_size=dff_window_size,
            dff_smooth_window=dff_smooth_window,
        )
    except Exception as e:
        print(f"  Warning: Plot generation failed: {e}")

    if save_json:
        ops_to_json(ops_file)

    _cleanup_bin_files(plane_dir, keep_raw, keep_reg)

    # PC metrics
    save_pc_panels_and_metrics(ops_file, plane_dir / "pc_metrics")

    if progress_callback:
        progress_callback(step="done", message="Plane complete")

    return ops_file


def grid_search(*args, **kwargs):
    """Run a grid search over Suite2p parameters. See lbm_suite2p_python.grid_search module."""
    from lbm_suite2p_python.grid_search import grid_search as _grid_search

    return _grid_search(*args, **kwargs)
