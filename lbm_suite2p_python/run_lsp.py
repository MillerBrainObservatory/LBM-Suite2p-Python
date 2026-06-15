import logging
import logging.handlers
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import traceback
import copy

import numpy as np

from lbm_suite2p_python.default_ops import default_ops
from lbm_suite2p_python.db_settings import save_ops_db_settings, reconcile_detection_keys
from lbm_suite2p_python.postprocessing import (
    ops_to_json,
    load_ops,
    dff_rolling_percentile,
    zscore_trace,
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
        from lbm_suite2p_python.db_settings import _ensure_diameter_array
        for k, v in detect_outputs.items():
            if k == "diameter" and v is not None:
                ops[k] = _ensure_diameter_array(v)
            elif k != "diameter":
                ops[k] = v
    ops["plane_times"] = plane_times
    return ops


def compute_enhanced_mean_image(img, ops):
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
    "spks.npy", "norm_traces.npy", "redcell.npy", "zcorr.npy",
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

    Returns True when a copy actually happened. Never overwrites a
    non-empty destination.
    """
    import shutil
    if not src.exists():
        return False
    if dst.exists():
        if not (overwrite_empty and dst.stat().st_size == 0):
            return False
    shutil.copy2(src, dst)
    return True


def _stage_source_into_plane_dir(
    src_dir: Path,
    plane_dir: Path,
    ops: dict,
) -> None:
    """Stage files from a source plane dir into ``plane_dir``.

    copies ops.npy and detection outputs into ``plane_dir`` so figures
    and post-processing read locally. binaries (data.bin, data_raw.bin,
    data_chan2.bin) are NOT copied — ops['raw_file'] / ops['reg_file'] /
    ops['chan2_file'] are repointed at the source paths so the new
    output dir stays small and can be copied around without dragging
    the binary along. source binaries are read-only from this plane's
    perspective.

    ``ops`` is mutated in place to point at the source binaries.
    """
    src_dir = Path(src_dir)
    plane_dir = Path(plane_dir)
    plane_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Staging plane outputs: {src_dir.name} -> {plane_dir.name}")
    _copy_if_needed(src_dir / "ops.npy", plane_dir / "ops.npy")
    for fname in _DETECTION_OUTPUT_FILES:
        _copy_if_needed(src_dir / fname, plane_dir / fname)

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
    plot_zplane_figures,
    plot_filtered_cells,
    plot_filter_exclusions,
    plot_cell_filter_summary,
    plot_volume_accepted_rejected_overlay,
)

DEFAULT_CELL_FILTERS = []
from mbo_utilities.log import get as get_logger
from mbo_utilities.metadata import get_voxel_size


logger = get_logger("run_lsp")


_external_logging_attached = False


def _attach_external_loggers(level: int = logging.INFO) -> None:
    """Surface suite2p / cellpose progress messages on stdout.

    Mainline suite2p (`suite2p.detection.anatomical`,
    `suite2p.registration.*`) and cellpose (`cellpose.models`) use plain
    `logging.getLogger(__name__)` loggers with no handlers attached. Their
    `logger.info(...)` calls — registration progress, ">>>> CELLPOSE finding
    masks", median-diameter reports — propagate to root and get silently
    dropped by the default WARNING-level lastResort handler. Without this
    hookup the user can't tell whether detection is running or what params
    cellpose actually picked. Idempotent, called once per pipeline entry.
    """
    global _external_logging_attached
    if _external_logging_attached:
        return
    fmt = logging.Formatter("%(name)s: %(message)s")
    for name in ("suite2p", "cellpose"):
        lg = logging.getLogger(name)
        if lg.level == logging.NOTSET or lg.level > level:
            lg.setLevel(level)
        has_stream = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in lg.handlers
        )
        if not has_stream:
            h = logging.StreamHandler()
            h.setFormatter(fmt)
            h.setLevel(level)
            lg.addHandler(h)
        lg.propagate = False
    _external_logging_attached = True


def _is_valid_torch_checkpoint(path) -> bool:
    """Cheap integrity check for a cellpose/torch checkpoint (a zip archive).

    A download interrupted mid-write leaves a truncated file with no zip
    end-of-central-directory record; is_zipfile catches that case.
    """
    import zipfile

    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def _prewarm_cellpose_model(ops) -> None:
    """Download the cellpose model once, in the parent, before workers fan out.

    cellpose's model cache downloads to a temp file then renames with no
    cross-process lock. Multiple workers hitting an empty cache at once
    race: one wins the rename, the rest fail (Windows WinError 32/183) or read a
    half-written file (PytorchStreamReader miniz error). Warming here serializes
    the download so workers only ever read a complete file. A corrupt leftover
    from a prior failed run is removed and re-downloaded.
    """
    if not (ops.get("roidetect", True)
            and (ops.get("anatomical_only", 0) > 0 or ops.get("algorithm") == "cellpose")):
        return
    try:
        from cellpose import models as cp_models
    except Exception:
        return  # cellpose absent; detection falls back to functional mode
    cached = Path(cp_models.MODEL_DIR) / "cpsam"
    if cached.exists() and not _is_valid_torch_checkpoint(cached):
        print(f"Cellpose model cache corrupt, re-downloading: {cached}")
        try:
            cached.unlink()
        except OSError:
            pass
    try:
        # cellpose 4.x renamed cache_CPSAM_model_path() -> cache_model_path(backbone)
        if hasattr(cp_models, "cache_model_path"):
            cp_models.cache_model_path("cpsam")
        else:
            cp_models.cache_CPSAM_model_path()
    except Exception as exc:
        print(
            f"Warning: could not pre-download cellpose model ({exc}); "
            "workers may race on first download"
        )


from lbm_suite2p_python.volume import (
    plot_volume_diagnostics,
    plot_orthoslices,
    plot_3d_roi_map,
    plot_3d_rastermap_clusters,
    plot_volume_trace_figures,
    get_volume_stats,
)

PIPELINE_TAGS = ("plane", "roi", "z", "plane_", "roi_", "z_")

from lbm_suite2p_python.utils import _is_lazy_array, _get_num_planes


def _get_suite2p_version():
    """Get suite2p version string."""
    try:
        return version("suite2p")
    except PackageNotFoundError:
        return "not installed"


def _resolve_gpu_env() -> None:
    """Honor MBO_GPU -> CUDA_VISIBLE_DEVICES before torch/cupy/cellpose init.

    Lets ``MBO_GPU=0 lsp ...`` (or programmatic use) force CPU across suite2p
    and cellpose without per-call device args; ``MBO_GPU=N`` pins a device.
    No-op when MBO_GPU is unset. Entry points call this first so the env is
    set before torch initializes CUDA, and before workers are spawned (they
    inherit it).
    """
    raw = os.environ.get("MBO_GPU")
    if raw is None:
        return
    try:
        from mbo_utilities.gpu import apply_gpu_policy
        apply_gpu_policy(raw)
        return
    except Exception:
        pass
    token = raw.strip().lower()
    if token in ("0", "off", "false", "no", "cpu", "none"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif token and token.replace(",", "").isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = token


def _apply_thread_limits(threads_per_worker: int | None) -> None:
    """Cap BLAS / OMP / numba / torch thread counts per process.

    Sets env vars so any child process spawned later inherits the cap,
    and calls torch.set_num_threads for the current process (where the
    BLAS env vars may have been read at import time and cannot be
    changed without threadpoolctl). No-op when value is None or <= 0.
    """
    if not threads_per_worker or threads_per_worker <= 0:
        return
    n = str(int(threads_per_worker))
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[var] = n
    try:
        import torch
        torch.set_num_threads(int(threads_per_worker))
    except Exception:
        pass


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


def _extract_rastermap_section(rastermap_kwargs, section):
    """Pull the sub-dict for "planar" or "volumetric" from a unified rastermap_kwargs.

    Returns None when the section was not requested. Returns an empty dict
    (meaning "use defaults") when the key is present with a None / empty value.
    """
    if not rastermap_kwargs or section not in rastermap_kwargs:
        return None
    sub = rastermap_kwargs[section]
    return {} if sub is None else dict(sub)


def _extract_planar_rastermap_kwargs(rastermap_kwargs):
    return _extract_rastermap_section(rastermap_kwargs, "planar")


def _extract_volumetric_rastermap_kwargs(rastermap_kwargs):
    return _extract_rastermap_section(rastermap_kwargs, "volumetric")


def _resolve_input_source(input_arr, input_data):
    """Return a picklable source spec for parallel workers to imread().

    Workers re-open the data themselves rather than receiving a pickled
    lazy array (which may not survive spawn). Preference order:
      1. Original input_data if it's a str/Path/list of paths.
      2. input_arr.filenames for mbo lazy arrays backed by files.
    Raises ValueError when neither is available.
    """
    if isinstance(input_data, (str, Path)):
        return str(input_data)
    if isinstance(input_data, (list, tuple)) and all(
        isinstance(p, (str, Path)) for p in input_data
    ):
        return [str(p) for p in input_data]
    if input_arr is not None and hasattr(input_arr, "filenames"):
        fns = list(input_arr.filenames) if input_arr.filenames else []
        if fns:
            return [str(p) for p in fns]
    raise ValueError(
        "parallel mode (workers != 1) requires a path-based input. "
        "Pass a file path, list of paths, or a lazy array loaded from disk, "
        "or set workers=1 to process in-memory arrays sequentially."
    )


def _resolve_worker_count(workers, num_planes):
    """Resolve `workers` kwarg into a concrete worker count.

    workers=1: sequential (caller short-circuits before calling this).
    workers in (None, <=0): auto = min(num_planes, cpu_count//2, 8).
    workers>1: clamp to num_planes.
    """
    if workers is None or (isinstance(workers, int) and workers <= 0):
        cpu = os.cpu_count() or 1
        return max(1, min(num_planes, cpu // 2 or 1, 8))
    return max(1, min(int(workers), num_planes))


class _QueueStreamRedirect:
    """File-like that forwards writes to a logger so print()/tqdm output
    from a worker process flows through the multiprocessing log queue."""

    def __init__(self, logger_, level=logging.INFO):
        self._logger = logger_
        self._level = level
        self._buf = ""

    def write(self, msg):
        if not msg:
            return
        self._buf += msg
        while "\n" in self._buf:
            line, _, self._buf = self._buf.partition("\n")
            if line:
                self._logger.log(self._level, line)

    def flush(self):
        if self._buf:
            self._logger.log(self._level, self._buf)
            self._buf = ""

    def isatty(self):
        return False


def _plane_worker(input_source, current_ops, save_path, run_plane_kwargs,
                  reader_kwargs, log_queue):
    """Process one zplane in a child process. Returns (plane_num, ops_path).

    Module-level so it pickles for ProcessPoolExecutor. The fully-prepared
    `current_ops` is built in the main process; this worker only re-opens
    the input via imread() and calls run_plane().
    """
    plane_num = current_ops.get("plane")

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.INFO)
    # suite2p / cellpose loggers set propagate=False, so route them directly.
    for name in ("suite2p", "cellpose"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.addHandler(logging.handlers.QueueHandler(log_queue))
        lg.setLevel(logging.INFO)
        lg.propagate = False
    global _external_logging_attached
    _external_logging_attached = True

    plane_logger = logging.getLogger(f"plane{plane_num:02d}")
    plane_logger.setLevel(logging.INFO)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _QueueStreamRedirect(plane_logger, logging.INFO)
    sys.stderr = _QueueStreamRedirect(plane_logger, logging.INFO)

    try:
        from mbo_utilities import imread
        arr = imread(input_source, **(reader_kwargs or {}))
        ops_path = run_plane(
            input_data=arr,
            save_path=save_path,
            ops=current_ops,
            **run_plane_kwargs,
        )
        return plane_num, ops_path
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr


def _prepare_plane_ops(*, base_ops, plane_idx, num_planes, input_arr,
                       volume_voxel, volume_source_meta, volume_source_shape,
                       z_selection, frame_indices):
    """Build a per-plane ops dict from base ops.

    Centralizes the per-plane prep so both sequential and parallel paths
    apply identical voxel-size propagation and reactive metadata. Mirrors
    the inline prep that previously lived in run_volume's loop.
    """
    plane_num = plane_idx + 1
    current_ops = copy.deepcopy(load_ops(base_ops)) if base_ops else default_ops()
    reconcile_detection_keys(current_ops)
    current_ops["plane"] = plane_num
    current_ops["num_zplanes"] = num_planes

    if input_arr is not None and hasattr(input_arr, "metadata"):
        _source_idx = (input_arr.metadata or {}).get("selected_planes_0based")
        if _source_idx is not None and plane_idx < len(_source_idx):
            current_ops["source_plane_num"] = int(_source_idx[plane_idx]) + 1

    if volume_voxel is not None:
        if volume_voxel.dz is not None:
            current_ops.setdefault("dz", volume_voxel.dz)
            current_ops.setdefault("z_step", volume_voxel.dz)
        if volume_voxel.dx != 1.0:
            current_ops.setdefault("dx", volume_voxel.dx)
        if volume_voxel.dy != 1.0:
            current_ops.setdefault("dy", volume_voxel.dy)

    _apply_reactive_metadata(
        ops=current_ops,
        source_metadata=volume_source_meta,
        source_shape=volume_source_shape,
        frame_indices=frame_indices,
        plane_indices=z_selection,
        logger=logger,
    )
    return current_ops


def _resolve_timepoints(timepoints=None, frames=None, frame_indices=None):
    """Resolve the canonical 1-based ``timepoints`` selection.

    ``frames`` (1-based) and ``frame_indices`` (0-based) are deprecated
    aliases and emit a DeprecationWarning. Returns a 1-based list, or
    None for all timepoints.
    """
    import warnings

    if frames is not None:
        warnings.warn(
            "'frames' is deprecated, use 'timepoints' (1-based)",
            DeprecationWarning,
            stacklevel=3,
        )
        if timepoints is None:
            timepoints = frames
    if frame_indices is not None:
        warnings.warn(
            "'frame_indices' is deprecated, use 'timepoints' (1-based)",
            DeprecationWarning,
            stacklevel=3,
        )
        if timepoints is None:
            timepoints = [int(i) + 1 for i in frame_indices]
    if timepoints is None:
        return None
    if isinstance(timepoints, (int, np.integer)):
        return [int(timepoints)]
    return [int(t) for t in timepoints]


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
    replot: bool = True,
    timepoints: list | int | None = None,
    num_timepoints: int = None,
    num_zplanes: int = None,
    frame_indices: list | None = None,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    norm_method: str = "zscore",
    correct_neuropil: bool = True,
    cell_filters: list = None,
    accept_all_cells: bool = False,
    rastermap_kwargs: dict = None,
    save_json: bool = False,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
    workers: int | None = 1,
    skip_volumetric: bool = False,
    threads_per_worker: int | None = None,
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
        Flat Suite2p parameter overrides (the fork's ops shape). If None,
        uses suite2p defaults with metadata auto-populated from input.
        Detection is selectable either way and kept consistent: suite2p's
        ``"algorithm"`` ("cellpose" / "sparsery" / "sourcery") or the fork's
        ``"anatomical_only"`` (0=functional, 1=max_proj/mean, 2=mean,
        4=max_proj). Setting one fills in the other; invalid or legacy
        values (e.g. ``anatomical_only=3``) are warned and mapped to what
        suite2p accepts.
    db, settings : dict, optional
        suite2p's nested ``db`` / ``settings`` dicts (the new upstream
        shape). Flattened and merged into ``ops``; explicit ``ops`` keys win.
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
    replot : bool, default True
        Regenerate per-plane figures. Set False to skip per-plane figure
        regeneration (e.g. the volumetric aggregate over already-plotted
        planes); suite2p and the volumetric plots are unaffected.
    timepoints : list[int] or int, optional
        Explicit 1-based timepoints to process. Supports stride
        (e.g. ``list(range(1, 1575, 2))`` for every other timepoint).
        When provided, the implicit stride is used by `OutputMetadata`
        to reactively scale `fs` (e.g. stride of 2 → `fs / 2` in the
        output ops.npy). Takes precedence over ``num_timepoints``.
    num_timepoints : int, optional
        Limit processing to first N timepoints (truncation only). For an
        explicit set or a strided selection, use ``timepoints`` instead.
    num_zplanes : int, optional
        Limit processing to the first N z-planes. Shortcut for
        ``planes=[1..N]``; ignored when ``planes`` is given.
    frame_indices : list[int], optional
        Deprecated alias for ``timepoints`` (0-based). Emits a
        DeprecationWarning.
    dff_window_size : int, optional
        Window size for rolling percentile dF/F baseline (frames).
        If None, auto-calculated as ~10 * tau * fs.
    dff_percentile : int, default 20
        Percentile for baseline F0 estimation.
    dff_smooth_window : int, optional
        Temporal smoothing window for dF/F traces (frames).
        If None, auto-calculated. Set to 1 to disable.
    norm_method : str, default "zscore"
        Normalization for the saved norm_traces.npy. "dff" uses the rolling
        percentile ΔF/F (dff_window_size / dff_percentile / dff_smooth_window
        apply); "zscore" uses per-ROI (F - mean) / std. Quality metrics
        (SNR / skew / shot-noise) always use ΔF/F regardless of this setting.
    correct_neuropil : bool, default True
        If True, the norm trace is computed on F - 0.7 * Fneu. If False, on
        raw F. Affects norm_traces.npy, the trace plots, and trace-quality
        scoring.
    cell_filters : list, optional
        Filters to apply to detected ROIs. Default is no filters (off).
        Currently supports diameter bounds in microns or pixels.
        Example: [{"name": "max_diameter", "min_diameter_um": 4, "max_diameter_um": 35}]
    accept_all_cells : bool, default False
        If True, mark all detected ROIs as accepted, overriding
        Suite2p's built-in classifier results.
    rastermap_kwargs : dict, optional
        Rastermap configuration. Default None (rastermap disabled).
        Pass a dict with optional ``"planar"`` and ``"volumetric"`` keys
        to enable each independently — presence of the key turns it on.
        Each sub-dict holds ``rastermap.Rastermap()`` overrides; an empty
        dict (or None) means use built-in defaults.
        Example::

            rastermap_kwargs={
                "planar": {"n_clusters": 50, "n_PCs": 64},
                "volumetric": {"n_clusters": 40},
            }
    save_json : bool, default False
        Save ops as JSON in addition to .npy format.
    reader_kwargs : dict, optional
        Arguments passed to mbo_utilities.imread().
    writer_kwargs : dict, optional
        Arguments passed to binary writer (e.g., output_format).
    workers : int or None, default 1
        Number of zplane worker processes for volumetric input. ``1``
        keeps the sequential code path. ``None`` (or any value <= 0)
        auto-picks ``min(num_planes, cpu_count // 2, 8)``. Parallel mode
        requires a path-based input; per-plane outputs go to disjoint
        subdirectories. Cellpose on GPU may OOM with multiple workers —
        reduce ``workers`` or switch cellpose to CPU when this happens.
        Ignored for planar (single-plane) inputs.
    skip_volumetric : bool, default False
        When True, return per-plane ops_files immediately after the
        per-plane loop, skipping merge_mrois, volume_stats, and
        volumetric plots. Useful when farming planes across machines
        and aggregating later.
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

    _attach_external_loggers()

    _resolve_gpu_env()
    _apply_thread_limits(threads_per_worker)

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
        if num_timepoints is None:
            num_timepoints = num_frames

    # canonical 1-based timepoint selection (frames/frame_indices deprecated).
    timepoints = _resolve_timepoints(timepoints, kwargs.pop("frames", None), frame_indices)
    frame_indices = None
    # num_zplanes is a count shortcut for planes=[1..N].
    if num_zplanes is not None and planes is None:
        planes = list(range(1, int(num_zplanes) + 1))

    # flatten (db, settings) into ops so downstream run_volume / run_plane
    # don't each need to forward the pair. explicit ops keys still win.
    if db is not None or settings is not None:
        from lbm_suite2p_python.db_settings import merge_db_settings_into_ops
        ops = merge_db_settings_into_ops(ops if isinstance(ops, dict) else None, db, settings)

    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}
    # num_timepoints truncation reaches the writer; an explicit `timepoints`
    # selection is forwarded as a param and rebuilt by run_plane.
    if num_timepoints is not None:
        writer_kwargs["num_timepoints"] = num_timepoints

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
            replot=replot,
            timepoints=timepoints,
            dff_window_size=dff_window_size,
            dff_percentile=dff_percentile,
            dff_smooth_window=dff_smooth_window,
            norm_method=norm_method,
            correct_neuropil=correct_neuropil,
            accept_all_cells=accept_all_cells,
            cell_filters=cell_filters,
            rastermap_kwargs=rastermap_kwargs,
            save_json=save_json,
            reader_kwargs=reader_kwargs,
            writer_kwargs=writer_kwargs,
            workers=workers,
            skip_volumetric=skip_volumetric,
            threads_per_worker=threads_per_worker,
            **kwargs,
        )
    else:
        if workers != 1:
            logger.warning(
                "workers=%r ignored: input is planar (single zplane), no parallelism applicable",
                workers,
            )
        # run_plane is planar-only — extract just the planar sub-dict
        planar_kwargs = _extract_planar_rastermap_kwargs(rastermap_kwargs)
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
            replot=replot,
            timepoints=timepoints,
            dff_window_size=dff_window_size,
            dff_percentile=dff_percentile,
            dff_smooth_window=dff_smooth_window,
            norm_method=norm_method,
            correct_neuropil=correct_neuropil,
            accept_all_cells=accept_all_cells,
            cell_filters=cell_filters,
            rastermap_kwargs=planar_kwargs,
            save_json=save_json,
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
    replot: bool = True,
    timepoints: list | int | None = None,
    num_timepoints: int = None,
    num_zplanes: int = None,
    frame_indices: list | None = None,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    norm_method: str = "zscore",
    correct_neuropil: bool = True,
    accept_all_cells: bool = False,
    cell_filters: list = None,
    rastermap_kwargs: dict = None,
    save_json: bool = False,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
    workers: int | None = 1,
    skip_volumetric: bool = False,
    threads_per_worker: int | None = None,
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
    replot : bool, default True
        Regenerate per-plane figures (passed to run_plane). Set False to skip
        per-plane figure regeneration during a volumetric aggregate.
    frame_indices : list, default None
        List of frame indices to process.
    dff_window_size, dff_percentile, dff_smooth_window : optional
        dF/F calculation parameters (used when norm_method="dff").
    norm_method : str, default "zscore"
        Normalization for norm_traces.npy: "dff" or "zscore". See pipeline().
    accept_all_cells : bool, default False
        Mark all ROIs as accepted.
    cell_filters : list, optional
        Filters to apply (see run_plane).
    rastermap_kwargs : dict, optional
        Unified rastermap config. Default None (off). Keys ``"planar"`` /
        ``"volumetric"`` independently enable each mode; their values hold
        rastermap.Rastermap() overrides (empty dict / None = defaults).
        See pipeline() for the full schema.
    save_json : bool, default False
        Save ops as JSON.
    workers : int or None, default 1
        Number of zplane worker processes. ``1`` (default) keeps the
        sequential path. ``None`` or any value <= 0 auto-picks
        ``min(num_planes, cpu_count // 2, 8)``. Values > 1 run that many
        workers (clamped to num_planes). Parallel mode requires a
        path-based input. Workers reload input via ``imread`` and write
        to disjoint per-plane subdirectories.
        Caveat: cellpose on GPU may OOM with multiple workers; reduce
        ``workers`` or switch cellpose to CPU when this happens.
    skip_volumetric : bool, default False
        When True, return per-plane ops_files immediately after the
        per-plane loop, skipping merge_mrois, volume_stats, and all
        volumetric plots. Useful when farming planes across machines.
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

    _resolve_gpu_env()
    _apply_thread_limits(threads_per_worker)

    # canonical 1-based timepoints (frames/frame_indices deprecated); keep a
    # 0-based frame_indices for this function's reactive-metadata plumbing,
    # and forward `timepoints` to run_plane.
    timepoints = _resolve_timepoints(timepoints, kwargs.pop("frames", None), frame_indices)
    frame_indices = [int(t) - 1 for t in timepoints] if timepoints is not None else None
    if num_zplanes is not None and planes is None:
        planes = list(range(1, int(num_zplanes) + 1))
    writer_kwargs = dict(writer_kwargs or {})
    if num_timepoints is not None:
        writer_kwargs.setdefault("num_timepoints", num_timepoints)

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
        tuple(input_arr._shape5d())
        if input_arr is not None and hasattr(input_arr, "_shape5d")
        else None
    )

    _volume_start = time.time()
    print(
        f"Processing {len(planes_indices)} planes in volume (Total planes: {num_planes})"
    )
    print(f"Output: {save_path}")

    progress_callback = kwargs.pop("progress_callback", None)

    # decompose unified rastermap_kwargs into planar (per-plane) and
    # volumetric (post-loop). Each is None when the user did not enable
    # that mode, an empty dict when they enabled it without overrides.
    planar_rastermap_kwargs = _extract_planar_rastermap_kwargs(rastermap_kwargs)
    volumetric_rastermap_kwargs = _extract_volumetric_rastermap_kwargs(rastermap_kwargs)

    ops_files = []

    # shared run_plane kwargs (identical for sequential and parallel paths)
    _run_plane_kwargs = dict(
        keep_reg=keep_reg,
        keep_raw=keep_raw,
        force_reg=force_reg,
        force_detect=force_detect,
        replot=replot,
        timepoints=timepoints,
        dff_window_size=dff_window_size,
        dff_percentile=dff_percentile,
        dff_smooth_window=dff_smooth_window,
        norm_method=norm_method,
        correct_neuropil=correct_neuropil,
        accept_all_cells=accept_all_cells,
        cell_filters=cell_filters,
        rastermap_kwargs=planar_rastermap_kwargs,
        save_json=save_json,
        reader_kwargs=reader_kwargs,
        writer_kwargs=writer_kwargs,
        **kwargs,
    )

    run_parallel = workers != 1
    if run_parallel:
        n_workers = _resolve_worker_count(workers, len(planes_indices))

    if not run_parallel:
        # Iterate sequentially (original behavior)
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

            current_ops = _prepare_plane_ops(
                base_ops=ops,
                plane_idx=plane_idx,
                num_planes=num_planes,
                input_arr=input_arr,
                volume_voxel=_volume_voxel,
                volume_source_meta=_volume_source_meta,
                volume_source_shape=_volume_source_shape,
                z_selection=_volume_z_selection,
                frame_indices=frame_indices,
            )

            # Call run_plane
            try:
                print(f"\n--- Volume Step: Plane {plane_num} ---")
                _plane_start = time.time()
                ops_file = run_plane(
                    input_data=current_input,
                    save_path=save_path,
                    ops=current_ops,
                    **_run_plane_kwargs,
                )
                ops_files.append(ops_file)
                print(
                    f"--- Plane {plane_num} elapsed: {time.time() - _plane_start:.1f}s ---"
                )

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
    else:
        # Parallel: resolve a picklable input source, pre-build all per-plane
        # ops in the main process, then run planes in a ProcessPoolExecutor.
        try:
            input_source = _resolve_input_source(input_arr, input_data)
        except ValueError:
            raise

        prepared = []  # list of (plane_num, current_ops)
        for plane_idx in planes_indices:
            current_ops = _prepare_plane_ops(
                base_ops=ops,
                plane_idx=plane_idx,
                num_planes=num_planes,
                input_arr=input_arr,
                volume_voxel=_volume_voxel,
                volume_source_meta=_volume_source_meta,
                volume_source_shape=_volume_source_shape,
                z_selection=_volume_z_selection,
                frame_indices=frame_indices,
            )
            prepared.append((plane_idx + 1, current_ops))

        # Eager picklability check on the first plane's ops so users see
        # a clear error at submission rather than a swallowed future.
        try:
            pickle.dumps(prepared[0][1])
            pickle.dumps(_run_plane_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"parallel mode cannot pickle the prepared per-plane ops or "
                f"run_plane kwargs ({exc!r}). Use workers=1 or remove the "
                f"unpicklable item."
            ) from exc

        print(
            f"Running {len(planes_indices)} planes across {n_workers} worker process(es)..."
        )

        # Download the cellpose model once before fanning out; concurrent
        # first-time downloads race on the shared cache file. Use a prepared
        # per-plane ops (already reconciled) so algorithm="cellpose" without
        # anatomical_only still triggers the warm-up.
        _prewarm_cellpose_model(prepared[0][1] if prepared else ops)

        manager = mp.Manager()
        log_queue = manager.Queue()
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        listener = logging.handlers.QueueListener(
            log_queue, stdout_handler, respect_handler_level=False
        )
        listener.start()

        if progress_callback:
            for i, (plane_num, _) in enumerate(prepared):
                progress_callback(
                    plane=i,
                    total_planes=len(planes_indices),
                    step="plane_start",
                    message=f"Plane {plane_num}",
                )

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                future_to_plane = {}
                plane_start_times = {}
                for plane_num, current_ops in prepared:
                    fut = pool.submit(
                        _plane_worker,
                        input_source,
                        current_ops,
                        save_path,
                        _run_plane_kwargs,
                        reader_kwargs,
                        log_queue,
                    )
                    future_to_plane[fut] = plane_num
                    plane_start_times[plane_num] = time.time()

                done_count = 0
                for fut in as_completed(future_to_plane):
                    plane_num = future_to_plane[fut]
                    try:
                        result_plane_num, ops_file = fut.result()
                        ops_files.append((result_plane_num, ops_file))
                        done_count += 1
                        print(
                            f"--- Plane {result_plane_num} elapsed: "
                            f"{time.time() - plane_start_times[plane_num]:.1f}s ---"
                        )
                        if progress_callback:
                            progress_callback(
                                plane=done_count - 1,
                                total_planes=len(planes_indices),
                                step="plane_done",
                                message=f"Plane {result_plane_num} complete",
                            )
                    except Exception as e:
                        print(f"ERROR processing plane {plane_num}: {e}")
                        traceback.print_exc()
        finally:
            listener.stop()

        # Sort results back into z-order — volumetric stats/plots rely on
        # list order matching plane order.
        ops_files = [p for _, p in sorted(ops_files, key=lambda t: t[0])]

    if not ops_files:
        raise RuntimeError(
            "run_volume failed: All planes resulted in exceptions during processing."
        )

    if skip_volumetric:
        logger.info(
            "skip_volumetric=True; returning per-plane ops_files without aggregation"
        )
        return ops_files

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
            # aggregate per-plane ops into volume_stats.npy
            get_volume_stats(ops_files, overwrite=True)

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
                plot_volume_accepted_rejected_overlay(
                    ops_files, save_path / "volume_segmentation_overlay.png"
                )
            except Exception as e:
                print(f"Warning: Volume plots failed: {e}")
                traceback.print_exc()

            if volumetric_rastermap_kwargs is not None:
                try:
                    plot_3d_rastermap_clusters(
                        save_path,
                        save_path=save_path / "rastermap_3d.png",
                        rastermap_kwargs=volumetric_rastermap_kwargs,
                    )
                except Exception as e:
                    print(f"Warning: Volumetric rastermap failed: {e}")
                    traceback.print_exc()

            # volume-level trace figures (trace analysis, top-N raw/dF/F,
            # sorted-activity rastermap.png). runs after plot_3d_rastermap_clusters
            # so the cached rastermap_model.npy can be reused for the heatmap.
            try:
                plot_volume_trace_figures(
                    ops_files,
                    save_path,
                    rastermap_kwargs=volumetric_rastermap_kwargs,
                    correct_neuropil=correct_neuropil,
                    norm_method=norm_method,
                )
            except Exception as e:
                print(f"Warning: Volume trace figures failed: {e}")
                traceback.print_exc()

        except Exception as e:
            print(f"Warning: Volume statistics failed: {e}")
            traceback.print_exc()

    _volume_elapsed = time.time() - _volume_start
    _h, _rem = divmod(int(_volume_elapsed), 3600)
    _m, _s = divmod(_rem, 60)
    print(
        f"\nVolume total elapsed: {_h:02d}:{_m:02d}:{_s:02d} "
        f"({_volume_elapsed:.1f}s) across {len(ops_files)} plane(s)"
    )
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

    Returns True if registration *should* be run, False otherwise.

    Decision rule:
      1. If the plane's data.bin (suite2p's registered output) is missing,
         registration has not run yet on this plane — return True
         unconditionally. ops.npy field state is not trusted here because
         write_ops merges the source array's metadata into a freshly-written
         ops.npy, and source metadata can carry suite2p-output-shaped keys
         (e.g. when imread of a directory pulls in a sibling ops.npy) that
         falsely advertise "registration done" on what is actually raw data.
      2. With data.bin present, fall back to ops.npy field inspection
         (refImg / meanImg / xoff / yoff / regDX / regPC) to decide whether
         the previous registration is complete enough to reuse.
    """
    ops_path = Path(ops_path)
    reg_bin = ops_path.parent / "data.bin"
    if not reg_bin.exists():
        return True

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
    has_metrics = any(
        isinstance(ops.get(k), np.ndarray) and ops[k].size > 0
        for k in ("regDX", "regPC", "regPC1", "regDX1")
    )

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
    # preserve an already-valid absolute path (staging points raw_file/
    # reg_file/chan2_file at the source dir — overwriting those would
    # break the link).
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

    # Graceful fallback for "reload a completed plane" workflow:
    # - keep_raw=False (default) deletes data_raw.bin after a run.
    # - When the user reloads that plane_dir and clicks Run again, the
    #   GUI hands us an ops where raw_file is missing but reg_file (the
    #   registered data.bin) is present. Re-registration needs raw data,
    #   but detection/extraction can still run against data.bin.
    # - Instead of crashing, downgrade to detection-only and log why.
    if run_registration and raw_file is None:
        _existing_reg = ops.get("reg_file")
        if _existing_reg and Path(_existing_reg).exists():
            print(
                "NOTE: raw_file missing but reg_file exists — downgrading "
                "do_registration=1 → 0 (detection/extraction only against "
                f"{Path(_existing_reg).name}). Pass keep_raw=True on the "
                "first run if you want to re-register a reloaded plane."
            )
            ops["do_registration"] = 0
            run_registration = False
        else:
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
        # save user's input diameter for provenance — cellpose later overwrites
        # ops["diameter"] with the estimated median from detection.
        # do not default/coerce here: db_settings._ensure_diameter_list
        # converts to [dy, dx] at the upstream-pipeline boundary, and mainline
        # suite2p / cellpose handle None / scalar / list / aspect-pair natively.
        ops["diameter_user"] = ops["diameter"]

    # reset detection-derived parameters when re-running registration or detection
    # so compute_enhanced_mean_image() reinitializes them from diameter.
    # prevents inheriting spatscale_pix=0 from a failed run, which causes
    # meanImgE to be computed with a [1,1] filter (all 0.5 output).
    # (run_registration / run_detection were resolved earlier, near the
    # raw_file check — keep them in scope here.)
    if run_registration:
        # full reset: clear both registration and detection intermediates.
        # registration outputs (badframes/xoff/yoff/corrXY) are cleared too —
        # otherwise a prior divergent run leaves badframes=True for most
        # frames and detection silently excludes them on the rerun even
        # though the new registration succeeded.
        for key in [
            "spatscale_pix", "Vcorr", "Vmax", "Vmap", "Vsplit", "ihop",
            "badframes", "xoff", "yoff", "corrXY",
            "xoff1", "yoff1", "corrXY1",
        ]:
            if key in ops:
                del ops[key]
    elif run_detection:
        # detection only: clear detection intermediates but keep registration outputs
        for key in ["spatscale_pix", "Vmax", "Vmap", "Vsplit", "ihop"]:
            if key in ops:
                del ops[key]

    reg_file_chan2 = ops_parent / "data_chan2_reg.bin" if use_chan2 else None

    # NOTE: previous versions hard-coded `ops["anatomical_red"] = False`
    # and `ops["chan2_thres"] = 0.1` here. Both were silently overwriting
    # the user's settings and the suite2p schema defaults
    # (detection.chan2_threshold = 0.25). They've been removed — the
    # user's chan2 threshold now flows through unchanged. If a downstream
    # consumer needs anatomical_red / chan2_thres, set them explicitly
    # in the caller's ops dict.

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
                "Either turn registration back on, or point reg_file at an "
                "existing data.bin."
            )

        if "yrange" not in ops or "xrange" not in ops:
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
            ops["meanImgE"] = compute_enhanced_mean_image(
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
        ops["meanImgE"] = compute_enhanced_mean_image(
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
    replot: bool = True,
    timepoints: list | int | None = None,
    num_timepoints: int = None,
    frame_indices: list | None = None,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    norm_method: str = "zscore",
    correct_neuropil: bool = True,
    accept_all_cells: bool = False,
    cell_filters: list = None,
    rastermap_kwargs: dict = None,
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
        Path to or dict of user‐supplied ops (flat fork shape). Detection
        accepts suite2p's ``"algorithm"`` or the fork's ``"anatomical_only"``
        interchangeably (kept consistent; invalid/legacy values warned and
        mapped to suite2p-valid ones). Pass ``db`` / ``settings`` for the
        nested upstream shape.
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
    replot : bool, default True
        Generate per-plane figures. Set False to skip figure generation
        (keeps ROI stats; only the figures are skipped).
    dff_window_size : int, optional
        Frames for rolling percentile baseline. Default: auto-calculated (~10*tau*fs).
    dff_percentile : int, default 20
        Percentile for baseline F0.
    dff_smooth_window : int, optional
        Smoothing window for dF/F. Default: auto-calculated.
    norm_method : str, default "zscore"
        Normalization for norm_traces.npy: "dff" (rolling percentile ΔF/F)
        or "zscore" (per-ROI (F - mean) / std). Quality metrics always use
        ΔF/F regardless of this setting.
    accept_all_cells : bool, default False
        If True, mark all detected ROIs as accepted cells.
    cell_filters : list[dict], optional
        Filters to apply to detected ROIs (e.g. diameter, area).
        Default is no filters.
        Example: [{"name": "max_diameter", "min_diameter_um": 4, "max_diameter_um": 35}]
    rastermap_kwargs : dict, optional
        Per-plane rastermap config. Default None (off). Pass a dict
        (possibly empty) to enable; contents override the cell-count-aware
        defaults passed to rastermap.Rastermap().
        Example: {"n_clusters": 50, "n_PCs": 64}.
    save_json : bool, default False
        Save ops as JSON.
    timepoints : list[int] or int, optional
        Explicit 1-based timepoints. Supports stride
        (e.g. ``list(range(1, 1575, 2))`` for every other timepoint).
        When provided, the binary on disk contains exactly these
        timepoints, and `OutputMetadata` reactively scales `fs` in ops.npy
        based on the implicit stride. Takes precedence over
        ``num_timepoints``.
    num_timepoints : int, optional
        Limit processing to first N timepoints (truncation only).
    frame_indices : list[int], optional
        Deprecated alias for ``timepoints`` (0-based). Emits a
        DeprecationWarning.
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

    _resolve_gpu_env()

    # canonical 1-based timepoints (frames/frame_indices deprecated); convert to
    # the 0-based frame_indices this function consumes internally.
    timepoints = _resolve_timepoints(timepoints, kwargs.pop("frames", None), frame_indices)
    frame_indices = [int(t) - 1 for t in timepoints] if timepoints is not None else None
    writer_kwargs = dict(writer_kwargs or {})
    if num_timepoints is not None:
        writer_kwargs.setdefault("num_timepoints", num_timepoints)

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
    # keep the flat (anatomical_only/sparse_mode) and suite2p (algorithm)
    # detection spellings consistent + valid before they merge into ops.
    reconcile_detection_keys(ops_user)
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
        # binary input with ops but different save_path: stage detection
        # outputs + ops.npy into the new plane_dir, repoint raw_file /
        # reg_file / chan2_file at the source binaries. binaries stay
        # at the source so the new dir is small and copy-pasteable.
        skip_imwrite = True
        file = None

        # resolve the plane-specific source dir (raises on ambiguity / miss).
        target_plane = ops.get("plane", 1)
        source_plane_num = ops.get("source_plane_num")
        src_dir = _resolve_source_plane_dir(
            input_arr, input_path, target_plane, source_plane_num=source_plane_num
        )

        existing_ops = np.load(src_dir / "ops.npy", allow_pickle=True).item()
        # remember the original acquisition source before the merge below
        # overwrites data_path with the staged binary path. needed by
        # force_reg path when the source has no data_raw.bin.
        original_data_path = existing_ops.get("data_path")
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
            subdir_name = generate_plane_dirname(
                plane=naming_plane,
                nframes=nframes_hint,
                frame_start=(int(min(frame_indices)) + 1) if frame_indices else 1,
                frame_stop=(int(max(frame_indices)) + 1) if frame_indices else None,
            )

        plane_dir = save_path / subdir_name
        plane_dir.mkdir(exist_ok=True)
        ops_file = plane_dir / "ops.npy"

        # merge existing ops with user overrides BEFORE staging, so the
        # stage call can mutate the path pointers on the final ops dict
        # without them being wiped by a later merge.
        ops = {
            **ops_default,
            **existing_ops,
            **ops_user,
            "data_path": str(input_path.resolve()),
        }

        if not force_reg:
            # registration writes to data.bin, but data.bin lives in the
            # source dir — running it would overwrite the user's source
            # binary. force detection-only; user can re-register from the
            # original tiff/zarr by passing force_reg=True (Force in the
            # GUI), which routes writes into the new plane_dir.
            if ops_user.get("do_registration", 1):
                logger.warning(
                    "do_registration ignored: the source directory already "
                    "contains a data.bin and running registration here would "
                    "overwrite it. Forcing do_registration=0 (detection-only). "
                    "Select Force in the Registration column (force_reg=True) "
                    "to re-register from the original input into the new "
                    "save_path."
                )
                ops["do_registration"] = 0

        _stage_source_into_plane_dir(src_dir, plane_dir, ops)

        if force_reg:
            # writes go to plane_dir/data.bin — discard the source pointer
            # that _stage_source_into_plane_dir set.
            ops.pop("reg_file", None)
            # existing_ops carries a `raw_file` path from the original
            # run; if keep_raw=False removed the file, the string is
            # still present but stale. existence-check, don't trust the
            # string alone.
            _raw = ops.get("raw_file")
            if not _raw or not Path(_raw).exists():
                import shutil
                target_raw = plane_dir / "data_raw.bin"
                # prefer the original acquisition (tiff/zarr) when it's
                # still on disk — avoids re-registering already-
                # registered data. Skip when it points at the same .bin
                # the user just loaded (no improvement, and imread of a
                # stale data_path could be wrong).
                use_original = (
                    bool(original_data_path)
                    and Path(original_data_path).exists()
                    and Path(original_data_path).resolve() != input_path.resolve()
                )
                if use_original:
                    print(
                        f"  Force registration: rewriting data_raw.bin from "
                        f"{Path(original_data_path).name}"
                    )
                    file = imread(Path(original_data_path), **reader_kwargs)
                    if hasattr(file, "metadata"):
                        metadata = dict(file.metadata)
                    else:
                        metadata = get_metadata(Path(original_data_path))
                    skip_imwrite = False
                else:
                    # original acquisition is gone (or IS the input).
                    # seed data_raw.bin from the staged data.bin so
                    # registration has frames to work with. Re-registers
                    # already-registered data (near-zero shifts) — fine
                    # for detection / bad-frame re-runs.
                    print(
                        f"  Force registration: original acquisition "
                        f"unavailable; seeding data_raw.bin from "
                        f"{input_path.name} (re-registering already-"
                        f"registered data)"
                    )
                    shutil.copy2(input_path, target_raw)
                    ops["raw_file"] = str(target_raw)
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
                writer_kwargs.get("num_timepoints")
                or writer_kwargs.get("num_frames")
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
                frame_start=(int(min(frame_indices)) + 1) if frame_indices else 1,
                frame_stop=(int(max(frame_indices)) + 1) if frame_indices else None,
            )

        plane_dir = save_path / subdir_name
        plane_dir.mkdir(exist_ok=True)
        ops_file = plane_dir / "ops.npy"

        # extract expected dims from input for cache validation
        # honors num_timepoints truncation if user requested fewer frames
        exp_nframes = writer_kwargs.get("num_timepoints") or writer_kwargs.get("num_frames")
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
        # only override the lbm default — don't overwrite a value the
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
            tuple(file._shape5d()) if hasattr(file, "_shape5d") else None
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

        write_kw = dict(writer_kwargs)
        # If the caller gave us explicit timepoints, pass them as
        # `timepoints=` (1-based) to imwrite. This wins over any stale
        # truncation count in writer_kwargs — strided semantics require an
        # explicit index list, not a count.
        if frame_indices is not None:
            write_kw["timepoints"] = [int(i) + 1 for i in frame_indices]
            write_kw.pop("num_timepoints", None)
            write_kw.pop("num_frames", None)

        imwrite(
            file,
            plane_dir,
            ext=".bin",
            metadata=md_combined,
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
    # If we just wrote a fresh data_raw.bin in this run, any stat.npy /
    # ops.npy left in plane_dir from a prior run is stale and must not
    # short-circuit suite2p. fresh raw data needs registration and (unless
    # user-disabled) detection regardless of leftover artifacts.
    fresh_binary_written = (not skip_imwrite) and (file is not None) and should_write

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
        elif fresh_binary_written:
            needs_detect = not user_skip_detect
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
        elif fresh_binary_written:
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
            imwrite(
                chan2_data,
                plane_dir,
                ext=".bin",
                metadata=chan2_md,
                structural=True,
                show_progress=False,
            )
            ops["chan2_file"] = str((plane_dir / "data_chan2.bin").resolve())
            # Use _shape5d() (the always-5D contract) so natural-rank
            # arrays (BinArray, a 4D TZYX _ChannelView) still report the
            # real T dimension at index 0.
            if hasattr(chan2_data, "_shape5d"):
                ops["nframes_chan2"] = int(chan2_data._shape5d()[0])
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

    # Run Suite2p (skip entirely when nothing needs to be done)
    skip_suite2p = not needs_reg and not needs_detect

    # Persistence policy:
    # - When suite2p will run, save ops/db/settings together as a normal
    #   record of the run.
    # - When skipping (cache-hit / user-skip), still persist ops.npy so
    #   the path repoint above (save_path / ops_path / raw_file /
    #   reg_file) reaches disk — otherwise post-processing reloads the
    #   stale paths copied in by _stage_source_into_plane_dir and reads
    #   F.npy/etc. from the wrong location. settings.npy / db.npy are
    #   left untouched so they keep being a faithful record of how the
    #   on-disk artifacts were produced.
    if not skip_suite2p:
        save_ops_db_settings(ops_file, ops)
    else:
        np.save(ops_file, ops, allow_pickle=True)
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
            if "no ROIs were found" in str(e):
                print(f"Warning: no ROIs found for {plane_dir.name}, skipping plane.")
                _cleanup_bin_files(plane_dir, keep_raw, keep_reg)
                return ops_file
            print(f"Error in run_plane_bin: {e}")
            traceback.print_exc()
            _cleanup_bin_files(plane_dir, keep_raw, keep_reg)
            raise

        if not processed:
            _cleanup_bin_files(plane_dir, keep_raw, keep_reg)
            return ops_file

    # post-processing
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

    # 2. Cell Filtering (off by default — empty list)
    if cell_filters:
        print(f"  Applying cell filters: {[f['name'] for f in cell_filters]}")
        filter_start = time.time()
        try:
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
            # flatten filter_results into per-filter metadata for ops.npy
            filter_metadata = {}
            for r in filter_results:
                name = r["name"]
                config = r.get("config", {})
                info = r.get("info", {})
                removed = r.get("removed_mask", np.zeros(0, dtype=bool))
                params = {}
                for key in [
                    "min_diameter_um",
                    "max_diameter_um",
                    "min_diameter_px",
                    "max_diameter_px",
                    "correct_neuropil",
                    "percentile",
                    "window_size",
                    "reject_negative_F0",
                    "min_F0_abs",
                    "min_F0_rel",
                ]:
                    if key in config and config[key] is not None:
                        val = config[key]
                        params[key] = (
                            round(val, 3) if isinstance(val, float) else val
                        )
                if not params:
                    for key in ["min_px", "max_px"]:
                        if key in info and info[key] is not None:
                            params[key] = round(info[key], 1)
                filter_metadata[name] = {
                    "params": params,
                    "n_rejected": int(removed.sum()),
                }
            updated_ops["filter_metadata"] = filter_metadata
            save_ops_db_settings(ops_file, updated_ops)

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

    # 3. Normalized-trace calculation (norm_traces.npy)
    F_file = plane_dir / "F.npy"
    Fneu_file = plane_dir / "Fneu.npy"
    if F_file.exists() and Fneu_file.exists():
        print(f"  Computing norm_traces ({norm_method})...")
        norm_start = time.time()
        F = np.load(F_file)
        Fneu = np.load(Fneu_file)
        if correct_neuropil:
            F_corr = F - 0.7 * Fneu
        else:
            F_corr = F

        current_ops = load_ops(ops_file)
        if norm_method == "zscore":
            norm = zscore_trace(
                F_corr,
                smooth_window=dff_smooth_window,
                fs=current_ops.get("fs", 30.0),
                tau=current_ops.get("tau", 1.0),
            )
        else:
            norm = dff_rolling_percentile(
                F_corr,
                window_size=dff_window_size,
                percentile=dff_percentile,
                smooth_window=dff_smooth_window,
                fs=current_ops.get("fs", 30.0),
                tau=current_ops.get("tau", 1.0),
            )
        np.save(plane_dir / "norm_traces.npy", norm)

        _add_processing_step(
            current_ops,
            "norm_calculation",
            duration_seconds=time.time() - norm_start,
            extra={"method": norm_method, "percentile": dff_percentile},
        )
        save_ops_db_settings(ops_file, current_ops)

    # 3b. Persist post-processing kwargs to ops.npy unconditionally.
    # These live as top-level ops keys (NOT in the suite2p settings
    # schema), so settings.npy / db.npy stay a record of the suite2p
    # stages only — ops.npy carries the lsp / GUI-tunable knobs. Done
    # outside the F.npy/Fneu.npy gate above so detection-skipped runs
    # still leave a record of which knobs the caller passed in. Callers
    # like mbo_utilities / mbo studio diff these against their dataclass
    # defaults to flag "modified" parameters in the GUI.
    if ops_file.exists():
        try:
            _post_ops = load_ops(ops_file)
            _post_ops["dff_window_size"] = dff_window_size
            _post_ops["dff_percentile"] = dff_percentile
            _post_ops["dff_smooth_window"] = dff_smooth_window
            _post_ops["norm_method"] = norm_method
            _post_ops["correct_neuropil"] = bool(correct_neuropil)
            _post_ops["accept_all_cells"] = bool(accept_all_cells)
            _post_ops["save_json"] = bool(save_json)
            # cell_filters: list[dict] (or None). Stored as-is so a reload
            # can reconstruct each criterion's enabled/value pair.
            _post_ops["cell_filters"] = cell_filters
            # rastermap_kwargs: nested dict {"planar": {...}, "volumetric":
            # {...}} or None. Presence of a key is the per-mode enable
            # signal; sub-dict contents override Rastermap() defaults.
            _post_ops["rastermap_kwargs"] = rastermap_kwargs
            save_ops_db_settings(ops_file, _post_ops)
        except Exception as _e:
            print(f"  Warning: persisting post-processing kwargs failed: {_e}")

    # 3b. ROI statistics (cheap; feeds the volumetric aggregate, so always run).
    try:
        from lbm_suite2p_python.postprocessing import compute_roi_stats

        print("  Computing ROI statistics...")
        compute_roi_stats(plane_dir)
    except Exception as e:
        print(f"  Warning: ROI stats computation failed: {e}")

    # 4. Per-plane figures. Timed as the "plots" step (recorded in
    # processing_history). Skipped when replot=False — e.g. the volumetric
    # aggregate re-running already-plotted planes, where regenerating the
    # per-plane figures is the dominant redundant cost.
    if replot:
        plot_start = time.time()
        try:
            plot_zplane_figures(
                plane_dir,
                dff_percentile=dff_percentile,
                dff_window_size=dff_window_size,
                dff_smooth_window=dff_smooth_window,
                norm_method=norm_method,
                correct_neuropil=correct_neuropil,
                run_rastermap=rastermap_kwargs is not None,
                rastermap_kwargs=rastermap_kwargs,
            )
        except Exception as e:
            print(f"  Warning: Plot generation failed: {e}")

        if ops_file.exists():
            try:
                _t_ops = load_ops(ops_file)
                _add_processing_step(_t_ops, "plots", duration_seconds=time.time() - plot_start)
                save_ops_db_settings(ops_file, _t_ops)
            except Exception as _e:
                print(f"  Warning: recording plot timing failed: {_e}")
    else:
        print("  replot=False: keeping existing per-plane figures")

    if save_json:
        ops_to_json(ops_file)

    _cleanup_bin_files(plane_dir, keep_raw, keep_reg)

    if progress_callback:
        progress_callback(step="done", message="Plane complete")

    return ops_file


def grid_search(*args, **kwargs):
    """Run a grid search over Suite2p parameters. See lbm_suite2p_python.grid_search module."""
    from lbm_suite2p_python.grid_search import grid_search as _grid_search

    return _grid_search(*args, **kwargs)
