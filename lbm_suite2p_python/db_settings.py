"""
Bidirectional conversion between the flat `ops` dict used throughout
lbm_suite2p_python and the nested `db` / `settings` dicts introduced in
upstream MouseLand/suite2p (see suite2p/parameters.py).

Used by run_plane / pipeline to:
  - accept either `ops=...` OR `db=..., settings=...` as input
  - write `db.npy` + `settings.npy` alongside every `ops.npy`

The mapping below is derived directly from upstream's
`default_settings()` and `default_db()` shapes and must stay in sync
with them. Keys unique to the fork (dff_*, keep_raw, keep_reg, mbo
metadata, registration outputs like meanImg / yoff, ...) are preserved
in ops.npy only; they do not round-trip through db/settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# flat keys that live at the top level of settings in upstream
_SETTINGS_TOP_LEVEL = ("torch_device", "tau", "fs", "diameter")

# upstream nested section membership. Each entry lists the keys
# upstream puts under settings[section]. Only listed keys are migrated.
_SETTINGS_SECTIONS: dict[str, tuple[str, ...]] = {
    "run": (
        "do_registration",
        "do_regmetrics",
        "do_detection",
        "do_deconvolution",
        "multiplane_parallel",
    ),
    "io": (
        "combined",
        "save_mat",
        "save_NWB",
        "save_ops_orig",
        "delete_bin",
        "move_bin",
    ),
    "registration": (
        "align_by_chan2",
        "nimg_init",
        "maxregshift",
        "do_bidiphase",
        "bidiphase",
        "batch_size",
        "nonrigid",
        "maxregshiftNR",
        "block_size",
        "smooth_sigma_time",
        "smooth_sigma",
        "spatial_taper",
        "th_badframes",
        "norm_frames",
        "snr_thresh",
        "subpixel",
        "two_step_registration",
        "reg_tif",
        "reg_tif_chan2",
        "upsample_meanImg",
    ),
    "classification": (
        "classifier_path",
        "use_builtin_classifier",
        "preclassify",
        "skip_classification",
        "accept_all_cells",
    ),
    "extraction": (
        "snr_threshold",
        "batch_size",
        "neuropil_extract",
        "neuropil_coefficient",
        "inner_neuropil_radius",
        "min_neuropil_pixels",
        "lam_percentile",
        "allow_overlap",
        "circular_neuropil",
    ),
    "dcnv_preprocess": (
        "baseline",
        "win_baseline",
        "sig_baseline",
        "prctile_baseline",
    ),
}

# detection is nested two levels deep. These are the keys at
# settings["detection"][...] (top of detection).
_DETECTION_TOP_KEYS = (
    "algorithm",
    "denoise",
    "block_size",
    "nbins",
    "bin_size",
    "highpass_time",
    "threshold_scaling",
    "npix_norm_min",
    "npix_norm_max",
    "max_overlap",
    "soma_crop",
    "chan2_threshold",
    "cellpose_chan2",
)

_DETECTION_SPARSERY_KEYS = (
    "highpass_neuropil",
    "max_ROIs",
    "spatial_scale",
    "active_percentile",
)

_DETECTION_SOURCERY_KEYS = (
    "connected",
    "max_iterations",
    "smooth_masks",
)

_DETECTION_CELLPOSE_KEYS = (
    "cellpose_model",
    "img",
    "highpass_spatial",
    "flow_threshold",
    "cellprob_threshold",
    "params",
    "params_chan2",
)

# db is flat. These are the upstream db keys.
_DB_KEYS = (
    "data_path",
    "look_one_level_down",
    "input_format",
    "keep_movie_raw",
    "nplanes",
    "nrois",
    "nchannels",
    "swap_order",
    "functional_chan",
    "lines",
    "ignore_flyback",
    "subfolders",
    "file_list",
    "save_path0",
    "fast_disk",
    "save_folder",
    "h5py_key",
    "nwb_driver",
    "nwb_series",
    "force_sktiff",
    "bruker_bidirectional",
)
# NOTE: upstream db also has "dy", "dx", "batch_size". We deliberately
# do NOT migrate those — fork's ops["dy"/"dx"] are voxel-size metadata
# (um/pixel), which collides with upstream's db.dy/dx (mesoscope line
# info). fork's ops["batch_size"] historically means the registration
# batch, which we already route to settings["registration"]["batch_size"].

# fork → upstream renames (applies both directions)
_FORK_TO_UPSTREAM_RENAMES: dict[str, str] = {
    "roidetect": "do_detection",
    "spikedetect": "do_deconvolution",
    "save_nwb": "save_NWB",
    "neucoeff": "neuropil_coefficient",
    "chan2_thres": "chan2_threshold",
    "nbinned": "nbins",
    "high_pass": "highpass_time",
    "spatial_hp_cp": "highpass_spatial",
    "pretrained_model": "cellpose_model",
}

# fork's anatomical_only int (1-4) selects which image cellpose runs on.
# upstream replaced this with a string in cellpose_settings.img. mapping:
#   1 -> 'max_proj / meanImg'  (log ratio)
#   2 -> 'meanImg'
#   3 -> enhanced_mean_img  (REMOVED upstream — no equivalent string;
#                            falls through to max_proj branch)
#   4 -> 'max_proj'  (anything not matching the two strings hits else: img=max_proj)
_ANATOMICAL_ONLY_TO_IMG: dict[int, str] = {
    1: "max_proj / meanImg",
    2: "meanImg",
    4: "max_proj",
}
_UPSTREAM_TO_FORK_RENAMES = {v: k for k, v in _FORK_TO_UPSTREAM_RENAMES.items()}

# per-section flat-key disambiguation. Several upstream sections share
# an upstream key. When flattened to ops, the second iteration clobbers
# the first; on the reverse derivation, the surviving value gets fanned
# back into BOTH sections — silently corrupting one of them.
# This map gives the colliding (section, upstream_key) pair its own
# distinct flat-ops key so the round-trip preserves both values.
#
# Discovered collisions (audit via _SETTINGS_SECTIONS + detection groups):
#   "batch_size" — registration vs extraction.
#                  registration keeps the legacy flat name "batch_size"
#                  (matches the historical fork ops shape);
#                  extraction gets "extract_batch_size".
#   "block_size" — registration vs detection (suite2p uses the same
#                  upstream key for the rigid-block geometry AND the
#                  pca-denoise / sparsery block size).
#                  registration keeps the legacy flat name "block_size";
#                  detection gets "det_block_size".
#
# Both rename targets match the field names that mbo's GUI dataclass
# already uses (`extract_batch_size`, `det_block_size_y/_x`).
_SECTION_FLAT_RENAMES: dict[tuple[str, str], str] = {
    ("extraction", "batch_size"): "extract_batch_size",
    ("detection", "block_size"): "det_block_size",
}

# baseline values differ between fork and upstream:
#   fork dcnv.preprocess branches on "maximin" / "constant" / "constant_prctile"
#   upstream dcnv.preprocess branches on "maximin" / "constant" / "prctile"
# "constant_percentile" is an mbo-GUI-only typo that silently fell through in
# both — normalize it here.
_BASELINE_FORK_TO_UPSTREAM = {
    "constant_prctile": "prctile",
    "constant_percentile": "prctile",
}
_BASELINE_UPSTREAM_TO_FORK = {
    "prctile": "constant_prctile",
}


def _ensure_diameter_array(value: Any) -> np.ndarray:
    """Coerce diameter to an `np.ndarray` of shape (2,), mirroring suite2p
    `pipeline_s2p.py:150-160`: scalar -> [d, d]; list/tuple -> array; size-1
    array -> [d, d]; size>=2 ndarray passes through. None falls back to
    [6., 6.] (fork-only — upstream doesn't accept None at this point).
    """
    if value is None:
        return np.array([6.0, 6.0])
    if not isinstance(value, (list, tuple, np.ndarray)):
        return np.array([value, value])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return np.array([6.0, 6.0])
        value = np.array(value)
    if value.size == 1:
        v = value.item()
        return np.array([v, v])
    return value


def _ensure_do_registration_int(value: Any) -> int:
    """Upstream do_registration is 0/1/2; fork stores bool."""
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _derive_detection_algorithm(ops: dict) -> str | None:
    """
    Fork encodes detection algorithm via `sparse_mode` (bool) and
    `anatomical_only` (int 0-4). Upstream uses a single string.
    Returns None if no signal is present (caller leaves key untouched).
    """
    if "algorithm" in ops and ops["algorithm"]:
        return str(ops["algorithm"])
    anatomical = ops.get("anatomical_only", 0)
    if anatomical and int(anatomical) > 0:
        return "cellpose"
    if "sparse_mode" in ops:
        return "sparsery" if ops["sparse_mode"] else "sourcery"
    return None


def ops_to_db_settings(ops: dict) -> tuple[dict, dict]:
    """
    Split a flat lbm/fork ops dict into (db, settings) matching upstream.

    Keys absent from the mapping are dropped — the caller keeps them in
    ops.npy. The returned dicts are safe to `np.save` and round-trip
    through `db_settings_to_ops` for the mapped subset.
    """
    db: dict[str, Any] = {}
    settings: dict[str, Any] = {}

    # resolve fork→upstream renames into a temporary lookup that
    # doesn't mutate the caller's ops
    lookup = dict(ops)
    for fork_name, upstream_name in _FORK_TO_UPSTREAM_RENAMES.items():
        if fork_name in lookup and upstream_name not in lookup:
            lookup[upstream_name] = lookup[fork_name]

    # db (flat)
    for key in _DB_KEYS:
        if key in lookup:
            db[key] = lookup[key]

    # settings top-level
    for key in _SETTINGS_TOP_LEVEL:
        if key in lookup:
            if key == "diameter":
                settings[key] = _ensure_diameter_array(lookup[key])
            else:
                settings[key] = lookup[key]

    # flat sections (run, io, registration, classification, extraction, dcnv_preprocess)
    for section, keys in _SETTINGS_SECTIONS.items():
        bucket: dict[str, Any] = {}
        for key in keys:
            # per-section flat-key disambiguation. e.g. extraction reads
            # its batch_size from ops["extract_batch_size"] (not
            # ops["batch_size"]) so registration's batch_size doesn't get
            # spread into both sections during the flat→structured
            # derivation. Mirror of the write side in db_settings_to_ops.
            flat_key = _SECTION_FLAT_RENAMES.get((section, key), key)
            if flat_key not in lookup:
                continue
            value = lookup[flat_key]
            if section == "run" and key == "do_registration":
                value = _ensure_do_registration_int(value)
            if section == "registration" and key == "block_size" and isinstance(value, list):
                value = tuple(value)
            if section == "registration" and key == "align_by_chan2":
                # fork's align_by_chan (int 1/2) → upstream align_by_chan2 bool
                if "align_by_chan2" not in ops and "align_by_chan" in ops:
                    value = bool(ops.get("align_by_chan", 1) == 2)
            if section == "dcnv_preprocess" and key == "baseline":
                value = _BASELINE_FORK_TO_UPSTREAM.get(str(value), value)
            bucket[key] = value
        if bucket:
            settings[section] = bucket

    # handle align_by_chan specifically: upstream only has align_by_chan2
    reg = settings.setdefault("registration", {})
    if "align_by_chan2" not in reg and "align_by_chan" in lookup:
        reg["align_by_chan2"] = bool(lookup["align_by_chan"] == 2)
        if not reg:
            settings.pop("registration", None)

    # detection (two-level nested)
    detection: dict[str, Any] = {}
    algorithm = _derive_detection_algorithm(lookup)
    if algorithm is not None:
        detection["algorithm"] = algorithm
    for key in _DETECTION_TOP_KEYS:
        if key == "algorithm":
            continue
        # per-section flat-key disambiguation. detection.block_size reads
        # from ops["det_block_size"] (not ops["block_size"], which holds
        # the registration block size). Mirror of the write side.
        flat_key = _SECTION_FLAT_RENAMES.get(("detection", key), key)
        if flat_key not in lookup:
            continue
        value = lookup[flat_key]
        if key == "block_size" and isinstance(value, list):
            value = tuple(value)
        detection[key] = value

    sparsery: dict[str, Any] = {}
    for key in _DETECTION_SPARSERY_KEYS:
        if key in lookup:
            sparsery[key] = lookup[key]
    if sparsery:
        detection["sparsery_settings"] = sparsery

    sourcery: dict[str, Any] = {}
    for key in _DETECTION_SOURCERY_KEYS:
        if key in lookup:
            sourcery[key] = lookup[key]
    if sourcery:
        detection["sourcery_settings"] = sourcery

    cellpose: dict[str, Any] = {}
    for key in _DETECTION_CELLPOSE_KEYS:
        if key in lookup:
            cellpose[key] = lookup[key]

    # fork's anatomical_only int picks which image cellpose runs on.
    # upstream encodes this via cellpose_settings.img; without this
    # mapping anatomical_only=4 silently degrades to anatomical_only=1
    # (the upstream default 'max_proj / meanImg' = log ratio image)
    # because the translator only sets algorithm='cellpose' otherwise.
    anat = lookup.get("anatomical_only")
    if anat and "img" not in cellpose:
        try:
            anat_int = int(anat)
        except (TypeError, ValueError):
            anat_int = None
        if anat_int in _ANATOMICAL_ONLY_TO_IMG:
            cellpose["img"] = _ANATOMICAL_ONLY_TO_IMG[anat_int]
        elif anat_int == 3:
            # upstream removed enhanced_mean_img — closest fallback is
            # 'meanImg' (same source family, no median-ratio enhancement).
            # warn the caller so they know something changed.
            import warnings
            warnings.warn(
                "anatomical_only=3 (enhanced_mean_img) is no longer supported "
                "upstream; falling back to cellpose_settings.img='meanImg'. "
                "Use anatomical_only in {1, 2, 4} to avoid this fallback.",
                stacklevel=2,
            )
            cellpose["img"] = "max_proj"

    if cellpose:
        detection["cellpose_settings"] = cellpose

    if detection:
        settings["detection"] = detection

    return db, settings


def db_settings_to_ops(db: dict | None, settings: dict | None) -> dict:
    """
    Flatten (db, settings) back to an ops dict consumable by the fork's
    run_plane / pipeline. Upstream→fork renames are applied so downstream
    code that reads e.g. ops["roidetect"] still works.
    """
    ops: dict[str, Any] = {}

    if db:
        for key, value in db.items():
            ops[key] = value

    if not settings:
        return ops

    # top-level settings
    for key in _SETTINGS_TOP_LEVEL:
        if key in settings:
            value = settings[key]
            if key == "diameter":
                # keep upstream's [dy, dx] np.ndarray shape — fork consumers
                # that need a scalar should use np.mean / [0] explicitly.
                value = _ensure_diameter_array(value)
            ops[key] = value

    # flat sections
    for section, keys in _SETTINGS_SECTIONS.items():
        section_dict = settings.get(section, {}) or {}
        for key in keys:
            if key not in section_dict:
                continue
            value = section_dict[key]
            if section == "dcnv_preprocess" and key == "baseline":
                value = _BASELINE_UPSTREAM_TO_FORK.get(str(value), value)
            # per-section disambiguation first (e.g. extraction.batch_size
            # → extract_batch_size to avoid colliding with
            # registration.batch_size). Fall back to the global
            # upstream→fork rename map.
            target_name = _SECTION_FLAT_RENAMES.get(
                (section, key),
                _UPSTREAM_TO_FORK_RENAMES.get(key, key),
            )
            ops[target_name] = value

    # mirror align_by_chan2 back to align_by_chan for fork consumers
    reg = settings.get("registration", {}) or {}
    if "align_by_chan2" in reg and "align_by_chan" not in ops:
        ops["align_by_chan"] = 2 if reg["align_by_chan2"] else 1

    # detection
    detection = settings.get("detection", {}) or {}
    for key in _DETECTION_TOP_KEYS:
        if key in detection:
            value = detection[key]
            # per-section disambiguation first (e.g. detection.block_size
            # → det_block_size to avoid colliding with
            # registration.block_size).
            target_name = _SECTION_FLAT_RENAMES.get(
                ("detection", key),
                _UPSTREAM_TO_FORK_RENAMES.get(key, key),
            )
            ops[target_name] = value

    # reverse the algorithm → sparse_mode derivation for fork consumers
    if "algorithm" in detection and "sparse_mode" not in ops:
        ops["sparse_mode"] = detection["algorithm"] == "sparsery"

    for nested_name, keys in (
        ("sparsery_settings", _DETECTION_SPARSERY_KEYS),
        ("sourcery_settings", _DETECTION_SOURCERY_KEYS),
        ("cellpose_settings", _DETECTION_CELLPOSE_KEYS),
    ):
        nested = detection.get(nested_name, {}) or {}
        for key in keys:
            if key in nested:
                value = nested[key]
                target_name = _UPSTREAM_TO_FORK_RENAMES.get(key, key)
                ops[target_name] = value

    return ops


def save_ops_db_settings(ops_file: str | Path, ops: dict) -> None:
    """
    Save `ops` to `ops_file` and also write upstream-shaped `db.npy` and
    `settings.npy` siblings in the same directory. Safe to call in place
    of `np.save(ops_file, ops)`.
    """
    ops_path = Path(ops_file)
    np.save(ops_path, ops, allow_pickle=True)

    plane_dir = ops_path.parent
    try:
        db, settings = ops_to_db_settings(ops)
    except Exception:
        # never let the upstream-shape write break the pipeline — ops.npy
        # is the source of truth and was already saved above.
        return

    try:
        np.save(plane_dir / "db.npy", db, allow_pickle=True)
        np.save(plane_dir / "settings.npy", settings, allow_pickle=True)
    except Exception:
        pass


def merge_db_settings_into_ops(ops: dict | None, db: dict | None, settings: dict | None) -> dict:
    """
    For run_plane / pipeline callers that pass db= and settings= kwargs:
    produce a single flat ops dict by flattening (db, settings) first,
    then overlaying any explicitly-provided ops keys on top (explicit
    `ops=` wins the merge).
    """
    out = db_settings_to_ops(db, settings)
    if ops:
        out.update(ops)
    return out
