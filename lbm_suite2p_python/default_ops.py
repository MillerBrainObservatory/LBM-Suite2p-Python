"""Default ops for the LBM pipeline.

Historically this module hard-coded a flat ops dict whose values
diverged from suite2p's own defaults in a handful of fields
(`batch_size=500`, `chan2_thres=0.65`, `tau=1.3`, `diameter=4`, etc.).
That divergence made the round-trip through `settings.npy` confusing —
on-disk values that matched the lsp default were flagged as "modified"
against suite2p's schema, and vice versa.

`s2p_ops()` exposes exactly suite2p's defaults (the values from
`suite2p.default_settings()` + `suite2p.default_db()`), flattened to the
fork's flat-ops shape via the same `db_settings_to_ops` translation
that the rest of lsp uses. `default_ops()` then applies a small set of
LBM detection defaults (`_LBM_DETECTION_DEFAULTS`) on top, so a notebook,
a bare `pipeline()` / `run_plane()` call, and the GUI all start from the
same place. An explicit caller-supplied `ops=` is respected, not overlaid.
"""

from mbo_utilities.metadata import get_param, get_voxel_size


# LBM detection defaults that intentionally differ from suite2p's schema.
# default_ops() applies these on top of the suite2p mirror so notebook,
# function-call, and GUI runs share one set of defaults.
_LBM_DETECTION_DEFAULTS = {
    "do_regmetrics": False,      # PC reg-quality metrics: ~40s/plane on >=1500 frames
    "cellprob_threshold": -4.0,  # more permissive than suite2p's 0.0 for LBM data
    "flow_threshold": 0.0,       # flow check disabled
}


def s2p_ops() -> dict:
    """Suite2p's default ops in flat (fork-style) form.

    Composed from `suite2p.default_settings()` + `suite2p.default_db()`
    via `db_settings_to_ops`, so the rename map and per-section
    flat-key disambiguation (e.g. extraction.batch_size →
    extract_batch_size) are applied consistently.
    """
    from suite2p import default_settings, default_db
    from lbm_suite2p_python.db_settings import db_settings_to_ops

    return db_settings_to_ops(default_db(), default_settings())


def default_ops(metadata: dict | None = None, ops: dict | None = None) -> dict:
    """Return default ops for the LBM Suite2p pipeline.

    Parameters
    ----------
    metadata : dict, optional
        Source-data metadata. When provided, `fs` and the (`dx`, `dy`)
        voxel-size pair are pulled in from the metadata and overlaid
        onto the suite2p defaults.
    ops : dict, optional
        A user-supplied ops dict to start from. When None, starts from
        `s2p_ops()` (suite2p's defaults).

    Returns
    -------
    dict
        Flat ops dict ready to be passed to `pipeline()` / `run_plane()`.

    Notes
    -----
    `nplanes=1` and `nchannels=1` are forced at the end. The lsp
    pipeline always processes one plane at a time, so these stay fixed
    regardless of caller input.

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>> ops = lsp.default_ops()
    >>> lsp.run_plane(
    ...     ops=ops,
    ...     input_tiff="D:/demo/raw_data/raw_file_00001.tif",
    ...     save_path="D:/demo/results",
    ...     save_folder="v1",
    ... )
    """
    if ops is None:
        ops = s2p_ops()
        ops.update(_LBM_DETECTION_DEFAULTS)

    if metadata is not None:
        fs = get_param(metadata, "fs")
        if fs is not None:
            ops["fs"] = fs
        voxel = get_voxel_size(metadata)
        if voxel.dx != 1.0 or voxel.dy != 1.0:
            ops["dx"] = voxel.dx
            ops["dy"] = voxel.dy

    ops["nplanes"] = 1
    ops["nchannels"] = 1
    return ops
