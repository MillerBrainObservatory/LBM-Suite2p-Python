"""Default ops for the LBM pipeline.

Historically this module hard-coded a flat ops dict whose values
diverged from suite2p's own defaults in a handful of fields
(`batch_size=500`, `chan2_thres=0.65`, `tau=1.3`, `diameter=4`, etc.).
That divergence made the round-trip through `settings.npy` confusing —
on-disk values that matched the lsp default were flagged as "modified"
against suite2p's schema, and vice versa.

This module now exposes exactly suite2p's defaults (the values from
`suite2p.default_settings()` + `suite2p.default_db()`), flattened to the
fork's flat-ops shape via the same `db_settings_to_ops` translation
that the rest of lsp uses. There is now ONE source of truth for "what
default means": suite2p's parameter schema. Any LBM-specific tweaks
(diameter, tau, etc.) belong in user code, not in the default.
"""

from mbo_utilities.metadata import get_param, get_voxel_size


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
