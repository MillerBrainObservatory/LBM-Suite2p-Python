"""
Padding-aware extraction shim for upstream suite2p.

Problem
-------
When LBM data is axially registered, mbo's writer zero-pads planes so every
plane sits in a common (Ly, Lx) coordinate system. `ops["yrange"]` /
`ops["xrange"]` mark the valid (non-padded) region.

Upstream's `suite2p.detection.*` is yrange/xrange-aware: `bin_movie` crops
before computing meanImg / max_proj / Vcorr / cellpose inputs, so padding
never enters detection. Upstream's `suite2p.extraction.masks` is NOT:
`create_cell_pix` allocates `np.zeros((Ly, Lx))`, and `create_neuropil_masks`
treats any pixel with `cell_pix < 0.5` as a valid neuropil candidate. Zero-
padded pixels satisfy that predicate, so neuropil donuts around
boundary-adjacent ROIs grow into the padding and pull zeros into `Fneu`.
The corrupted `Fneu` propagates into `F - neuropil_coefficient * Fneu` and
every dF/F derived from it.

Fix
---
Monkeypatch `suite2p.extraction.masks.create_cell_pix` inside a context
manager scoped to the lbm call only. The replacement:

  1. Calls upstream's real implementation to produce the cell-pixel mask.
  2. Marks every pixel outside `ops["yrange"]` / `ops["xrange"]` as
     "occupied" (True / 1.0), which makes `create_neuropil_masks`'s
     `cell_pix[y,x] < 0.5` predicate return False for those pixels, so
     donuts grow only into the valid region.

The patch is applied *only* for the duration of the `with` block and
restored on exit, regardless of whether the wrapped code succeeds or
raises. External users of suite2p see no change; only our call path is
affected.

Why this rather than rebuilding neuropil_masks externally and passing
them to `extraction_wrapper(neuropil_masks=...)`
-------------------------------------------------------------------
The external-build approach would require us to bypass upstream's
`pipeline()` and re-orchestrate detection → classification → extraction →
deconvolution ourselves, just to inject the `neuropil_masks=` kwarg.
Monkeypatching a single leaf function is ~15 lines, keeps us on the
upstream orchestration path, and is trivially reversible.
"""

from __future__ import annotations

import contextlib
import numpy as np


@contextlib.contextmanager
def padding_aware_extraction(ops):
    """
    Context manager: inside the block, upstream's neuropil-mask builder
    treats pixels outside ``ops["yrange"]`` / ``ops["xrange"]`` as
    "occupied", preventing neuropil donuts from extending into the
    zero-padded axial-alignment border.

    Safe to call unconditionally: if no yrange/xrange are present on ops,
    or if they cover the full frame, the patch is a no-op.

    Parameters
    ----------
    ops : dict
        Flat suite2p ops dict. Reads ``ops["yrange"]`` and
        ``ops["xrange"]``; does not mutate.

    Yields
    ------
    None

    Examples
    --------
    >>> with padding_aware_extraction(ops):
    ...     reg_outputs, detect_outputs, stat, F, Fneu, ... = pipeline(
    ...         save_path=save_path, f_reg=f_reg, ..., settings=settings,
    ...     )
    """
    yrange = ops.get("yrange")
    xrange = ops.get("xrange")
    if not yrange or not xrange:
        # no crop information — nothing to protect against
        yield
        return

    try:
        y0, y1 = int(yrange[0]), int(yrange[1])
        x0, x1 = int(xrange[0]), int(xrange[1])
    except (TypeError, ValueError):
        yield
        return

    # import here so a missing suite2p doesn't break import of this module
    try:
        from suite2p.extraction import masks as _m
    except ImportError:
        yield
        return

    original_create_cell_pix = getattr(_m, "create_cell_pix", None)
    if original_create_cell_pix is None:
        # upstream moved or inlined create_cell_pix — no safe patch point.
        # fall through without modification rather than silently corrupting.
        yield
        return

    def _padding_aware_create_cell_pix(stats, Ly, Lx, lam_percentile=50.0):
        """Drop-in replacement that adds the padding mask to upstream's result."""
        cell_pix = original_create_cell_pix(stats, Ly, Lx, lam_percentile)
        # upstream returns either bool array or float >0.5 mask; normalize
        # to a bool copy so our additions don't alter any internal cache.
        cell_pix = np.asarray(cell_pix).copy()
        if cell_pix.dtype != bool:
            cell_pix = cell_pix > 0.5

        # mark the region OUTSIDE the valid window as occupied so the
        # neuropil donut never extends into zero-padded pixels.
        _y0 = max(0, min(y0, Ly))
        _y1 = max(_y0, min(y1, Ly))
        _x0 = max(0, min(x0, Lx))
        _x1 = max(_x0, min(x1, Lx))
        if _y0 > 0:
            cell_pix[:_y0, :] = True
        if _y1 < Ly:
            cell_pix[_y1:, :] = True
        if _x0 > 0:
            cell_pix[:, :_x0] = True
        if _x1 < Lx:
            cell_pix[:, _x1:] = True
        return cell_pix

    _m.create_cell_pix = _padding_aware_create_cell_pix
    try:
        yield
    finally:
        # always restore, even on exception, so external callers of
        # suite2p.extraction.masks.create_cell_pix see the unpatched
        # upstream function.
        _m.create_cell_pix = original_create_cell_pix
