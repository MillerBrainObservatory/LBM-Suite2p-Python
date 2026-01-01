"""
Format conversion between Suite2p and Cellpose.

Enables bidirectional conversion for:
- Editing Suite2p results in Cellpose GUI
- Using Cellpose detections with Suite2p trace extraction
- Merging results from both pipelines

Key functions:
- detect_format(): auto-detect suite2p vs cellpose
- to_suite2p(): convert any format to suite2p
- to_cellpose(): convert any format to cellpose
- export_for_gui(): export for cellpose GUI editing
- import_from_gui(): import edits back from cellpose GUI
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from mbo_utilities.util import load_npy


# file signatures for format detection
SUITE2P_REQUIRED = ["stat.npy", "iscell.npy", "ops.npy"]
SUITE2P_TRACES = ["F.npy", "Fneu.npy", "spks.npy"]
CELLPOSE_REQUIRED = ["masks.npy"]
CELLPOSE_ALT = ["cellpose_seg.npy", "masks.tif"]


def detect_format(path):
    """
    Detect whether path contains Suite2p or Cellpose results.

    Parameters
    ----------
    path : str or Path
        Directory to check.

    Returns
    -------
    str
        "suite2p", "suite2p_minimal", "cellpose", or "unknown"
    """
    path = Path(path)
    if not path.is_dir():
        path = path.parent

    has_suite2p = all((path / f).exists() for f in SUITE2P_REQUIRED)
    has_traces = all((path / f).exists() for f in SUITE2P_TRACES)
    has_cellpose = any((path / f).exists() for f in CELLPOSE_REQUIRED + CELLPOSE_ALT)

    if has_suite2p and has_traces:
        return "suite2p"
    elif has_cellpose:
        return "cellpose"
    elif has_suite2p:
        return "suite2p_minimal"
    return "unknown"


def validate_format(path, expected=None):
    """
    Validate format and check for required files.

    Parameters
    ----------
    path : str or Path
        Directory to validate.
    expected : str, optional
        Expected format ("suite2p" or "cellpose"). If None, auto-detects.

    Returns
    -------
    dict
        Validation result with keys: valid, format, files, warnings, n_rois, shape
    """
    path = Path(path)
    detected = detect_format(path)

    if expected and detected != expected and detected != f"{expected}_minimal":
        return {
            "valid": False,
            "format": detected,
            "error": f"Expected {expected}, found {detected}",
        }

    result = {
        "valid": True,
        "format": detected,
        "files": {},
        "warnings": [],
        "n_rois": 0,
        "shape": None,
    }

    if detected in ("suite2p", "suite2p_minimal"):
        for f in SUITE2P_REQUIRED + SUITE2P_TRACES:
            result["files"][f] = (path / f).exists()
        if not result["files"].get("F.npy"):
            result["warnings"].append("No traces (F.npy) - extraction needed")

        if (path / "stat.npy").exists():
            stat = np.load(path / "stat.npy", allow_pickle=True)
            result["n_rois"] = len(stat)
        if (path / "ops.npy").exists():
            ops = load_npy(path / "ops.npy").item()
            result["shape"] = (ops.get("Ly"), ops.get("Lx"))

    elif detected == "cellpose":
        for f in CELLPOSE_REQUIRED + CELLPOSE_ALT:
            result["files"][f] = (path / f).exists()

        if (path / "masks.npy").exists():
            masks = np.load(path / "masks.npy")
            result["n_rois"] = int(masks.max())
            result["shape"] = masks.shape

    return result


def _load_ops(path):
    """Load ops.npy from path."""
    path = Path(path)
    if path.is_dir():
        path = path / "ops.npy"
    return load_npy(path).item()


def _get_summary_image(ops, prefer="max_proj"):
    """
    Get a summary image from Suite2p ops, handling cropping correctly.

    Suite2p stores some images (max_proj, Vcorr) cropped to xrange/yrange,
    while others (meanImg, meanImgE, refImg) are full size.

    Parameters
    ----------
    ops : dict
        Suite2p ops dictionary.
    prefer : str, default "max_proj"
        Preferred image type. Options: "max_proj", "meanImg", "meanImgE",
        "refImg", "Vcorr". Falls back to other available images.

    Returns
    -------
    np.ndarray
        Summary image with shape (Ly, Lx), dtype float32.
    """
    Ly, Lx = ops["Ly"], ops["Lx"]

    # Images that are stored at full size (Ly, Lx)
    full_size_keys = ["meanImg", "meanImgE", "refImg"]

    # Images that are cropped to xrange/yrange
    cropped_keys = ["max_proj", "Vcorr"]

    # Get crop ranges if available
    yrange = ops.get("yrange", [0, Ly])
    xrange = ops.get("xrange", [0, Lx])

    def expand_cropped(img):
        """Expand a cropped image back to full (Ly, Lx) size."""
        if img.shape == (Ly, Lx):
            return img
        # Image is cropped - expand it
        full = np.zeros((Ly, Lx), dtype=img.dtype)
        y0, y1 = yrange
        x0, x1 = xrange
        # Verify dimensions match
        crop_h, crop_w = y1 - y0, x1 - x0
        if img.shape == (crop_h, crop_w):
            full[y0:y1, x0:x1] = img
            return full
        # Shape doesn't match expected crop - return as-is or pad
        return img

    # Build priority list based on preference
    if prefer in full_size_keys:
        priority = [prefer] + [k for k in full_size_keys if k != prefer] + cropped_keys
    elif prefer in cropped_keys:
        priority = [prefer] + [k for k in cropped_keys if k != prefer] + full_size_keys
    else:
        priority = full_size_keys + cropped_keys

    # Try each image type in priority order
    for key in priority:
        if key not in ops:
            continue
        img = ops[key]
        if not isinstance(img, np.ndarray):
            continue
        if img.ndim != 2:
            continue

        # Handle cropped vs full-size images
        if key in cropped_keys:
            img = expand_cropped(img)
        elif img.shape != (Ly, Lx):
            continue  # Skip if wrong shape

        if img.shape == (Ly, Lx):
            return img.astype(np.float32)

    # Fallback: return zeros
    return np.zeros((Ly, Lx), dtype=np.float32)


def _compute_outlines(masks):
    """Compute cell outlines from label mask."""
    from scipy.ndimage import binary_dilation

    outlines = np.zeros_like(masks, dtype=bool)
    for cell_id in range(1, masks.max() + 1):
        cell_mask = masks == cell_id
        dilated = binary_dilation(cell_mask)
        outlines |= (dilated ^ cell_mask)
    return outlines


def stat_to_masks(stat, shape):
    """
    Convert Suite2p stat array to Cellpose-style label mask.

    Parameters
    ----------
    stat : np.ndarray
        Suite2p stat array (array of dicts with ypix, xpix).
    shape : tuple
        Output shape (Ly, Lx) or (Z, Ly, Lx).

    Returns
    -------
    np.ndarray
        Label mask where each cell has unique ID (1-indexed).
    """
    masks = np.zeros(shape, dtype=np.uint32)
    for i, s in enumerate(stat):
        ypix = s["ypix"]
        xpix = s["xpix"]
        if "zpix" in s and len(shape) == 3:
            zpix = s["zpix"]
            masks[zpix, ypix, xpix] = i + 1
        else:
            masks[ypix, xpix] = i + 1
    return masks


def masks_to_stat(masks, img=None):
    """
    Convert Cellpose label mask to Suite2p stat array.

    Parameters
    ----------
    masks : np.ndarray
        Label mask (0=background, 1+=cell IDs).
    img : np.ndarray, optional
        Image for computing intensity stats.

    Returns
    -------
    np.ndarray
        Suite2p-compatible stat array.
    """
    from lbm_suite2p_python.cellpose import _masks_to_stat
    return _masks_to_stat(masks, img)


def suite2p_to_cellpose(
    suite2p_dir,
    output_dir=None,
    img_key="max_proj",
):
    """
    Convert Suite2p results to Cellpose format.

    Creates masks.npy, cellpose_seg.npy (GUI compatible), and projection.

    Parameters
    ----------
    suite2p_dir : str or Path
        Suite2p plane directory (containing stat.npy, ops.npy).
    output_dir : str or Path, optional
        Output directory. Defaults to suite2p_dir/cellpose.
    img_key : str
        Key in ops for background image ("max_proj", "meanImg").

    Returns
    -------
    dict
        Conversion metadata with paths and statistics.
    """
    import tifffile

    suite2p_dir = Path(suite2p_dir)
    output_dir = Path(output_dir) if output_dir else suite2p_dir / "cellpose"
    output_dir.mkdir(parents=True, exist_ok=True)

    # load suite2p data
    stat = np.load(suite2p_dir / "stat.npy", allow_pickle=True)
    iscell = np.load(suite2p_dir / "iscell.npy", allow_pickle=True)
    ops = _load_ops(suite2p_dir)

    Ly, Lx = ops["Ly"], ops["Lx"]
    n_rois = len(stat)

    # get background image
    img = ops.get(img_key, ops.get("meanImg", np.zeros((Ly, Lx))))

    # convert stat to masks
    masks = stat_to_masks(stat, (Ly, Lx))
    outlines = _compute_outlines(masks)

    # save masks
    np.save(output_dir / "masks.npy", masks)
    tifffile.imwrite(output_dir / "masks.tif", masks)

    # save projection
    tifffile.imwrite(output_dir / "projection.tif", img.astype(np.float32))

    # create cellpose GUI-compatible seg file
    seg_data = {
        "img": img.astype(np.float32),
        "masks": masks,
        "outlines": outlines,
        "chan_choose": [0, 0],
        "ismanual": np.zeros(n_rois, dtype=bool),
        "filename": str(output_dir / "projection.tif"),
        "flows": None,
        "est_diam": ops.get("diameter"),
    }
    np.save(output_dir / "cellpose_seg.npy", seg_data)

    # preserve iscell
    np.save(output_dir / "iscell.npy", iscell)

    # save conversion metadata
    meta = {
        "source_format": "suite2p",
        "source_path": str(suite2p_dir),
        "converted_at": datetime.now().isoformat(),
        "n_rois": n_rois,
        "shape": [Ly, Lx],
        "img_key": img_key,
    }
    np.save(output_dir / "conversion_meta.npy", meta)

    print(f"Converted {n_rois} ROIs to Cellpose format: {output_dir}")
    return meta


def cellpose_to_suite2p(
    cellpose_dir,
    output_dir=None,
    ops_template=None,
):
    """
    Convert Cellpose results to Suite2p format.

    Creates stat.npy, iscell.npy, ops.npy, and empty trace files.

    Parameters
    ----------
    cellpose_dir : str or Path
        Directory with Cellpose outputs (masks.npy or cellpose_seg.npy).
    output_dir : str or Path, optional
        Output directory. Defaults to cellpose_dir/suite2p/plane0.
    ops_template : dict, optional
        Template ops dict for additional metadata.

    Returns
    -------
    dict
        Conversion metadata with paths and statistics.

    Notes
    -----
    Creates empty F.npy, Fneu.npy, spks.npy. Use extract_traces() with
    a registered binary to populate these.
    """
    cellpose_dir = Path(cellpose_dir)
    output_dir = Path(output_dir) if output_dir else cellpose_dir / "suite2p" / "plane0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # find and load masks
    masks = None
    img = None

    if (cellpose_dir / "cellpose_seg.npy").exists():
        seg = np.load(cellpose_dir / "cellpose_seg.npy", allow_pickle=True).item()
        masks = seg.get("masks")
        img = seg.get("img")
    elif (cellpose_dir / "masks.npy").exists():
        masks = np.load(cellpose_dir / "masks.npy")

    if masks is None:
        raise FileNotFoundError(f"No masks found in {cellpose_dir}")

    # load projection if available
    if img is None:
        for proj_name in ["projection.tif", "projection.npy"]:
            proj_path = cellpose_dir / proj_name
            if proj_path.exists():
                if proj_name.endswith(".tif"):
                    import tifffile
                    img = tifffile.imread(proj_path)
                else:
                    img = np.load(proj_path)
                break

    # convert masks to stat
    stat = masks_to_stat(masks, img)
    n_rois = len(stat)

    # create iscell (all accepted by default)
    iscell = np.ones((n_rois, 2), dtype=np.float32)

    # check for existing iscell
    if (cellpose_dir / "iscell.npy").exists():
        iscell = np.load(cellpose_dir / "iscell.npy", allow_pickle=True)

    # build ops
    Ly, Lx = masks.shape[-2:]
    ops = ops_template.copy() if ops_template else {}
    ops.update({
        "Ly": Ly,
        "Lx": Lx,
        "nframes": 1,
        "save_path": str(output_dir),
    })
    if img is not None:
        ops["meanImg"] = img
        ops["max_proj"] = img

    # load cellpose metadata if available
    meta_path = cellpose_dir / "cellpose_meta.npy"
    if meta_path.exists():
        cp_meta = np.load(meta_path, allow_pickle=True).item()
        ops["diameter"] = cp_meta.get("diameter", ops.get("diameter"))
        ops["cellprob_threshold"] = cp_meta.get("cellprob_threshold")
        ops["flow_threshold"] = cp_meta.get("flow_threshold")

    # save suite2p files
    np.save(output_dir / "stat.npy", stat)
    np.save(output_dir / "iscell.npy", iscell)
    np.save(output_dir / "ops.npy", ops)

    # create empty trace files (placeholders)
    np.save(output_dir / "F.npy", np.zeros((n_rois, 1), dtype=np.float32))
    np.save(output_dir / "Fneu.npy", np.zeros((n_rois, 1), dtype=np.float32))
    np.save(output_dir / "spks.npy", np.zeros((n_rois, 1), dtype=np.float32))

    # save conversion metadata
    meta = {
        "source_format": "cellpose",
        "source_path": str(cellpose_dir),
        "converted_at": datetime.now().isoformat(),
        "n_rois": n_rois,
        "shape": [Ly, Lx],
        "traces_extracted": False,
    }
    np.save(output_dir / "conversion_meta.npy", meta)

    print(f"Converted {n_rois} ROIs to Suite2p format: {output_dir}")
    return meta


def to_suite2p(source, output_dir=None, **kwargs):
    """
    Convert any format to Suite2p.

    Parameters
    ----------
    source : str or Path
        Source directory (auto-detected format).
    output_dir : str or Path, optional
        Output directory.
    **kwargs
        Additional arguments passed to converter.

    Returns
    -------
    dict
        Conversion metadata.
    """
    fmt = detect_format(source)
    if fmt == "cellpose":
        return cellpose_to_suite2p(source, output_dir, **kwargs)
    elif fmt in ("suite2p", "suite2p_minimal"):
        print(f"Already Suite2p format: {source}")
        return {"format": fmt, "path": str(source)}
    else:
        raise ValueError(f"Unknown format at {source}")


def to_cellpose(source, output_dir=None, **kwargs):
    """
    Convert any format to Cellpose.

    Parameters
    ----------
    source : str or Path
        Source directory (auto-detected format).
    output_dir : str or Path, optional
        Output directory.
    **kwargs
        Additional arguments passed to converter.

    Returns
    -------
    dict
        Conversion metadata.
    """
    fmt = detect_format(source)
    if fmt in ("suite2p", "suite2p_minimal"):
        return suite2p_to_cellpose(source, output_dir, **kwargs)
    elif fmt == "cellpose":
        print(f"Already Cellpose format: {source}")
        return {"format": fmt, "path": str(source)}
    else:
        raise ValueError(f"Unknown format at {source}")


def convert(source, target, output_dir=None, **kwargs):
    """
    Convert between formats.

    Parameters
    ----------
    source : str or Path
        Source directory.
    target : str
        Target format: "suite2p" or "cellpose".
    output_dir : str or Path, optional
        Output directory.
    **kwargs
        Additional arguments.

    Returns
    -------
    dict
        Conversion metadata.
    """
    if target == "suite2p":
        return to_suite2p(source, output_dir, **kwargs)
    elif target == "cellpose":
        return to_cellpose(source, output_dir, **kwargs)
    else:
        raise ValueError(f"Unknown target format: {target}")


def export_for_gui(suite2p_dir, output_path=None, name=None):
    """
    Export Suite2p results for Cellpose GUI editing.

    Creates a _seg.npy file that can be opened directly in Cellpose GUI.
    The cellpose GUI expects files with names ending in '_seg.npy'.

    Parameters
    ----------
    suite2p_dir : str or Path
        Suite2p plane directory containing stat.npy and ops.npy.
    output_path : str or Path, optional
        Output directory. Defaults to suite2p_dir.
    name : str, optional
        Base name for output files. Defaults to 'projection'.
        Creates {name}.tif and {name}_seg.npy.

    Returns
    -------
    Path
        Path to the created _seg.npy file.

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>> seg_file = lsp.conversion.export_for_gui("path/to/suite2p/plane0")
    >>> lsp.cellpose.open_in_gui(seg_file)  # opens in cellpose GUI
    """
    import tifffile

    suite2p_dir = Path(suite2p_dir)
    name = name or "projection"

    # determine output directory
    if output_path is not None:
        output_dir = Path(output_path)
        if output_dir.suffix == ".npy":
            # user provided full path to file
            output_dir = output_dir.parent
            name = output_dir.stem.replace("_seg", "")
    else:
        output_dir = suite2p_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # load suite2p data
    stat = np.load(suite2p_dir / "stat.npy", allow_pickle=True)
    ops = _load_ops(suite2p_dir)

    Ly, Lx = ops["Ly"], ops["Lx"]
    img = _get_summary_image(ops)

    # convert to masks
    masks = stat_to_masks(stat, (Ly, Lx))
    outlines = _compute_outlines(masks)

    n_rois = len(stat)

    # save projection image (cellpose GUI needs this)
    proj_path = output_dir / f"{name}.tif"
    tifffile.imwrite(proj_path, img.astype(np.float32), compression="zlib")

    # create cellpose GUI-compatible _seg.npy
    seg_data = {
        "img": img.astype(np.float32),
        "masks": masks.astype(np.uint32),
        "outlines": outlines,
        "chan_choose": [0, 0],
        "ismanual": np.zeros(n_rois, dtype=bool),
        "filename": str(proj_path),
        "flows": None,
        "est_diam": ops.get("diameter"),
        "cellprob_threshold": ops.get("cellprob_threshold", 0.0),
        "flow_threshold": ops.get("flow_threshold", 0.4),
    }

    seg_file = output_dir / f"{name}_seg.npy"
    np.save(seg_file, seg_data, allow_pickle=True)

    # also save masks.tif for easy viewing
    tifffile.imwrite(
        output_dir / f"{name}_masks.tif",
        masks.astype(np.uint16),
        compression="zlib",
    )

    print(f"Exported {n_rois} ROIs for Cellpose GUI:")
    print(f"  Image: {proj_path}")
    print(f"  Seg file: {seg_file}")
    print(f"\nTo open in Cellpose GUI:")
    print(f"  lsp.cellpose.open_in_gui('{seg_file}')")
    return seg_file


def import_from_gui(seg_file, original_dir, output_dir=None, update_in_place=False):
    """
    Import edited results from Cellpose GUI back to Suite2p format.

    Parameters
    ----------
    seg_file : str or Path
        Path to edited _seg.npy from Cellpose GUI.
    original_dir : str or Path
        Original Suite2p directory (for ops, traces).
    output_dir : str or Path, optional
        Output directory. If None and update_in_place=False, creates 'edited' subdir.
    update_in_place : bool
        If True, overwrites original stat.npy and iscell.npy.

    Returns
    -------
    dict
        Import result with change summary.
    """
    seg_file = Path(seg_file)
    original_dir = Path(original_dir)

    if update_in_place:
        output_dir = original_dir
    else:
        output_dir = Path(output_dir) if output_dir else original_dir / "edited"
        output_dir.mkdir(parents=True, exist_ok=True)

    # load edited data
    seg = np.load(seg_file, allow_pickle=True).item()
    masks = seg["masks"]
    ismanual = seg.get("ismanual", np.zeros(int(masks.max()), dtype=bool))

    # load original for comparison
    stat_orig = np.load(original_dir / "stat.npy", allow_pickle=True)
    n_orig = len(stat_orig)

    # convert edited masks to stat
    img = seg.get("img")
    stat_new = masks_to_stat(masks, img)
    n_new = len(stat_new)

    # create iscell (preserve manual additions)
    iscell = np.ones((n_new, 2), dtype=np.float32)

    # track changes
    n_added = max(0, n_new - n_orig)
    n_removed = max(0, n_orig - n_new)
    n_manual = ismanual.sum() if len(ismanual) == n_new else 0

    # save updated files
    np.save(output_dir / "stat.npy", stat_new)
    np.save(output_dir / "iscell.npy", iscell)

    # copy ops if not in place
    if not update_in_place:
        ops = _load_ops(original_dir)
        ops["save_path"] = str(output_dir)
        np.save(output_dir / "ops.npy", ops)

        # note: traces not copied - they're now invalid due to ROI changes

    result = {
        "n_original": n_orig,
        "n_edited": n_new,
        "n_added": n_added,
        "n_removed": n_removed,
        "n_manual": int(n_manual),
        "output_dir": str(output_dir),
    }

    print(f"Imported GUI edits: {n_orig} → {n_new} ROIs")
    if n_added > 0:
        print(f"  Added: {n_added}")
    if n_removed > 0:
        print(f"  Removed: {n_removed}")
    if n_manual > 0:
        print(f"  Manual: {n_manual}")

    if not update_in_place:
        print(f"  Note: Traces need re-extraction for modified ROIs")

    return result


def compare_detections(path_a, path_b, iou_threshold=0.5):
    """
    Compare ROI detections between two results.

    Parameters
    ----------
    path_a, path_b : str or Path
        Directories to compare (auto-detected format).
    iou_threshold : float
        IoU threshold for considering ROIs matched.

    Returns
    -------
    dict
        Comparison results with matched pairs and unique ROIs.
    """
    from scipy.ndimage import label

    path_a, path_b = Path(path_a), Path(path_b)

    # load masks from both
    def load_masks(path):
        fmt = detect_format(path)
        if fmt == "cellpose":
            return np.load(path / "masks.npy")
        else:
            stat = np.load(path / "stat.npy", allow_pickle=True)
            ops = _load_ops(path)
            return stat_to_masks(stat, (ops["Ly"], ops["Lx"]))

    masks_a = load_masks(path_a)
    masks_b = load_masks(path_b)

    n_a = int(masks_a.max())
    n_b = int(masks_b.max())

    # compute IoU for all pairs
    matched = []
    matched_a = set()
    matched_b = set()

    for i in range(1, n_a + 1):
        mask_i = masks_a == i
        area_i = mask_i.sum()
        if area_i == 0:
            continue

        for j in range(1, n_b + 1):
            mask_j = masks_b == j
            area_j = mask_j.sum()
            if area_j == 0:
                continue

            intersection = (mask_i & mask_j).sum()
            union = area_i + area_j - intersection
            iou = intersection / union if union > 0 else 0

            if iou >= iou_threshold:
                matched.append((i - 1, j - 1, float(iou)))
                matched_a.add(i - 1)
                matched_b.add(j - 1)

    unique_a = [i for i in range(n_a) if i not in matched_a]
    unique_b = [i for i in range(n_b) if i not in matched_b]

    result = {
        "n_a": n_a,
        "n_b": n_b,
        "n_matched": len(matched),
        "matched_pairs": matched,
        "unique_to_a": unique_a,
        "unique_to_b": unique_b,
        "mean_iou": np.mean([m[2] for m in matched]) if matched else 0,
    }

    print(f"Comparison: {n_a} vs {n_b} ROIs")
    print(f"  Matched: {len(matched)} (mean IoU: {result['mean_iou']:.3f})")
    print(f"  Unique to A: {len(unique_a)}")
    print(f"  Unique to B: {len(unique_b)}")

    return result


def get_results(path, include_traces=True):
    """
    Load segmentation results from Suite2p or Cellpose format.

    Returns a normalized dictionary with consistent structure regardless
    of input format. Useful for downstream analysis and GUI integration.

    Parameters
    ----------
    path : str or Path
        Path to results directory (containing stat.npy/ops.npy for Suite2p,
        or masks.npy/_seg.npy for Cellpose).
    include_traces : bool, default True
        Whether to load fluorescence traces (F, Fneu, spks) if available.

    Returns
    -------
    dict
        Normalized results dictionary with keys:
        - format: str - "suite2p", "cellpose", or "unknown"
        - path: Path - source directory
        - stat: ndarray - ROI statistics (Suite2p format)
        - masks: ndarray - label image (Cellpose format)
        - iscell: ndarray - cell classification (n_rois, 2)
        - n_rois: int - number of ROIs
        - shape: tuple - (Ly, Lx) image dimensions
        - image: ndarray - mean/max projection image
        - F: ndarray - fluorescence traces (if available)
        - Fneu: ndarray - neuropil traces (if available)
        - spks: ndarray - deconvolved spikes (if available)
        - dff: ndarray - dF/F traces (if available)
        - ops: dict - Suite2p ops dictionary (if available)
        - seg_file: Path - path to _seg.npy file (for cellpose GUI)

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>> results = lsp.get_results("path/to/suite2p/plane0")
    >>> print(f"Found {results['n_rois']} ROIs")
    >>> masks = results['masks']  # always available
    >>> stat = results['stat']    # always available
    """
    path = Path(path)
    if not path.is_dir():
        path = path.parent

    fmt = detect_format(path)

    result = {
        "format": fmt,
        "path": path,
        "stat": None,
        "masks": None,
        "iscell": None,
        "n_rois": 0,
        "shape": None,
        "image": None,
        "F": None,
        "Fneu": None,
        "spks": None,
        "dff": None,
        "ops": None,
        "seg_file": None,
    }

    if fmt in ("suite2p", "suite2p_minimal"):
        # Load Suite2p format
        stat_file = path / "stat.npy"
        ops_file = path / "ops.npy"
        iscell_file = path / "iscell.npy"

        if stat_file.exists():
            result["stat"] = np.load(stat_file, allow_pickle=True)
            result["n_rois"] = len(result["stat"])

        if ops_file.exists():
            result["ops"] = load_npy(ops_file).item()
            Ly, Lx = result["ops"].get("Ly"), result["ops"].get("Lx")
            result["shape"] = (Ly, Lx)
            # Get summary image (handles cropped images properly)
            result["image"] = _get_summary_image(result["ops"])

        if iscell_file.exists():
            result["iscell"] = np.load(iscell_file, allow_pickle=True)

        # Generate masks from stat
        if result["stat"] is not None and result["shape"] is not None:
            result["masks"] = stat_to_masks(result["stat"], result["shape"])

        # Load traces if requested
        if include_traces and fmt == "suite2p":
            for name in ["F", "Fneu", "spks", "dff"]:
                trace_file = path / f"{name}.npy"
                if trace_file.exists():
                    result[name] = np.load(trace_file)

        # Check for existing _seg.npy file
        seg_file = path / "projection_seg.npy"
        if seg_file.exists():
            result["seg_file"] = seg_file

    elif fmt == "cellpose":
        # Load Cellpose format
        masks_file = path / "masks.npy"
        seg_files = list(path.glob("*_seg.npy"))

        if seg_files:
            result["seg_file"] = seg_files[0]
            seg_data = np.load(seg_files[0], allow_pickle=True).item()
            result["masks"] = seg_data.get("masks")
            result["image"] = seg_data.get("img")
            if result["masks"] is not None:
                result["n_rois"] = int(result["masks"].max())
                result["shape"] = result["masks"].shape[-2:]
        elif masks_file.exists():
            result["masks"] = np.load(masks_file)
            result["n_rois"] = int(result["masks"].max())
            result["shape"] = result["masks"].shape[-2:]

        # Convert masks to stat
        if result["masks"] is not None:
            result["stat"] = masks_to_stat(result["masks"], result["image"])

        # Load iscell if available
        iscell_file = path / "iscell.npy"
        if iscell_file.exists():
            result["iscell"] = np.load(iscell_file)
        elif result["n_rois"] > 0:
            # Default: all cells accepted
            result["iscell"] = np.ones((result["n_rois"], 2), dtype=np.float32)

        # Load projection image if not from seg file
        if result["image"] is None:
            for img_name in ["projection.tif", "projection.npy"]:
                img_file = path / img_name
                if img_file.exists():
                    if img_name.endswith(".tif"):
                        import tifffile
                        result["image"] = tifffile.imread(img_file)
                    else:
                        result["image"] = np.load(img_file)
                    break

    return result


def ensure_cellpose_format(path, force=False):
    """
    Ensure a _seg.npy file exists for cellpose GUI compatibility.

    If the path contains Suite2p results without a _seg.npy file,
    creates one using export_for_gui().

    Parameters
    ----------
    path : str or Path
        Path to results directory.
    force : bool, default False
        If True, recreate _seg.npy even if it exists.

    Returns
    -------
    Path
        Path to the _seg.npy file.
    """
    path = Path(path)
    if not path.is_dir():
        path = path.parent

    seg_file = path / "projection_seg.npy"

    if seg_file.exists() and not force:
        return seg_file

    fmt = detect_format(path)

    if fmt in ("suite2p", "suite2p_minimal"):
        return export_for_gui(path)
    elif fmt == "cellpose":
        # Already cellpose format, check for _seg.npy
        seg_files = list(path.glob("*_seg.npy"))
        if seg_files:
            return seg_files[0]
        # No _seg.npy, create from masks
        masks_file = path / "masks.npy"
        if masks_file.exists():
            masks = np.load(masks_file)
            # Find image
            image = None
            for img_name in ["projection.tif", "projection.npy"]:
                img_file = path / img_name
                if img_file.exists():
                    if img_name.endswith(".tif"):
                        import tifffile
                        image = tifffile.imread(img_file)
                    else:
                        image = np.load(img_file)
                    break
            if image is None:
                image = np.zeros(masks.shape[-2:], dtype=np.float32)

            # Use save_gui_results from cellpose module
            from lbm_suite2p_python.cellpose import save_gui_results
            return save_gui_results(path, masks, image, name="projection")

    raise ValueError(f"Cannot create _seg.npy for format: {fmt}")
