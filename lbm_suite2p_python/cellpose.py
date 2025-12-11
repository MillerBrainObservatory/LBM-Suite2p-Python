"""
Cellpose segmentation module for LBM data.

This module provides direct Cellpose integration without going through Suite2p,
giving full control over Cellpose parameters and outputs. Results are saved in
formats compatible with both the Cellpose GUI and downstream analysis.

References:
- Cellpose API: https://cellpose.readthedocs.io/en/latest/api.html
- Cellpose Inputs: https://cellpose.readthedocs.io/en/latest/inputs.html
- Cellpose Outputs: https://cellpose.readthedocs.io/en/latest/outputs.html
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from mbo_utilities import imread
from mbo_utilities.arrays import _normalize_planes


# lazy array types from mbo_utilities
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


def _get_num_planes(arr):
    """Get number of z-planes from array."""
    if hasattr(arr, "num_planes"):
        return arr.num_planes
    if hasattr(arr, "num_channels"):
        return arr.num_channels
    shape = arr.shape
    if len(shape) == 4:
        return shape[1]  # TZYX format
    return 1


def _compute_projection(
    arr,
    plane_idx: int = None,
    method: str = "max",
    percentile: float = 99,
) -> np.ndarray:
    """
    Compute temporal projection for Cellpose input.

    Parameters
    ----------
    arr : array-like
        Input array (T, Y, X) for 3D or (T, Z, Y, X) for 4D.
    plane_idx : int, optional
        For 4D arrays, which z-plane to extract (0-indexed).
        If None, uses all planes for 3D segmentation.
    method : str
        Projection method: 'max', 'mean', 'std', or 'percentile'.
    percentile : float
        Percentile value if method='percentile'.

    Returns
    -------
    np.ndarray
        2D or 3D projection suitable for Cellpose.
    """
    ndim = len(arr.shape)

    if ndim == 4:
        # (T, Z, Y, X)
        if plane_idx is not None:
            # extract single plane -> (T, Y, X)
            data = arr[:, plane_idx, :, :]
        else:
            # keep all planes -> (T, Z, Y, X)
            data = arr[:]
    elif ndim == 3:
        # (T, Y, X)
        data = arr[:]
    else:
        raise ValueError(f"Expected 3D or 4D array, got {ndim}D")

    # convert to numpy if lazy
    if hasattr(data, "compute"):
        data = data.compute()
    data = np.asarray(data)

    # compute temporal projection
    if method == "max":
        proj = np.max(data, axis=0)
    elif method == "mean":
        proj = np.mean(data, axis=0)
    elif method == "std":
        proj = np.std(data, axis=0)
    elif method == "percentile":
        proj = np.percentile(data, percentile, axis=0)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    return proj.astype(np.float32)


def _normalize_image(img, percentile_low=1, percentile_high=99):
    """Normalize image to 0-1 range using percentiles."""
    low = np.percentile(img, percentile_low)
    high = np.percentile(img, percentile_high)
    if high - low < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - low) / (high - low), 0, 1).astype(np.float32)


def _masks_to_stat(masks, img=None):
    """
    Convert Cellpose masks to Suite2p-style stat array.

    Parameters
    ----------
    masks : np.ndarray
        2D or 3D label image from Cellpose.
    img : np.ndarray, optional
        Original image for computing additional statistics.

    Returns
    -------
    np.ndarray
        Array of stat dictionaries compatible with Suite2p.
    """
    from scipy import ndimage

    stat = []
    n_rois = masks.max()

    for roi_id in range(1, n_rois + 1):
        roi_mask = masks == roi_id
        if not roi_mask.any():
            continue

        # get pixel coordinates
        if masks.ndim == 2:
            ypix, xpix = np.where(roi_mask)
            zpix = None
        else:
            zpix, ypix, xpix = np.where(roi_mask)

        # compute centroid
        med_y = np.median(ypix)
        med_x = np.median(xpix)

        # compute bounding box for aspect ratio
        y_range = ypix.max() - ypix.min() + 1
        x_range = xpix.max() - xpix.min() + 1
        aspect = max(y_range, x_range) / max(1, min(y_range, x_range))

        # approximate radius from area
        npix = len(xpix)
        radius = np.sqrt(npix / np.pi)

        roi_stat = {
            "ypix": ypix.astype(np.int32),
            "xpix": xpix.astype(np.int32),
            "npix": npix,
            "med": [med_y, med_x],
            "radius": radius,
            "aspect_ratio": aspect,
            "compact": npix / (np.pi * radius**2) if radius > 0 else 0,
        }

        if zpix is not None:
            roi_stat["zpix"] = zpix.astype(np.int32)
            roi_stat["med_z"] = np.median(zpix)

        # add intensity stats if image provided
        if img is not None:
            if img.ndim == 2:
                roi_vals = img[ypix, xpix]
            else:
                roi_vals = img[zpix, ypix, xpix] if zpix is not None else img[ypix, xpix]
            roi_stat["mean_intensity"] = float(np.mean(roi_vals))
            roi_stat["max_intensity"] = float(np.max(roi_vals))

        stat.append(roi_stat)

    return np.array(stat, dtype=object)


def _save_cellpose_output(
    save_dir: Path,
    masks: np.ndarray,
    flows: tuple,
    styles: np.ndarray,
    img: np.ndarray,
    plane_idx: int = None,
    metadata: dict = None,
):
    """
    Save Cellpose outputs in multiple formats.

    Creates:
    - masks.npy / masks.tif: label image
    - flows.npy: flow fields
    - stat.npy: Suite2p-compatible ROI statistics
    - cellpose_seg.npy: full Cellpose output (GUI compatible)
    - projection.tif: the image used for segmentation
    """
    import tifffile

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plane_suffix = f"_plane{plane_idx:02d}" if plane_idx is not None else ""

    # save masks as both npy and tiff
    np.save(save_dir / f"masks{plane_suffix}.npy", masks)
    tifffile.imwrite(
        save_dir / f"masks{plane_suffix}.tif",
        masks.astype(np.uint16),
        compression="zlib",
    )

    # save flows
    if flows is not None:
        np.save(save_dir / f"flows{plane_suffix}.npy", np.array(flows, dtype=object))

    # save styles
    if styles is not None:
        np.save(save_dir / f"styles{plane_suffix}.npy", styles)

    # save projection image
    tifffile.imwrite(
        save_dir / f"projection{plane_suffix}.tif",
        img.astype(np.float32),
        compression="zlib",
    )

    # save Suite2p-compatible stat
    stat = _masks_to_stat(masks, img)
    np.save(save_dir / f"stat{plane_suffix}.npy", stat)

    # create iscell array (all accepted by default)
    n_rois = len(stat)
    iscell = np.ones((n_rois, 2), dtype=np.float32)  # column 0: is_cell, column 1: probability
    np.save(save_dir / f"iscell{plane_suffix}.npy", iscell)

    # save Cellpose GUI-compatible _seg.npy
    seg_data = {
        "masks": masks,
        "outlines": None,  # can be computed from masks if needed
        "chan_choose": [0, 0],
        "ismanual": np.zeros(n_rois, dtype=bool),
        "filename": str(save_dir / f"projection{plane_suffix}.tif"),
        "flows": flows,
        "est_diam": None,
    }
    np.save(save_dir / f"cellpose_seg{plane_suffix}.npy", seg_data)

    # save metadata
    meta = metadata or {}
    meta.update({
        "n_rois": n_rois,
        "masks_shape": list(masks.shape),
        "plane_idx": plane_idx,
        "timestamp": datetime.now().isoformat(),
    })
    np.save(save_dir / f"cellpose_meta{plane_suffix}.npy", meta)

    print(f"  Saved {n_rois} ROIs to {save_dir}")
    return stat, iscell


def cellpose(
    input_data,
    save_path: str | Path = None,
    planes: list | int = None,
    projection: Literal["max", "mean", "std", "percentile"] = "max",
    projection_percentile: float = 99,
    # cellpose model parameters
    model_type: str = "cyto3",
    gpu: bool = True,
    # cellpose eval parameters
    diameter: float = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    batch_size: int = 8,
    normalize: bool = True,
    # 3D options
    do_3D: bool = False,
    anisotropy: float = None,
    stitch_threshold: float = 0.0,
    # i/o options
    reader_kwargs: dict = None,
    overwrite: bool = False,
) -> dict:
    """
    Run Cellpose segmentation directly on imaging data.

    This function bypasses Suite2p and runs Cellpose directly, providing full
    control over segmentation parameters. Accepts any input format supported
    by mbo_utilities.imread().

    Parameters
    ----------
    input_data : str, Path, or array
        Input data source. Can be:
        - Path to a file (TIFF, Zarr, HDF5)
        - Path to a directory containing files
        - Pre-loaded lazy array from mbo_utilities
    save_path : str or Path, optional
        Output directory for results. If None, creates 'cellpose/' subdirectory
        next to the input.
    planes : int or list, optional
        Which z-planes to process (1-indexed). Options:
        - None: Process all planes (default)
        - int: Process single plane (e.g., planes=7)
        - list: Process specific planes (e.g., planes=[1, 5, 10])
    projection : str, default 'max'
        Temporal projection method: 'max', 'mean', 'std', or 'percentile'.
    projection_percentile : float, default 99
        Percentile value if projection='percentile'.

    model_type : str, default 'cyto3'
        Cellpose model to use. Options:
        - 'cyto3': Latest cytoplasm model (recommended)
        - 'cyto2': Previous cytoplasm model
        - 'cyto': Original cytoplasm model
        - 'nuclei': Nuclear segmentation
        - Path to custom model file
    gpu : bool, default True
        Use GPU if available.
    diameter : float, optional
        Expected cell diameter in pixels. If None, Cellpose auto-estimates.
    flow_threshold : float, default 0.4
        Maximum allowed error of flows for each mask.
    cellprob_threshold : float, default 0.0
        Probability threshold for cell detection. Lower = more cells.
    min_size : int, default 15
        Minimum number of pixels per mask.
    batch_size : int, default 8
        Batch size for GPU processing.
    normalize : bool, default True
        Whether to normalize images before segmentation.
    do_3D : bool, default False
        Run 3D segmentation (for volumetric data).
    anisotropy : float, optional
        Ratio of z-resolution to xy-resolution for 3D segmentation.
    stitch_threshold : float, default 0.0
        IoU threshold for stitching masks across z-planes.
    reader_kwargs : dict, optional
        Keyword arguments passed to mbo_utilities.imread().
    overwrite : bool, default False
        Overwrite existing results.

    Returns
    -------
    dict
        Dictionary containing:
        - 'save_path': Path to output directory
        - 'planes': List of processed plane indices
        - 'n_rois': Total number of ROIs detected
        - 'stat': Combined stat array for all planes
        - 'timing': Processing timing information

    Examples
    --------
    >>> import lbm_suite2p_python as lsp

    >>> # Basic usage with auto diameter estimation
    >>> result = lsp.cellpose("path/to/data.zarr", save_path="output/")

    >>> # Specific planes with custom parameters
    >>> result = lsp.cellpose(
    ...     "path/to/data",
    ...     planes=[1, 5, 10],
    ...     diameter=8,
    ...     cellprob_threshold=-2,
    ...     flow_threshold=0.6,
    ... )

    >>> # 3D volumetric segmentation
    >>> result = lsp.cellpose(
    ...     "path/to/volume.zarr",
    ...     do_3D=True,
    ...     anisotropy=2.0,  # z is 2x coarser than xy
    ... )

    >>> # Use mean projection instead of max
    >>> result = lsp.cellpose(
    ...     "path/to/data",
    ...     projection="mean",
    ... )

    Notes
    -----
    Output structure::

        save_path/
        ├── masks_plane00.tif       # Label image for plane 0
        ├── masks_plane00.npy       # Same as numpy array
        ├── stat_plane00.npy        # Suite2p-compatible ROI stats
        ├── iscell_plane00.npy      # Cell classification (all accepted)
        ├── projection_plane00.tif  # Image used for segmentation
        ├── cellpose_seg_plane00.npy  # Cellpose GUI-compatible output
        ├── flows_plane00.npy       # Flow fields
        └── cellpose_meta.npy       # Processing metadata

    The outputs are compatible with:
    - Cellpose GUI (load cellpose_seg*.npy)
    - Suite2p analysis (stat.npy, iscell.npy)
    - Standard image viewers (masks*.tif)

    See Also
    --------
    pipeline : Full Suite2p pipeline with Cellpose integration
    filter_by_max_diameter : Post-segmentation filtering
    """
    from cellpose import models, core

    start_time = time.time()
    timing = {}

    # normalize reader_kwargs
    reader_kwargs = reader_kwargs or {}

    print("Cellpose Segmentation")
    print("=" * 60)

    # load input data
    print("Loading input data...")
    t0 = time.time()

    if _is_lazy_array(input_data):
        arr = input_data
        filenames = getattr(arr, "filenames", [])
        print(f"  Input: {type(arr).__name__} (pre-loaded array)")
        if save_path is None:
            if filenames:
                save_path = Path(filenames[0]).parent / "cellpose"
            else:
                raise ValueError("save_path required for array input without filenames")
    elif isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
        print(f"  Input: {input_path}")
        arr = imread(input_path, **reader_kwargs)
        print(f"  Loaded as: {type(arr).__name__}")
        if save_path is None:
            save_path = (input_path.parent if input_path.is_file() else input_path) / "cellpose"
    else:
        raise TypeError(f"input_data must be path or lazy array, got {type(input_data)}")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    timing["load_data"] = time.time() - t0

    # get array info
    shape = arr.shape
    ndim = len(shape)
    num_planes = _get_num_planes(arr)
    num_frames = shape[0]

    print(f"\nDataset info:")
    print(f"  Shape: {shape}")
    print(f"  Frames: {num_frames}")
    print(f"  Planes: {num_planes}")
    print(f"  Data type: {'4D volumetric' if ndim == 4 else '3D planar'}")

    # normalize planes to 0-indexed list
    if ndim == 4:
        planes_to_process = _normalize_planes(planes, num_planes)
    else:
        planes_to_process = [None]  # single plane data

    print(f"\nProcessing plan:")
    if planes_to_process[0] is not None:
        print(f"  Planes: {[p+1 for p in planes_to_process]}")
    else:
        print(f"  Single plane data")
    print(f"  Projection: {projection}")
    print(f"  Output: {save_path}")

    # check GPU
    use_gpu = gpu and core.use_gpu()
    print(f"\nGPU: {'enabled' if use_gpu else 'disabled'}")

    # load model
    print(f"\nLoading Cellpose model ({model_type})...")
    t0 = time.time()
    model = models.CellposeModel(model_type=model_type, gpu=use_gpu)
    timing["model_load"] = time.time() - t0
    print(f"  Model loaded in {timing['model_load']:.2f}s")

    # process each plane
    all_stat = []
    all_iscell = []
    plane_results = []

    for plane_idx in planes_to_process:
        plane_start = time.time()

        if plane_idx is not None:
            print(f"\n{'='*60}")
            print(f"Processing plane {plane_idx + 1}/{num_planes}")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Processing single plane")
            print(f"{'='*60}")

        # check if already processed
        suffix = f"_plane{plane_idx:02d}" if plane_idx is not None else ""
        masks_file = save_path / f"masks{suffix}.npy"

        if masks_file.exists() and not overwrite:
            print(f"  Loading existing results...")
            masks = np.load(masks_file)
            stat = np.load(save_path / f"stat{suffix}.npy", allow_pickle=True)
            iscell = np.load(save_path / f"iscell{suffix}.npy")
            n_rois = len(stat)
            print(f"  Found {n_rois} existing ROIs")
        else:
            # compute projection
            print(f"  Computing {projection} projection...")
            t0 = time.time()
            proj = _compute_projection(
                arr,
                plane_idx=plane_idx,
                method=projection,
                percentile=projection_percentile,
            )
            timing[f"projection_plane{plane_idx}"] = time.time() - t0
            print(f"  Projection shape: {proj.shape}, took {timing[f'projection_plane{plane_idx}']:.2f}s")

            # run cellpose
            print(f"  Running Cellpose...")
            t0 = time.time()

            # build eval kwargs
            eval_kwargs = {
                "batch_size": batch_size,
                "flow_threshold": flow_threshold,
                "cellprob_threshold": cellprob_threshold,
                "min_size": min_size,
                "normalize": normalize,
            }

            if diameter is not None:
                eval_kwargs["diameter"] = diameter

            if do_3D or (proj.ndim == 3 and plane_idx is None):
                eval_kwargs["do_3D"] = True
                eval_kwargs["z_axis"] = 0
                if anisotropy is not None:
                    eval_kwargs["anisotropy"] = anisotropy
                if stitch_threshold > 0:
                    eval_kwargs["stitch_threshold"] = stitch_threshold

            masks, flows, styles = model.eval(proj, **eval_kwargs)
            timing[f"cellpose_plane{plane_idx}"] = time.time() - t0

            n_rois = int(masks.max())
            print(f"  Found {n_rois} ROIs in {timing[f'cellpose_plane{plane_idx}']:.2f}s")

            # save outputs
            stat, iscell = _save_cellpose_output(
                save_path,
                masks=masks,
                flows=flows,
                styles=styles,
                img=proj,
                plane_idx=plane_idx,
                metadata={
                    "model_type": model_type,
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "projection": projection,
                    "do_3D": do_3D,
                },
            )

        all_stat.extend(stat)
        all_iscell.append(iscell)
        plane_results.append({
            "plane_idx": plane_idx,
            "n_rois": len(stat),
            "time": time.time() - plane_start,
        })

    # combine results
    total_rois = len(all_stat)
    combined_stat = np.array(all_stat, dtype=object)
    combined_iscell = np.vstack(all_iscell) if all_iscell else np.zeros((0, 2))

    # save combined results
    np.save(save_path / "stat.npy", combined_stat)
    np.save(save_path / "iscell.npy", combined_iscell)

    # save timing
    timing["total"] = time.time() - start_time
    np.save(save_path / "timing.npy", timing)

    # summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Total ROIs: {total_rois}")
    print(f"  Total time: {timing['total']:.1f}s")
    print(f"  Output: {save_path}")

    return {
        "save_path": save_path,
        "planes": [p + 1 if p is not None else 1 for p in planes_to_process],
        "n_rois": total_rois,
        "stat": combined_stat,
        "iscell": combined_iscell,
        "plane_results": plane_results,
        "timing": timing,
    }


def load_cellpose_results(
    cellpose_dir: str | Path,
    plane_idx: int = None,
) -> dict:
    """
    Load Cellpose results from a directory.

    Parameters
    ----------
    cellpose_dir : str or Path
        Directory containing Cellpose outputs.
    plane_idx : int, optional
        Specific plane to load (0-indexed). If None, loads combined results.

    Returns
    -------
    dict
        Dictionary with 'masks', 'stat', 'iscell', 'flows', 'projection'.
    """
    cellpose_dir = Path(cellpose_dir)

    if plane_idx is not None:
        suffix = f"_plane{plane_idx:02d}"
    else:
        suffix = ""

    result = {}

    # load masks
    masks_file = cellpose_dir / f"masks{suffix}.npy"
    if masks_file.exists():
        result["masks"] = np.load(masks_file)
    else:
        # try without suffix
        masks_file = cellpose_dir / "masks.npy"
        if masks_file.exists():
            result["masks"] = np.load(masks_file)

    # load stat
    stat_file = cellpose_dir / f"stat{suffix}.npy"
    if stat_file.exists():
        result["stat"] = np.load(stat_file, allow_pickle=True)
    elif (cellpose_dir / "stat.npy").exists():
        result["stat"] = np.load(cellpose_dir / "stat.npy", allow_pickle=True)

    # load iscell
    iscell_file = cellpose_dir / f"iscell{suffix}.npy"
    if iscell_file.exists():
        result["iscell"] = np.load(iscell_file)
    elif (cellpose_dir / "iscell.npy").exists():
        result["iscell"] = np.load(cellpose_dir / "iscell.npy")

    # load flows
    flows_file = cellpose_dir / f"flows{suffix}.npy"
    if flows_file.exists():
        result["flows"] = np.load(flows_file, allow_pickle=True)

    # load projection
    import tifffile
    proj_file = cellpose_dir / f"projection{suffix}.tif"
    if proj_file.exists():
        result["projection"] = tifffile.imread(proj_file)

    # load metadata
    meta_file = cellpose_dir / f"cellpose_meta{suffix}.npy"
    if meta_file.exists():
        result["metadata"] = np.load(meta_file, allow_pickle=True).item()

    return result


def masks_to_stat(masks: np.ndarray, image: np.ndarray = None) -> np.ndarray:
    """
    Convert cellpose masks to suite2p stat array.

    Parameters
    ----------
    masks : ndarray
        2D or 3D label image (0=background, 1,2,...=roi ids).
    image : ndarray, optional
        Original image for intensity statistics.

    Returns
    -------
    ndarray
        Array of stat dictionaries compatible with suite2p.
    """
    return _masks_to_stat(masks, image)


def stat_to_masks(stat: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert suite2p stat array back to label mask.

    Parameters
    ----------
    stat : ndarray
        Array of stat dictionaries from suite2p.
    shape : tuple
        Output shape (Y, X) or (Z, Y, X).

    Returns
    -------
    ndarray
        Label mask (0=background, 1,2,...=roi ids).
    """
    masks = np.zeros(shape, dtype=np.uint32)

    for roi_id, s in enumerate(stat, start=1):
        ypix = s["ypix"]
        xpix = s["xpix"]
        if "zpix" in s and len(shape) == 3:
            zpix = s["zpix"]
            masks[zpix, ypix, xpix] = roi_id
        else:
            masks[ypix, xpix] = roi_id

    return masks


def _masks_to_outlines(masks: np.ndarray) -> np.ndarray:
    """Extract outlines from label mask."""
    from scipy import ndimage

    outlines = np.zeros_like(masks, dtype=bool)

    for roi_id in range(1, masks.max() + 1):
        roi_mask = masks == roi_id
        dilated = ndimage.binary_dilation(roi_mask)
        boundary = dilated & ~roi_mask
        outlines |= boundary

    return outlines


def save_gui_results(
    save_path: str | Path,
    masks: np.ndarray,
    image: np.ndarray,
    flows: tuple = None,
    styles: np.ndarray = None,
    diameter: float = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    name: str = None,
) -> Path:
    """
    Save cellpose results in GUI-compatible format.

    Creates:
    - {name}_seg.npy: cellpose gui format (can be loaded directly)
    - {name}_masks.tif: label image viewable in imagej/napari
    - {name}_stat.npy: suite2p-compatible roi statistics

    Parameters
    ----------
    save_path : str or Path
        Directory to save results.
    masks : ndarray
        Labeled mask array from cellpose (0=background, 1,2,...=roi ids).
    image : ndarray
        Image used for segmentation (projection).
    flows : tuple, optional
        Flow outputs from cellpose model.eval().
    styles : ndarray, optional
        Style vector from cellpose.
    diameter : float, optional
        Cell diameter used for segmentation.
    cellprob_threshold : float
        Cellprob threshold used.
    flow_threshold : float
        Flow threshold used.
    name : str, optional
        Base name for output files. Defaults to 'cellpose'.

    Returns
    -------
    Path
        Path to the _seg.npy file (for gui loading).
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    name = name or "cellpose"

    n_rois = int(masks.max())

    seg_data = {
        "img": image.astype(np.float32),
        "masks": masks.astype(np.uint32),
        "outlines": _masks_to_outlines(masks),
        "chan_choose": [0, 0],
        "ismanual": np.zeros(n_rois, dtype=bool),
        "filename": str(save_path / f"{name}.tif"),
        "flows": flows,
        "est_diam": diameter,
        "cellprob_threshold": cellprob_threshold,
        "flow_threshold": flow_threshold,
    }
    seg_file = save_path / f"{name}_seg.npy"
    np.save(seg_file, seg_data, allow_pickle=True)

    try:
        import tifffile
        tifffile.imwrite(
            save_path / f"{name}.tif",
            image.astype(np.float32),
            compression="zlib",
        )
        tifffile.imwrite(
            save_path / f"{name}_masks.tif",
            masks.astype(np.uint16),
            compression="zlib",
        )
    except ImportError:
        pass

    stat = masks_to_stat(masks, image)
    np.save(save_path / f"{name}_stat.npy", stat, allow_pickle=True)

    iscell = np.ones((n_rois, 2), dtype=np.float32)
    np.save(save_path / f"{name}_iscell.npy", iscell)

    print(f"saved {n_rois} rois to {save_path}")
    return seg_file


def load_seg_file(seg_path: str | Path) -> dict:
    """
    Load cellpose results from _seg.npy file.

    Parameters
    ----------
    seg_path : str or Path
        Path to _seg.npy file or directory containing it.

    Returns
    -------
    dict
        Dictionary with 'masks', 'img', 'flows', 'outlines', etc.
    """
    seg_path = Path(seg_path)

    if seg_path.is_dir():
        seg_files = list(seg_path.glob("*_seg.npy"))
        if not seg_files:
            raise FileNotFoundError(f"no _seg.npy files in {seg_path}")
        seg_path = seg_files[0]

    data = np.load(seg_path, allow_pickle=True).item()
    return data


def open_in_gui(
    seg_path: str | Path = None,
    image: np.ndarray = None,
    masks: np.ndarray = None,
):
    """
    Open cellpose gui with results or image.

    Parameters
    ----------
    seg_path : str or Path, optional
        Path to _seg.npy file to load in gui.
    image : ndarray, optional
        Image to open directly (without loading from file).
    masks : ndarray, optional
        Masks to overlay (requires image).

    Notes
    -----
    Requires cellpose to be installed with gui dependencies.
    """
    from cellpose.gui import gui

    if seg_path is not None:
        seg_path = Path(seg_path)
        if seg_path.is_dir():
            seg_files = list(seg_path.glob("*_seg.npy"))
            if seg_files:
                seg_path = seg_files[0]

        data = load_seg_file(seg_path)
        img_file = data.get("filename")
        if img_file and Path(img_file).exists():
            gui.run(image=str(img_file))
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
                import tifffile
                tifffile.imwrite(f.name, data["img"].astype(np.float32))
                gui.run(image=f.name)
    elif image is not None:
        import tempfile
        import tifffile
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tifffile.imwrite(f.name, image.astype(np.float32))
            if masks is not None:
                seg_data = {
                    "img": image.astype(np.float32),
                    "masks": masks.astype(np.uint32),
                    "outlines": _masks_to_outlines(masks),
                    "chan_choose": [0, 0],
                    "ismanual": np.zeros(int(masks.max()), dtype=bool),
                    "filename": f.name,
                    "flows": None,
                }
                seg_file = f.name.replace(".tif", "_seg.npy")
                np.save(seg_file, seg_data, allow_pickle=True)
            gui.run(image=f.name)
    else:
        gui.run()


def save_comparison(
    save_path: str | Path,
    results: dict,
    base_name: str = "comparison",
):
    """
    Save multiple cellpose results for comparison.

    Parameters
    ----------
    save_path : str or Path
        Directory to save results.
    results : dict
        Dictionary mapping method names to dicts with 'masks', 'proj', 'n_cells'.
    base_name : str
        Base name for output files.

    Example
    -------
    >>> save_comparison(
    ...     "output/",
    ...     {
    ...         "max": {"masks": masks1, "proj": proj1, "n_cells": 100},
    ...         "p99": {"masks": masks2, "proj": proj2, "n_cells": 120},
    ...     }
    ... )
    """
    import json

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    summary = []

    for method_name, data in results.items():
        masks = data["masks"]
        proj = data["proj"]
        n_cells = data.get("n_cells", int(masks.max()))

        safe_name = method_name.replace(" ", "_").replace("+", "_")

        save_gui_results(
            save_path,
            masks=masks,
            image=proj,
            name=f"{base_name}_{safe_name}",
        )

        summary.append({
            "method": method_name,
            "n_cells": int(n_cells),
            "file": f"{base_name}_{safe_name}_seg.npy",
        })

    with open(save_path / f"{base_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved {len(results)} comparisons to {save_path}")
    return save_path


def cellpose_to_suite2p(
    cellpose_dir: str | Path,
    suite2p_dir: str | Path = None,
    plane_idx: int = None,
):
    """
    Convert Cellpose results to Suite2p format for GUI viewing.

    Creates a Suite2p-compatible directory structure that can be
    opened in the Suite2p GUI.

    Parameters
    ----------
    cellpose_dir : str or Path
        Directory containing Cellpose outputs.
    suite2p_dir : str or Path, optional
        Output directory for Suite2p files. If None, creates 'suite2p/'
        subdirectory in cellpose_dir.
    plane_idx : int, optional
        Specific plane to convert (0-indexed).

    Returns
    -------
    Path
        Path to the Suite2p directory.
    """
    cellpose_dir = Path(cellpose_dir)
    if suite2p_dir is None:
        suite2p_dir = cellpose_dir / "suite2p"
    suite2p_dir = Path(suite2p_dir)

    # load cellpose results
    results = load_cellpose_results(cellpose_dir, plane_idx)

    # create plane directory
    plane_dir = suite2p_dir / "plane0"
    plane_dir.mkdir(parents=True, exist_ok=True)

    # save stat
    if "stat" in results:
        np.save(plane_dir / "stat.npy", results["stat"])

    # save iscell
    if "iscell" in results:
        np.save(plane_dir / "iscell.npy", results["iscell"])

    # create minimal ops
    ops = {
        "save_path": str(plane_dir),
        "Ly": results.get("masks", np.zeros((1, 1))).shape[-2],
        "Lx": results.get("masks", np.zeros((1, 1))).shape[-1],
        "nframes": 1,
        "fs": 1.0,
    }

    if "projection" in results:
        ops["meanImg"] = results["projection"]
        ops["max_proj"] = results["projection"]

    np.save(plane_dir / "ops.npy", ops)

    # create empty F, Fneu, spks if not present
    n_rois = len(results.get("stat", []))
    if n_rois > 0:
        np.save(plane_dir / "F.npy", np.zeros((n_rois, 1)))
        np.save(plane_dir / "Fneu.npy", np.zeros((n_rois, 1)))
        np.save(plane_dir / "spks.npy", np.zeros((n_rois, 1)))

    print(f"Converted to Suite2p format: {suite2p_dir}")
    return suite2p_dir
