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


from lbm_suite2p_python.utils import _is_lazy_array, _get_num_planes


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


def _generate_roi_colors(n_rois, seed=42):
    """generate distinct colors for rois using hsv colorspace."""
    np.random.seed(seed)
    colors = np.zeros((n_rois + 1, 3))
    for i in range(1, n_rois + 1):
        h = np.random.rand()
        s = 0.8
        v = 0.9
        # hsv to rgb conversion
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        colors[i] = [r + m, g + m, b + m]
    return colors


def _create_mask_overlay(img, masks, alpha=0.4):
    """create rgb overlay of masks on normalized grayscale image."""
    img_norm = _normalize_image(img)
    rgb = np.stack([img_norm] * 3, axis=-1)

    if masks.max() > 0:
        n_rois = int(masks.max())
        colors = _generate_roi_colors(n_rois)
        mask_rgb = colors[masks]
        mask_area = masks > 0
        rgb[mask_area] = (1 - alpha) * rgb[mask_area] + alpha * mask_rgb[mask_area]

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def _create_outline_overlay(img, masks, outline_color=(1, 1, 0)):
    """create rgb overlay of outlines on normalized grayscale image."""
    from scipy import ndimage

    img_norm = _normalize_image(img)
    rgb = np.stack([img_norm] * 3, axis=-1)

    if masks.max() > 0:
        # compute outlines
        outlines = np.zeros_like(masks, dtype=bool)
        for roi_id in range(1, masks.max() + 1):
            roi_mask = masks == roi_id
            dilated = ndimage.binary_dilation(roi_mask)
            boundary = dilated & ~roi_mask
            outlines |= boundary

        # apply outline color
        for c in range(3):
            rgb[:, :, c][outlines] = outline_color[c]

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def _masks_to_stat(masks, img=None, compute_overlap=True):
    """
    Convert Cellpose masks to Suite2p-style stat array.

    Parameters
    ----------
    masks : np.ndarray
        2D or 3D label image from Cellpose.
    img : np.ndarray, optional
        Original image for computing intensity weights (lam).
    compute_overlap : bool, default True
        Whether to compute overlap field (required for Suite2p extraction).

    Returns
    -------
    np.ndarray
        Array of stat dictionaries compatible with Suite2p extraction.
        Includes 'lam' (intensity weights) and 'overlap' (pixel overlap mask).
    """
    stat = []
    n_rois = int(masks.max())

    if n_rois == 0:
        return np.array([], dtype=object)

    # Build pixel count map for overlap detection
    if compute_overlap and masks.ndim == 2:
        # Count how many ROIs claim each pixel (for overlap computation)
        # Since cellpose masks are non-overlapping by design, we check boundaries
        from scipy import ndimage
        # Dilate each mask slightly to find potential overlaps at boundaries
        overlap_map = np.zeros(masks.shape, dtype=np.int32)
        for roi_id in range(1, n_rois + 1):
            roi_mask = masks == roi_id
            overlap_map += roi_mask.astype(np.int32)

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

        # Compute lam (intensity weights) - required for Suite2p extraction
        if img is not None:
            if img.ndim == 2:
                roi_vals = img[ypix, xpix].astype(np.float32)
            else:
                roi_vals = img[zpix, ypix, xpix].astype(np.float32) if zpix is not None else img[ypix, xpix].astype(np.float32)
            # Normalize to sum to 1 (Suite2p convention)
            roi_vals = roi_vals - roi_vals.min()  # shift to positive
            lam_sum = roi_vals.sum()
            if lam_sum > 0:
                lam = roi_vals / lam_sum
            else:
                lam = np.ones(npix, dtype=np.float32) / npix
        else:
            # Uniform weights if no image provided
            lam = np.ones(npix, dtype=np.float32) / npix

        # Compute overlap mask - required for Suite2p extraction
        if compute_overlap and masks.ndim == 2:
            # For cellpose, masks are non-overlapping, so overlap is all False
            # But we still need the field for Suite2p compatibility
            overlap = np.zeros(npix, dtype=bool)
        else:
            overlap = np.zeros(npix, dtype=bool)

        roi_stat = {
            "ypix": ypix.astype(np.int32),
            "xpix": xpix.astype(np.int32),
            "npix": npix,
            "lam": lam.astype(np.float32),
            "overlap": overlap,
            "med": np.array([med_y, med_x]),
            "radius": float(radius),
            "aspect_ratio": float(aspect),
            "compact": float(npix / (np.pi * radius**2)) if radius > 0 else 0.0,
        }

        if zpix is not None:
            roi_stat["zpix"] = zpix.astype(np.int32)
            roi_stat["med_z"] = float(np.median(zpix))

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
    - projection_masks.png: masks overlaid on normalized input image
    - projection_outlines.png: outlines overlaid on normalized input image
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

    # save visualization pngs
    from PIL import Image

    # masks overlay on normalized input image
    mask_overlay = _create_mask_overlay(img, masks)
    Image.fromarray(mask_overlay).save(save_dir / f"projection{plane_suffix}_masks.png")

    # outlines overlay on normalized input image
    outline_overlay = _create_outline_overlay(img, masks)
    Image.fromarray(outline_overlay).save(save_dir / f"projection{plane_suffix}_outlines.png")

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
    model_type: str = "cpsam",
    gpu: bool = True,
    # cellpose eval parameters
    diameter: float = None,
    flow_threshold: float = 0.0,
    cellprob_threshold: float = -6.0,
    min_size: int = 2,
    max_size: int = None,
    max_size_fraction: float = None,
    max_size_um: float = None,
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

    model_type : str, default 'cpsam'
        Cellpose model to use. Options:
        - 'cpsam': CP-SAM model (default, recommended for calcium imaging)
        - Path to custom trained model (must be trained from cpsam base)

        .. note::
            Currently only cpsam-based models are supported. Use
            ``lsp.train_cellpose()`` to fine-tune cpsam on your data, then
            pass the model path here.
    gpu : bool, default True
        Use GPU if available.
    diameter : float, optional
        Expected cell diameter in pixels. If None, Cellpose auto-estimates.
    flow_threshold : float, default 0.4
        Maximum allowed error of flows for each mask.
    cellprob_threshold : float, default 0.0
        Probability threshold for cell detection. Lower = more cells.
    min_size : int, default 2
        Minimum number of pixels per mask.
    max_size : int, optional
        Maximum number of pixels per mask. Overrides max_size_fraction and max_size_um.
    max_size_fraction : float, optional
        Maximum mask size as fraction of total image area. E.g., 0.4 means masks
        larger than 40% of image area are removed.
    max_size_um : float, optional
        Maximum cell diameter in microns. Converted to pixel area using dx/dy
        metadata from the array. E.g., 35 for cortical neurons. Requires array
        to have pixel size metadata.
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
        ├── masks_plane00.tif            # Label image for plane 0
        ├── masks_plane00.npy            # Same as numpy array
        ├── stat_plane00.npy             # Suite2p-compatible ROI stats
        ├── iscell_plane00.npy           # Cell classification (all accepted)
        ├── projection_plane00.tif       # Image used for segmentation
        ├── projection_plane00_masks.png # Masks overlaid on normalized input
        ├── projection_plane00_outlines.png  # Outlines on normalized input
        ├── cellpose_seg_plane00.npy     # Cellpose GUI-compatible output
        ├── flows_plane00.npy            # Flow fields
        └── cellpose_meta.npy            # Processing metadata

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

            # generate pngs if missing
            masks_png = save_path / f"projection{suffix}_masks.png"
            outlines_png = save_path / f"projection{suffix}_outlines.png"
            if not masks_png.exists() or not outlines_png.exists():
                import tifffile
                from PIL import Image
                proj_file = save_path / f"projection{suffix}.tif"
                if proj_file.exists():
                    proj = tifffile.imread(proj_file)
                    print(f"  Generating visualization PNGs...")
                    if not masks_png.exists():
                        mask_overlay = _create_mask_overlay(proj, masks)
                        Image.fromarray(mask_overlay).save(masks_png)
                    if not outlines_png.exists():
                        outline_overlay = _create_outline_overlay(proj, masks)
                        Image.fromarray(outline_overlay).save(outlines_png)
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

            # compute max_size: priority is max_size > max_size_um > max_size_fraction
            computed_max_size = None
            if max_size is not None:
                computed_max_size = max_size
            elif max_size_um is not None:
                # compute from microns using pixel size from array metadata
                pixel_size = None
                if hasattr(arr, "dx") and arr.dx is not None:
                    pixel_size = arr.dx
                elif hasattr(arr, "dy") and arr.dy is not None:
                    pixel_size = arr.dy
                if pixel_size is not None:
                    # max_size_um is diameter in microns, convert to pixel area
                    max_diameter_px = max_size_um / pixel_size
                    computed_max_size = int(np.pi * (max_diameter_px / 2) ** 2)
                    print(f"  Max size from {max_size_um} um: {computed_max_size} pixels (pixel size: {pixel_size:.3f} um)")
                else:
                    print(f"  Warning: max_size_um specified but no pixel size in array metadata")
            elif max_size_fraction is not None:
                # compute as fraction of total image area
                total_pixels = proj.shape[-2] * proj.shape[-1]
                computed_max_size = int(total_pixels * max_size_fraction)
                print(f"  Max size from fraction {max_size_fraction}: {computed_max_size} pixels")

            if computed_max_size is not None:
                eval_kwargs["max_size"] = computed_max_size

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
                    "min_size": min_size,
                    "max_size": computed_max_size,
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
    path: str | Path = None,
    image: np.ndarray = None,
    masks: np.ndarray = None,
    # legacy parameter name
    seg_path: str | Path = None,
):
    """
    Open cellpose GUI for viewing results or annotating images.

    This function launches the cellpose GUI for:
    - Viewing existing segmentation results
    - Manually annotating images to create training data
    - Correcting automatic segmentations

    Parameters
    ----------
    path : str or Path, optional
        Path to open. Can be:
        - Directory containing .tif images (for annotation)
        - Directory containing _seg.npy files (to view/edit results)
        - Path to specific _seg.npy file
        - Path to specific image file
    image : ndarray, optional
        Image array to open directly (without file).
    masks : ndarray, optional
        Masks to overlay on image (requires image parameter).
    seg_path : str or Path, optional
        Deprecated. Use ``path`` instead.

    Notes
    -----
    Requires cellpose to be installed with GUI dependencies::

        pip install cellpose[gui]

    Training workflow:
    1. Prepare images with ``lsp.annotate()`` or save projections as .tif
    2. Open with ``lsp.open_in_gui(path)``
    3. Draw cell masks: Ctrl+click to start, click to add points, Enter to finish
    4. Delete masks: select + Delete key
    5. Save annotations: Ctrl+S (creates _seg.npy files)
    6. Train model: ``lsp.train_cellpose(path, mask_filter='_seg.npy')``

    Examples
    --------
    Open a directory of images for annotation:

    >>> import lbm_suite2p_python as lsp
    >>> lsp.open_in_gui("D:/annotations")

    View existing segmentation results:

    >>> lsp.open_in_gui("D:/results/cellpose")

    Open a specific file:

    >>> lsp.open_in_gui("D:/results/projection_seg.npy")

    See Also
    --------
    annotate : Prepare images for annotation
    train_cellpose : Train model on annotated data
    save_gui_results : Save results in GUI-compatible format
    """
    import warnings

    # handle legacy parameter
    if seg_path is not None and path is None:
        warnings.warn(
            "seg_path parameter is deprecated, use path instead",
            DeprecationWarning,
            stacklevel=2,
        )
        path = seg_path

    # patch QCheckBox for Qt5/Qt6 compatibility
    try:
        from qtpy.QtWidgets import QCheckBox
        if not hasattr(QCheckBox, 'checkStateChanged'):
            QCheckBox.checkStateChanged = QCheckBox.stateChanged
    except ImportError:
        pass

    from cellpose.gui import gui

    if path is not None:
        path = Path(path)

        if path.is_dir():
            # check for _seg.npy files first
            seg_files = list(path.glob("*_seg.npy"))
            tif_files = list(path.glob("*.tif")) + list(path.glob("*.tiff"))

            if seg_files:
                # load first seg file
                print(f"Found {len(seg_files)} _seg.npy files")
                data = load_seg_file(seg_files[0])
                img_file = data.get("filename")
                if img_file and Path(img_file).exists():
                    print(f"Opening: {img_file}")
                    gui.run(image=str(img_file))
                else:
                    # save image to temp and open
                    import tempfile
                    import tifffile
                    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
                        tifffile.imwrite(f.name, data["img"].astype(np.float32))
                        gui.run(image=f.name)
            elif tif_files:
                # open first tif file
                print(f"Found {len(tif_files)} image files in {path}")
                print(f"Opening: {tif_files[0].name}")
                print("\nTo annotate: Ctrl+click to draw, Enter to finish, Ctrl+S to save")
                gui.run(image=str(tif_files[0]))
            else:
                print(f"No .tif or _seg.npy files found in {path}")
                print("Opening empty GUI...")
                gui.run()

        elif path.suffix == ".npy":
            # load seg file
            data = load_seg_file(path)
            img_file = data.get("filename")
            if img_file and Path(img_file).exists():
                gui.run(image=str(img_file))
            else:
                import tempfile
                import tifffile
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
                    tifffile.imwrite(f.name, data["img"].astype(np.float32))
                    gui.run(image=f.name)

        elif path.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            # open image file directly
            print(f"Opening: {path}")
            print("\nTo annotate: Ctrl+click to draw, Enter to finish, Ctrl+S to save")
            gui.run(image=str(path))

        else:
            print(f"Unsupported file type: {path.suffix}")
            gui.run()

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


# supported base model for training (currently only cpsam)
_SUPPORTED_BASE_MODEL = "cpsam"


def train_cellpose(
    train_dir: str | Path,
    test_dir: str | Path = None,
    model_name: str = "lbm_custom",
    learning_rate: float = 1e-5,
    weight_decay: float = 0.1,
    n_epochs: int = 100,
    batch_size: int = 1,
    min_train_masks: int = 5,
    save_every: int = 100,
    save_each: bool = False,
    mask_filter: str = "_masks",
    image_filter: str = None,
    gpu: bool = True,
    normalize: bool = True,
) -> Path:
    """
    Train/fine-tune a Cellpose model on your data.

    Fine-tunes the cpsam (Cellpose-SAM) model on your labeled images. The trained
    model can then be used with ``lsp.pipeline()`` or ``lsp.cellpose()`` by passing
    the model path as ``pretrained_model`` or ``model_type``.

    .. note::
        Currently only fine-tuning from cpsam is supported. Custom base models
        are not yet available.

    Parameters
    ----------
    train_dir : str or Path
        Directory containing training images and masks. Images should be named
        like ``image.tif`` with corresponding ``image_masks.tif`` (or use
        ``mask_filter`` to specify a different suffix). Can also use GUI
        annotations (``image_seg.npy``) with ``mask_filter="_seg.npy"``.
    test_dir : str or Path, optional
        Directory containing test images and masks for validation. If None,
        no validation is performed during training.
    model_name : str, default "lbm_custom"
        Name for the trained model. The model will be saved to
        ``{train_dir}/models/{model_name}``.
    learning_rate : float, default 1e-5
        Learning rate for training. The default is optimized for fine-tuning.
    weight_decay : float, default 0.1
        L2 regularization weight decay.
    n_epochs : int, default 100
        Number of training epochs.
    batch_size : int, default 1
        Batch size for training. Keep small (1-2) for fine-tuning.
    min_train_masks : int, default 5
        Minimum number of masks an image must have to be included in training.
    save_every : int, default 100
        Save checkpoint every N epochs.
    save_each : bool, default False
        If True, save a separate checkpoint file for each epoch.
    mask_filter : str, default "_masks"
        Suffix for mask files. Use ``"_seg.npy"`` for GUI annotations.
    image_filter : str, optional
        Suffix for image files (e.g., ``"_img"`` for ``wells_000_img.tif``).
    gpu : bool, default True
        Use GPU if available.
    normalize : bool, default True
        Normalize images during training.

    Returns
    -------
    Path
        Path to the trained model file.

    Examples
    --------
    Basic training workflow:

    >>> import lbm_suite2p_python as lsp
    >>>
    >>> # 1. prepare your training data (images + masks in a folder)
    >>> # 2. train the model
    >>> model_path = lsp.train_cellpose(
    ...     train_dir="D:/training_data",
    ...     test_dir="D:/test_data",
    ...     model_name="my_neurons",
    ...     n_epochs=200,
    ... )
    >>>
    >>> # 3. use the trained model
    >>> result = lsp.cellpose(
    ...     "D:/new_data",
    ...     model_type=str(model_path),
    ... )

    Using GUI annotations for training:

    >>> # after annotating images in cellpose gui, train with _seg.npy files
    >>> model_path = lsp.train_cellpose(
    ...     train_dir="D:/annotated_images",
    ...     mask_filter="_seg.npy",
    ...     model_name="gui_trained",
    ... )

    See Also
    --------
    prepare_training_data : Organize pipeline outputs for training
    cellpose : Run segmentation with trained model
    open_in_gui : Open images for manual annotation
    """
    from cellpose import models, train, io, core

    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    if test_dir is not None:
        test_dir = Path(test_dir)
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

    print("Cellpose Model Training")
    print("=" * 60)
    print(f"Training directory: {train_dir}")
    if test_dir:
        print(f"Test directory: {test_dir}")
    print(f"Model name: {model_name}")
    print(f"Base model: {_SUPPORTED_BASE_MODEL} (only supported option)")
    print(f"Epochs: {n_epochs}, LR: {learning_rate}, WD: {weight_decay}")

    # check gpu
    use_gpu = gpu and core.use_gpu()
    print(f"GPU: {'enabled' if use_gpu else 'disabled'}")

    # load training data
    print(f"\nLoading training data (mask_filter='{mask_filter}')...")
    output = io.load_train_test_data(
        train_dir=str(train_dir),
        test_dir=str(test_dir) if test_dir else None,
        image_filter=image_filter,
        mask_filter=mask_filter,
        look_one_level_down=False,
    )
    images, labels, image_names, test_images, test_labels, test_image_names = output

    print(f"  Training images: {len(images)}")
    if test_images is not None:
        print(f"  Test images: {len(test_images)}")

    if len(images) == 0:
        raise ValueError(
            f"No training images found in {train_dir}. "
            f"Expected image files with matching '{mask_filter}' mask files."
        )

    # load base model (cpsam only)
    print(f"\nLoading base model ({_SUPPORTED_BASE_MODEL})...")
    model = models.CellposeModel(model_type=_SUPPORTED_BASE_MODEL, gpu=use_gpu)

    # train
    print(f"\nStarting training...")
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        train_files=None,
        test_data=test_images,
        test_labels=test_labels,
        test_files=None,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        batch_size=batch_size,
        min_train_masks=min_train_masks,
        save_every=save_every,
        save_each=save_each,
        model_name=model_name,
        save_path=str(train_dir),
        normalize=normalize,
    )

    model_path = Path(model_path)

    # save training losses
    losses_file = model_path.parent / f"{model_name}_losses.npy"
    np.save(losses_file, {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "base_model": _SUPPORTED_BASE_MODEL,
    })

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Model saved: {model_path}")
    print(f"Losses saved: {losses_file}")
    print(f"\nTo use this model:")
    print(f"  lsp.cellpose(..., model_type='{model_path}')")
    print(f"  or")
    print(f"  lsp.pipeline(..., ops={{'pretrained_model': '{model_path}'}})")

    return model_path


def prepare_training_data(
    source_dirs: list | str | Path,
    output_dir: str | Path,
    projection: Literal["max", "mean", "std"] = "max",
    use_seg_files: bool = True,
    use_mask_files: bool = True,
    copy_images: bool = True,
    image_suffix: str = ".tif",
) -> Path:
    """
    Prepare training data from pipeline outputs or annotated images.

    Collects images and masks from multiple source directories and organizes
    them into the format expected by ``train_cellpose()``. This is useful for
    combining annotations from multiple experiments.

    Parameters
    ----------
    source_dirs : list, str, or Path
        Directory or list of directories containing:
        - ``*_seg.npy`` files (cellpose GUI annotations), and/or
        - ``*_masks.tif`` files with corresponding images
        - ``projection*.tif`` or other image files
    output_dir : str or Path
        Output directory for organized training data.
    projection : str, default "max"
        If source has time-series data, which projection to use.
    use_seg_files : bool, default True
        Look for ``_seg.npy`` files (GUI annotations).
    use_mask_files : bool, default True
        Look for ``_masks.tif`` or ``masks*.npy`` files.
    copy_images : bool, default True
        Copy image files to output directory. If False, creates symlinks.
    image_suffix : str, default ".tif"
        Suffix for output image files.

    Returns
    -------
    Path
        Path to output directory ready for training.

    Examples
    --------
    Collect training data from multiple pipeline runs:

    >>> import lbm_suite2p_python as lsp
    >>>
    >>> # collect from multiple annotated folders
    >>> train_dir = lsp.prepare_training_data(
    ...     source_dirs=[
    ...         "D:/experiment1/cellpose",
    ...         "D:/experiment2/cellpose",
    ...     ],
    ...     output_dir="D:/training_data",
    ... )
    >>>
    >>> # train on collected data
    >>> model_path = lsp.train_cellpose(train_dir)

    See Also
    --------
    train_cellpose : Train model on prepared data
    save_gui_results : Save results in GUI-compatible format
    """
    import shutil
    import tifffile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # normalize source_dirs to list
    if isinstance(source_dirs, (str, Path)):
        source_dirs = [source_dirs]
    source_dirs = [Path(d) for d in source_dirs]

    print("Preparing Training Data")
    print("=" * 60)
    print(f"Source directories: {len(source_dirs)}")
    print(f"Output directory: {output_dir}")

    collected = 0

    for src_dir in source_dirs:
        if not src_dir.exists():
            print(f"  Warning: {src_dir} not found, skipping")
            continue

        print(f"\nScanning: {src_dir}")

        # find _seg.npy files (gui annotations)
        if use_seg_files:
            for seg_file in src_dir.rglob("*_seg.npy"):
                try:
                    data = np.load(seg_file, allow_pickle=True).item()
                    masks = data.get("masks")
                    img = data.get("img")

                    if masks is None or img is None:
                        continue

                    if masks.max() == 0:
                        print(f"    Skipping {seg_file.name}: no masks")
                        continue

                    # create output name
                    base_name = seg_file.stem.replace("_seg", "")
                    out_name = f"{src_dir.name}_{base_name}"

                    # save image
                    img_out = output_dir / f"{out_name}{image_suffix}"
                    tifffile.imwrite(img_out, img.astype(np.float32), compression="zlib")

                    # save masks
                    masks_out = output_dir / f"{out_name}_masks{image_suffix}"
                    tifffile.imwrite(masks_out, masks.astype(np.uint16), compression="zlib")

                    n_masks = int(masks.max())
                    print(f"    {seg_file.name}: {n_masks} masks")
                    collected += 1

                except Exception as e:
                    print(f"    Error loading {seg_file}: {e}")

        # find masks.npy or *_masks.tif files
        if use_mask_files:
            for masks_file in list(src_dir.rglob("masks*.npy")) + list(src_dir.rglob("*_masks.tif")):
                try:
                    # skip if we already got this from _seg.npy
                    if "_seg" in str(masks_file):
                        continue

                    # load masks
                    if masks_file.suffix == ".npy":
                        masks = np.load(masks_file)
                    else:
                        masks = tifffile.imread(masks_file)

                    if masks.max() == 0:
                        print(f"    Skipping {masks_file.name}: no masks")
                        continue

                    # find corresponding image
                    base = masks_file.stem.replace("_masks", "").replace("masks", "")
                    img_candidates = [
                        masks_file.parent / f"projection{base}.tif",
                        masks_file.parent / f"{base}.tif",
                        masks_file.parent / f"{base}_projection.tif",
                    ]

                    img = None
                    for cand in img_candidates:
                        if cand.exists():
                            img = tifffile.imread(cand)
                            break

                    if img is None:
                        print(f"    Skipping {masks_file.name}: no matching image")
                        continue

                    # create output name
                    out_name = f"{src_dir.name}_{base}" if base else f"{src_dir.name}_{masks_file.stem}"

                    # save image
                    img_out = output_dir / f"{out_name}{image_suffix}"
                    tifffile.imwrite(img_out, img.astype(np.float32), compression="zlib")

                    # save masks
                    masks_out = output_dir / f"{out_name}_masks{image_suffix}"
                    tifffile.imwrite(masks_out, masks.astype(np.uint16), compression="zlib")

                    n_masks = int(masks.max())
                    print(f"    {masks_file.name}: {n_masks} masks")
                    collected += 1

                except Exception as e:
                    print(f"    Error loading {masks_file}: {e}")

    print(f"\n{'='*60}")
    print(f"Collected {collected} image-mask pairs")
    print(f"Output directory: {output_dir}")
    print(f"\nTo train:")
    print(f"  lsp.train_cellpose('{output_dir}')")

    return output_dir


def annotate(
    input_data,
    save_path: str | Path = None,
    planes: list | int = None,
    projection: Literal["max", "mean", "std"] = "max",
    reader_kwargs: dict = None,
) -> Path:
    """
    Prepare images for annotation in cellpose GUI.

    Computes projections from input data and saves them in a format ready
    for manual annotation in the cellpose GUI. After annotating, the
    resulting ``_seg.npy`` files can be used for training.

    Parameters
    ----------
    input_data : str, Path, or array
        Input data (same formats as ``lsp.cellpose()``).
    save_path : str or Path, optional
        Output directory for projection images.
    planes : int or list, optional
        Which planes to prepare (1-indexed).
    projection : str, default "max"
        Projection method for time series.
    reader_kwargs : dict, optional
        Arguments passed to mbo_utilities.imread().

    Returns
    -------
    Path
        Path to directory containing images ready for annotation.

    Examples
    --------
    Prepare images for annotation:

    >>> import lbm_suite2p_python as lsp
    >>>
    >>> # 1. prepare projections
    >>> annotation_dir = lsp.annotate("D:/data.zarr", planes=[5, 10, 15])
    >>>
    >>> # 2. open in gui and annotate (draws masks, saves _seg.npy)
    >>> lsp.open_in_gui(annotation_dir)
    >>>
    >>> # 3. train on annotations
    >>> model_path = lsp.train_cellpose(annotation_dir, mask_filter="_seg.npy")

    See Also
    --------
    open_in_gui : Launch cellpose GUI
    train_cellpose : Train model on annotated data
    """
    import tifffile

    reader_kwargs = reader_kwargs or {}

    # load input data
    if _is_lazy_array(input_data):
        arr = input_data
        if save_path is None:
            filenames = getattr(arr, "filenames", [])
            if filenames:
                save_path = Path(filenames[0]).parent / "annotations"
            else:
                raise ValueError("save_path required for array input")
    elif isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
        arr = imread(input_path, **reader_kwargs)
        if save_path is None:
            save_path = (input_path.parent if input_path.is_file() else input_path) / "annotations"
    else:
        raise TypeError(f"input_data must be path or array, got {type(input_data)}")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print("Preparing Images for Annotation")
    print("=" * 60)
    print(f"Output: {save_path}")

    # get planes
    num_planes = _get_num_planes(arr)
    if planes is None:
        planes_to_process = list(range(num_planes))
    elif isinstance(planes, int):
        planes_to_process = [planes - 1]  # convert to 0-indexed
    else:
        planes_to_process = [p - 1 for p in planes]

    print(f"Planes: {[p+1 for p in planes_to_process]}")

    for plane_idx in planes_to_process:
        print(f"\nPlane {plane_idx + 1}:")

        # compute projection
        proj = _compute_projection(arr, plane_idx=plane_idx, method=projection)

        # save as tiff
        out_file = save_path / f"plane{plane_idx+1:02d}.tif"
        tifffile.imwrite(out_file, proj.astype(np.float32), compression="zlib")
        print(f"  Saved: {out_file.name}")

    print(f"\n{'='*60}")
    print(f"Images saved to: {save_path}")
    print(f"\nTo annotate:")
    print(f"  1. lsp.open_in_gui('{save_path}')")
    print(f"  2. Draw masks on each image (Ctrl+click to add cells)")
    print(f"  3. Save (Ctrl+S) to create _seg.npy files")
    print(f"\nTo train after annotation:")
    print(f"  lsp.train_cellpose('{save_path}', mask_filter='_seg.npy')")

    return save_path
