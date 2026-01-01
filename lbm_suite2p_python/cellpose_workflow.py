"""
Cellpose workflow functions for human-in-the-loop training.

This module provides functions for:
- Enhancing summary images for better cellpose detection
- Re-running detection with trained models or pre-computed masks
- Extracting traces from cellpose masks

Typical workflow:
    1. Run initial pipeline: lsp.pipeline(data)
    2. Evaluate traces, decide detection needs improvement
    3. Enhance summary image: lsp.enhance_summary_image(path)
    4. Annotate in GUI: lsp.open_in_gui(path)
    5. Train model: lsp.train_cellpose(path)
    6. Re-detect: lsp.redetect(path, model_path=model)
"""

from pathlib import Path
from typing import Literal

import numpy as np

from lbm_suite2p_python.cellpose import _masks_to_stat


def enhance_summary_image(
    suite2p_path: str | Path,
    method: Literal["meanImg", "meanImgE", "max_proj", "log_ratio", "correlation"] = "meanImgE",
    output_path: str | Path = None,
    denoise: bool = True,
    spatial_hp: int = 0,
    percentile_range: tuple = (1, 99),
) -> Path:
    """
    Create or enhance a summary image for cellpose annotation.

    Extracts or computes a summary image from Suite2p outputs, optionally
    applying enhancements to improve cellpose detection.

    Parameters
    ----------
    suite2p_path : str or Path
        Path to Suite2p plane directory containing ops.npy.
    method : str, default "meanImgE"
        Summary image type:
        - "meanImg": Raw mean image
        - "meanImgE": Enhanced mean (spatial high-pass)
        - "max_proj": Maximum projection (binned + temporal HP)
        - "log_ratio": log(max_proj / meanImg) - emphasizes active regions
        - "correlation": Vcorr correlation image (if available)
    output_path : str or Path, optional
        Output file path. Defaults to suite2p_path/enhanced_{method}.tif.
    denoise : bool, default True
        Apply PCA denoising to reduce noise.
    spatial_hp : int, default 0
        Spatial high-pass filter size. 0 to disable.
    percentile_range : tuple, default (1, 99)
        Percentile range for intensity normalization.

    Returns
    -------
    Path
        Path to the saved enhanced image.

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>>
    >>> # After initial pipeline run, enhance the summary image
    >>> img_path = lsp.enhance_summary_image(
    ...     "output/plane01",
    ...     method="log_ratio",
    ...     denoise=True,
    ... )
    >>>
    >>> # Open in cellpose GUI for annotation
    >>> lsp.open_in_gui(img_path)

    See Also
    --------
    annotate : Prepare images for annotation from raw data
    export_for_gui : Export existing Suite2p results for GUI
    """
    import tifffile
    from mbo_utilities.util import load_npy

    suite2p_path = Path(suite2p_path)
    ops_path = suite2p_path / "ops.npy" if suite2p_path.is_dir() else suite2p_path
    if not ops_path.exists():
        raise FileNotFoundError(f"ops.npy not found: {ops_path}")

    ops = load_npy(ops_path).item()

    print("Enhancing Summary Image")
    print("-" * 40)
    print(f"Source: {suite2p_path}")
    print(f"Method: {method}")

    # Get base image based on method
    if method == "meanImg":
        if "meanImg" not in ops:
            raise ValueError("meanImg not found in ops. Run registration first.")
        img = ops["meanImg"].astype(np.float32)

    elif method == "meanImgE":
        if "meanImgE" not in ops:
            raise ValueError("meanImgE not found in ops. Run registration first.")
        img = ops["meanImgE"].astype(np.float32)

    elif method == "max_proj":
        if "max_proj" not in ops:
            raise ValueError("max_proj not found in ops. Run detection first.")
        img = ops["max_proj"].astype(np.float32)

    elif method == "log_ratio":
        if "max_proj" not in ops or "meanImg" not in ops:
            raise ValueError("max_proj and meanImg required for log_ratio.")
        max_proj = ops["max_proj"].astype(np.float32)
        mean_img = ops["meanImg"].astype(np.float32)
        mean_img = np.maximum(mean_img, np.percentile(mean_img, 1))
        img = np.log(max_proj / mean_img + 1)

    elif method == "correlation":
        if "Vcorr" not in ops:
            raise ValueError("Vcorr not found in ops. Run functional detection first.")
        img = ops["Vcorr"].astype(np.float32)

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"  Base image shape: {img.shape}")

    # Apply spatial high-pass if requested
    if spatial_hp > 0:
        from scipy.ndimage import uniform_filter
        img_hp = img - uniform_filter(img, size=spatial_hp)
        img = img_hp
        print(f"  Applied spatial high-pass (size={spatial_hp})")

    # Apply PCA denoising if requested
    if denoise:
        try:
            from suite2p.detection.denoise import pca_denoise
            # pca_denoise expects (n_frames, Ly, Lx), block_size, n_comps_frac
            block_size = [img.shape[0] // 4, img.shape[1] // 4]
            block_size = [max(16, b) for b in block_size]  # minimum block size
            img = pca_denoise(img[np.newaxis, ...], block_size=block_size, n_comps_frac=0.5)[0]
            print("  Applied PCA denoising")
        except (ImportError, Exception) as e:
            print(f"  Warning: PCA denoising failed: {e}")

    # Normalize to percentile range
    p_low, p_high = np.percentile(img, percentile_range)
    img = np.clip(img, p_low, p_high)
    img = (img - p_low) / (p_high - p_low + 1e-10)
    print(f"  Normalized to {percentile_range} percentile range")

    # Save
    if output_path is None:
        output_path = suite2p_path / f"enhanced_{method}.tif"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, img.astype(np.float32), compression="zlib")

    print(f"\nEnhanced image saved: {output_path}")
    print(f"\nTo annotate: lsp.open_in_gui('{output_path}')")

    return output_path


def redetect(
    suite2p_path: str | Path,
    model_path: str | Path = None,
    masks_path: str | Path = None,
    # cellpose parameters (used if model_path provided)
    diameter: float = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    # extraction parameters
    neucoeff: float = 0.7,
    # processing options
    run_extraction: bool = True,
    run_classification: bool = True,
    run_spikes: bool = True,
    overwrite: bool = True,
) -> Path:
    """
    Re-run detection on existing registered data with a new model or masks.

    Uses existing registration (data.bin) and re-runs ROI detection with
    either a trained cellpose model or pre-computed masks. Then extracts
    fluorescence traces, runs classification, and computes spikes.

    Parameters
    ----------
    suite2p_path : str or Path
        Path to Suite2p plane directory containing ops.npy and data.bin.
    model_path : str or Path, optional
        Path to trained cellpose model. If provided, runs cellpose detection.
    masks_path : str or Path, optional
        Path to pre-computed masks (masks.npy or _seg.npy). If provided,
        uses these masks directly instead of running cellpose.
    diameter : float, optional
        Cell diameter for cellpose. If None, auto-estimates.
    flow_threshold : float, default 0.4
        Cellpose flow threshold.
    cellprob_threshold : float, default 0.0
        Cellpose cell probability threshold.
    min_size : int, default 15
        Minimum cell size in pixels.
    neucoeff : float, default 0.7
        Neuropil coefficient for trace extraction.
    run_extraction : bool, default True
        Extract fluorescence traces (F, Fneu).
    run_classification : bool, default True
        Run ROI classifier.
    run_spikes : bool, default True
        Run spike deconvolution.
    overwrite : bool, default True
        Overwrite existing stat.npy, F.npy, etc.

    Returns
    -------
    Path
        Path to updated ops.npy file.

    Examples
    --------
    Re-detect with a trained model:

    >>> import lbm_suite2p_python as lsp
    >>>
    >>> # After training a custom model
    >>> model = lsp.train_cellpose("training_data/", model_name="my_model")
    >>>
    >>> # Re-detect on existing pipeline output
    >>> lsp.redetect("output/plane01", model_path=model)

    Re-detect with pre-computed masks from lsp.cellpose():

    >>> # Run cellpose separately
    >>> result = lsp.cellpose("data.zarr", planes=[1])
    >>>
    >>> # Use those masks for extraction
    >>> lsp.redetect("output/plane01", masks_path=result["save_path"] / "masks.npy")

    See Also
    --------
    pipeline : Full processing pipeline
    train_cellpose : Train custom cellpose model
    cellpose : Run cellpose detection directly
    """
    import time
    import tifffile
    from cellpose import models, core
    from suite2p.extraction import extraction_wrapper
    from suite2p.classification import classify, builtin_classfile
    from suite2p.extraction import preprocess, oasis
    from suite2p.io.binary import BinaryFile
    from mbo_utilities.util import load_npy

    suite2p_path = Path(suite2p_path)
    ops_path = suite2p_path / "ops.npy"

    if not ops_path.exists():
        raise FileNotFoundError(f"ops.npy not found: {ops_path}")

    if model_path is None and masks_path is None:
        raise ValueError("Either model_path or masks_path must be provided")

    ops = load_npy(ops_path).item()

    print("Re-detecting ROIs")
    print("-" * 40)
    print(f"Suite2p path: {suite2p_path}")

    # Check for existing files
    stat_path = suite2p_path / "stat.npy"
    f_path = suite2p_path / "F.npy"
    if stat_path.exists() and not overwrite:
        raise FileExistsError(f"stat.npy exists. Use overwrite=True to replace.")

    # Get image dimensions and paths
    Ly, Lx = ops["Ly"], ops["Lx"]
    reg_file = ops.get("reg_file", suite2p_path / "data.bin")
    reg_file = Path(reg_file)
    if not reg_file.exists():
        reg_file = suite2p_path / "data.bin"
    if not reg_file.exists():
        raise FileNotFoundError(f"Registered binary not found: {reg_file}")

    n_frames = ops.get("nframes", None)
    if n_frames is None:
        n_frames = reg_file.stat().st_size // (Ly * Lx * 2)

    print(f"  Image size: {Ly} x {Lx}")
    print(f"  Frames: {n_frames}")

    # Get summary image for lam computation
    if "meanImg" in ops:
        summary_img = ops["meanImg"].astype(np.float32)
    elif "meanImgE" in ops:
        summary_img = ops["meanImgE"].astype(np.float32)
    else:
        summary_img = None

    t0 = time.time()
    extraction_time = 0
    classification_time = 0
    spikes_time = 0

    # DETECTION
    if masks_path is not None:
        masks_path = Path(masks_path)
        print(f"\nLoading masks from: {masks_path}")

        if masks_path.suffix == ".npy":
            data = np.load(masks_path, allow_pickle=True)
            if data.dtype == object:
                data = data.item()
                masks = data.get("masks", data.get("outlines"))
            else:
                masks = data
        elif masks_path.suffix in (".tif", ".tiff"):
            masks = tifffile.imread(masks_path)
        else:
            raise ValueError(f"Unknown mask format: {masks_path.suffix}")

        print(f"  Loaded masks: {masks.shape}, {int(masks.max())} ROIs")
        stat = _masks_to_stat(masks, img=summary_img)
        print(f"  Converted to stat: {len(stat)} ROIs")

    else:
        print(f"\nRunning Cellpose detection")
        print(f"  Model: {model_path}")

        use_gpu = core.use_gpu()
        print(f"  GPU: {'enabled' if use_gpu else 'disabled'}")

        model = models.CellposeModel(
            model_type=str(model_path) if model_path else "cpsam",
            gpu=use_gpu,
        )

        if "meanImgE" in ops:
            detect_img = ops["meanImgE"]
        elif "max_proj" in ops:
            detect_img = ops["max_proj"]
        elif "meanImg" in ops:
            detect_img = ops["meanImg"]
        else:
            raise ValueError("No suitable image for detection in ops")

        detect_img = detect_img.astype(np.float32)

        masks, flows, styles = model.eval(
            detect_img,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
        )

        print(f"  Detected {int(masks.max())} ROIs")
        stat = _masks_to_stat(masks, img=summary_img)

        np.save(suite2p_path / "masks.npy", masks)
        tifffile.imwrite(
            suite2p_path / "masks.tif",
            masks.astype(np.uint16),
            compression="zlib",
        )

    detection_time = time.time() - t0
    print(f"  Detection time: {detection_time:.1f}s")

    if len(stat) == 0:
        print("\nWarning: No ROIs detected!")
        np.save(stat_path, stat)
        np.save(ops_path, ops)
        return ops_path

    # EXTRACTION
    if run_extraction:
        print(f"\nExtracting fluorescence traces...")
        t1 = time.time()

        ops["neucoeff"] = neucoeff
        ops["allow_overlap"] = True
        ops["inner_neuropil_radius"] = ops.get("inner_neuropil_radius", 2)
        ops["min_neuropil_pixels"] = ops.get("min_neuropil_pixels", 350)

        with BinaryFile(Ly=Ly, Lx=Lx, filename=str(reg_file), n_frames=n_frames) as f_reg:
            stat, F, Fneu, F_chan2, Fneu_chan2 = extraction_wrapper(
                stat, f_reg, f_reg_chan2=None, ops=ops
            )

        extraction_time = time.time() - t1
        print(f"  Extracted {F.shape[0]} traces, {F.shape[1]} frames")
        print(f"  Extraction time: {extraction_time:.1f}s")

        np.save(f_path, F)
        np.save(suite2p_path / "Fneu.npy", Fneu)
    else:
        F = None
        Fneu = None

    # CLASSIFICATION
    if run_classification and F is not None:
        print(f"\nRunning ROI classification...")
        t2 = time.time()

        classfile = ops.get("classifier_path", builtin_classfile)
        iscell = classify(stat=stat, classfile=classfile)

        classification_time = time.time() - t2
        print(f"  Classified {int(iscell[:, 0].sum())}/{len(iscell)} as cells")
        print(f"  Classification time: {classification_time:.1f}s")

        np.save(suite2p_path / "iscell.npy", iscell)
    else:
        iscell = np.ones((len(stat), 2), dtype=np.float32)
        iscell[:, 1] = 0.5
        np.save(suite2p_path / "iscell.npy", iscell)

    # SPIKE DECONVOLUTION
    if run_spikes and F is not None:
        print(f"\nRunning spike deconvolution...")
        t3 = time.time()

        dF = F - ops["neucoeff"] * Fneu

        dF = preprocess(
            F=dF,
            baseline=ops.get("baseline", "maximin"),
            win_baseline=ops.get("win_baseline", 60.0),
            sig_baseline=ops.get("sig_baseline", 10.0),
            fs=ops.get("fs", 10.0),
            prctile_baseline=ops.get("prctile_baseline", 8.0),
        )

        spks = oasis(
            F=dF,
            batch_size=ops.get("batch_size", 500),
            tau=ops.get("tau", 1.0),
            fs=ops.get("fs", 10.0),
        )

        spikes_time = time.time() - t3
        print(f"  Deconvolution time: {spikes_time:.1f}s")

        np.save(suite2p_path / "spks.npy", spks)
    else:
        spks = np.zeros_like(F) if F is not None else None
        if spks is not None:
            np.save(suite2p_path / "spks.npy", spks)

    # Save stat and ops
    np.save(stat_path, stat)

    ops["redetect_timing"] = {
        "detection": detection_time,
        "extraction": extraction_time,
        "classification": classification_time,
        "spikes": spikes_time,
    }
    ops["redetect_model"] = str(model_path) if model_path else None
    ops["redetect_masks"] = str(masks_path) if masks_path else None

    np.save(ops_path, ops)

    total_time = time.time() - t0
    print(f"\nRe-detection complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  ROIs: {len(stat)}")
    if F is not None:
        print(f"  Traces: {F.shape}")
    print(f"  Output: {suite2p_path}")

    return ops_path
