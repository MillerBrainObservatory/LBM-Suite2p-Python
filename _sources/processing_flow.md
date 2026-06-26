(processing-flow)=
# Processing Flow

High-level overview of all processing steps executed by `lsp.pipeline()`.

## Registration

| Step | Description | Outputs | Disable |
|------|-------------|---------|---------|
| Rigid Registration | Correct XY frame shifts via phase correlation | `data.bin` (registered), `ops["yoff"]`, `ops["xoff"]` | `ops["do_registration"] = 0` |
| Non-rigid Registration | Correct local warping via block-wise shifts | `data.bin` (registered) | `ops["nonrigid"] = False` |
| Compute meanImg | Average all registered frames | `ops["meanImg"]` | N/A |
| Compute meanImgE | Enhanced mean via spatial high-pass | `ops["meanImgE"]` | N/A |

## Detection Preprocessing

These steps are applied to the registered movie before ROI detection:

| Step | Description | Affects | Disable |
|------|-------------|---------|---------|
| Temporal Binning | Average consecutive frames to reduce noise | Detection movie, `ops["max_proj"]` | `ops["nbinned"] = nframes` |
| Temporal High-Pass | Remove slow baseline drift | Detection movie, `ops["max_proj"]` | `ops["high_pass"] = 0`* |

*\*The mbo fork allows disabling high_pass; standard Suite2p forces a value.*

### What each projection image contains

| Image | Source | Binned? | Temporally HP filtered? |
|-------|--------|---------|------------------------|
| `meanImg` | Average of registered frames | No | No |
| `meanImgE` | Spatial HP of meanImg | No | No |
| `max_proj` | Max of detection movie | Yes | Yes |

## Detection: Anatomical Mode (Cellpose)

Cellpose segments one static image, chosen by `cellpose_settings.img`. Suite2p 1.1.0
accepts three values:

| `img` | Image | Notes |
|-------|-------|-------|
| `"max_proj / meanImg"` | `log(max_proj / meanImg)` | **default**; log ratio emphasizes active regions |
| `"meanImg"` | mean image | raw mean intensity |
| `"max_proj"` | max projection | binned + temporally high-pass filtered |

The enhanced mean image (`meanImgE`) is no longer a cellpose input — the old
`anatomical_only=3` was removed upstream.

### Selecting the image

Recommended — enable cellpose with `algorithm="cellpose"` and pick `img`:

```python
ops = {"algorithm": "cellpose", "img": "max_proj / meanImg"}  # default
ops = {"algorithm": "cellpose", "img": "meanImg"}
ops = {"algorithm": "cellpose", "img": "max_proj"}
```

Legacy — the integer `anatomical_only` still works and maps to `img`:

```python
ops = {"anatomical_only": 1}   # img = "max_proj / meanImg" (default)
ops = {"anatomical_only": 2}   # img = "meanImg"
ops = {"anatomical_only": 4}   # img = "max_proj"
```

`anatomical_only=0` (or `algorithm="sparsery"`/`"sourcery"`) uses functional detection.
`anatomical_only=3` warns and falls back to 1.

| Step | Description | Disable |
|------|-------------|---------|
| PCA Denoise | Spatial denoising via block-wise PCA | `ops["denoise"] = 0` |
| Spatial High-Pass | Sharpen the chosen image before cellpose | `ops["spatial_hp_cp"] = 0` |
| Cellpose Segmentation | Run Cellpose model on the image | N/A |

### Spatial high-pass: `spatial_hp_cp` (upstream `cellpose_settings.highpass_spatial`)

Integer, default 0 (off). When > 0, the chosen `img` is normalized to [0, 1] and a
Gaussian-blurred copy (sigma = `diameter × spatial_hp_cp`) is subtracted twice, removing
illumination gradients and sharpening cell boundaries. It applies to whichever `img` you
select and does **not** modify the stored `ops["meanImg"]` or `ops["max_proj"]`.

## Detection: Functional Mode

When `ops["anatomical_only"] = 0`:

| Step | Description | Disable | Notes |
|------|-------------|---------|-------|
| Spatial High-Pass | High-pass filter for activity detection | `ops["spatial_hp_detect"] = 0` | sparse only |
| Compute Vcorr | Activity correlation images | N/A | sparse only |
| Seed Detection | Find candidate ROI centers from peaks | N/A | sparse only |
| ROI Growing | Expand seeds based on correlation | N/A | sparse only |
| SVD Decomposition | Reduce dimensionality for source detection | N/A | sourcery only (`sparse_mode=False`) |
| Source Detection | Find ROIs via iterative NMF-like method | N/A | sourcery only |
| ROI Overlap Removal | Merge/remove overlapping ROIs | `ops["max_overlap"] = 1.0` | all functional |

## Post-Detection (All Modes)

| Step | Description | Disable |
|------|-------------|---------|
| Extract Fluorescence (F) | Compute raw fluorescence from ROI pixels | N/A |
| Compute Neuropil (Fneu) | Estimate neuropil signal from surround | `ops["neuropil_extract"] = False` |
| Classifier | Apply ROI classifier (cell vs not-cell) | `ops["classifier_path"] = ""` |
| Spike Deconvolution | Infer spike times via OASIS | `ops["spikedetect"] = False` |
| Apply Filters | Size/shape filters on detected ROIs | `cell_filters=[]` in `lsp.pipeline()` |

## lsp.pipeline() Additional Steps

These are applied by `lsp.pipeline()` after suite2p completes:

| Step | Description | Disable |
|------|-------------|---------|
| Accept All Cells | Mark all suite2p-rejected ROIs as accepted | `accept_all_cells=False` (default) |
| Cell Filters | Size/shape filters (default 4-35 µm diameter) | `cell_filters=[]` |
| Compute dF/F | Rolling percentile baseline correction | N/A |
| Generate Figures | Diagnostic plots for each plane | N/A |

```{seealso}
- {doc}`User Guide <user_guide>` - Complete processing examples and parameter tuning
- {doc}`Postprocessing <postprocessing>` - ΔF/F and filtering functions
- {doc}`Pipeline Comparison <pipeline_comparison>` - CaImAn, Suite2p, EXTRACT comparison
```
