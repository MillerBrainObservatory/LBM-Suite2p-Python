(processing-flow)=
# Processing Flow

High-level overview of all processing steps executed by `lsp.pipeline()`.

## Pre-Detection (All Modes)

| Step | Description | Disable |
|------|-------------|---------|
| Load/Convert Data | Read input files, convert to binary format | N/A |
| Rigid Registration | Correct XY frame shifts via phase correlation | `ops["do_registration"] = 0` |
| Non-rigid Registration | Correct local warping via block-wise shifts | `ops["nonrigid"] = False` |
| Temporal Binning | Average consecutive frames to reduce noise | `ops["nbinned"] = nframes` |
| Temporal High-Pass | Remove slow baseline drift from movie | `ops["high_pass"] = 0`* |

*\*The mbo fork allows disabling high_pass; standard Suite2p always applies it.*

## Detection: Anatomical Mode (Cellpose)

When `ops["anatomical_only"] > 0`:

| Step | Description | Disable |
|------|-------------|---------|
| PCA Denoise | Spatial denoising via block-wise PCA | `ops["denoise"] = 0` |
| Compute Anatomical Image | Generate image based on mode (1-4) | N/A (select via `ops["anatomical_only"]`) |
| Spatial High-Pass | Remove illumination gradients | `ops["spatial_hp_cp"] = 0` |
| Cellpose Segmentation | Run Cellpose model on processed image | N/A |

## Detection: Functional Mode

When `ops["anatomical_only"] = 0`:

| Step | Description | Disable | Notes |
|------|-------------|---------|-------|
| Spatial High-Pass | High-pass filter for activity detection | `ops["spatial_hp_detect"] = 0` | sparse only |
| Compute max_proj, Vcorr | Activity correlation images | N/A | sparse only |
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
| Apply Filters | Size/shape filters on detected ROIs | `filters=None` in `lsp.pipeline()` |

---

## See Also

```{seealso}
- {doc}`User Guide <user_guide>` - Complete processing examples and parameter tuning
- {doc}`Postprocessing <postprocessing>` - ΔF/F and filtering functions
- {doc}`Pipeline Comparison <pipeline_comparison>` - CaImAn, Suite2p, EXTRACT comparison
```
