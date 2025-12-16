# Image Gallery

(image_gallery)=

Visual reference for LBM-Suite2p-Python outputs and documentation.

## Pipeline Outputs

```{figure} ./_images/outputs/02_max_projection.png
:alt: Maximum projection
:name: fig-max-proj
:width: 80%

Maximum intensity projection from registered movie.
```

```{figure} ./_images/outputs/04_mean_enhanced.png
:alt: Enhanced mean image
:name: fig-mean-enhanced
:width: 80%

Enhanced mean image with median high-pass filter applied.
```

```{figure} ./_images/outputs/04_mean_enhanced_segmentation.png
:alt: Segmentation overlay
:name: fig-segmentation
:width: 80%

ROI masks overlaid on enhanced mean image.
```

```{figure} ./_images/outputs/05_quality_diagnostics.png
:alt: Quality diagnostics
:name: fig-quality
:width: 100%

ROI quality metrics: size, SNR, compactness distributions.
```

```{figure} ./_images/outputs/07_traces_raw.png
:alt: Raw traces
:name: fig-traces-raw
:width: 100%

Raw fluorescence traces for top neurons by quality score.
```

```{figure} ./_images/outputs/08_traces_dff.png
:alt: ΔF/F traces
:name: fig-traces-dff
:width: 100%

ΔF/F traces computed with rolling percentile baseline.
```

## ΔF/F Analysis

```{figure} ./_images/dff/dff_methods_comparison.png
:alt: ΔF/F methods comparison
:name: gallery-dff-methods
:width: 100%

Comparison of baseline estimation methods.
```

```{figure} ./_images/dff/dff_window_size_effect.png
:alt: Window size effect
:name: gallery-window-size
:width: 100%

Effect of window size on baseline estimation.
```

```{figure} ./_images/dff/shot_noise_analysis.png
:alt: Shot noise analysis
:name: gallery-shot-noise
:width: 100%

Shot noise estimation across ROIs.
```

```{figure} ./_images/dff/quality_score_breakdown.png
:alt: Quality score breakdown
:name: gallery-quality-score
:width: 100%

Components of trace quality scoring.
```

## Projection Types

```{figure} ./_images/projections/01_raw_projections.png
:alt: Suite2p output images
:name: fig-raw-projections
:width: 100%

Suite2p output images: mean, enhanced mean, max projection, reference, and correlation.
```

```{figure} ./_images/projections/02_anatomical_modes.png
:alt: Anatomical detection modes
:name: fig-anatomical-modes
:width: 100%

Cellpose input images for different `anatomical_only` settings (1-4).
```

```{figure} ./_images/projections/03_spatial_hp_filter.png
:alt: Spatial high-pass filter effect
:name: gallery-spatial-hp
:width: 100%

Effect of `spatial_hp_cp` parameter on cell boundary enhancement.
```

```{figure} ./_images/projections/04_cellpose_final_input.png
:alt: Final cellpose input comparison
:name: fig-cellpose-input
:width: 100%

What Cellpose receives under different configuration settings.
```

```{figure} ./_images/projections/05_hp_filter_zoom.png
:alt: High-pass filter detail
:name: fig-hp-zoom
:width: 100%

Zoomed comparison of high-pass filter effect on cell boundaries.
```

## Parameter Effects

```{figure} ./_images/parameters/default_parameters.png
:alt: Default parameters
:name: gallery-default-params
:width: 80%

Segmentation with default parameters.
```

```{figure} ./_images/parameters/default_params_thr.png
:alt: Threshold scaling effect
:name: gallery-threshold
:width: 100%

Effect of threshold_scaling on detection.
```

```{figure} ./_images/parameters/default_parameters_tau.png
:alt: Tau effect
:name: gallery-tau
:width: 100%

Effect of tau on temporal binning and detection.
```

```{figure} ./_images/parameters/default_parameters_max_overlap.png
:alt: Max overlap effect
:name: gallery-max-overlap
:width: 100%

Effect of max_overlap on ROI acceptance.
```

## Diagrams

```{figure} ./_images/diagrams/ex_offset.svg
:alt: Offset correction
:name: fig-offset

Line offset correction diagram.
```

```{figure} ./_images/diagrams/ex_retile.svg
:alt: Retiling
:name: fig-retile

ROI retiling diagram.
```

```{figure} ./_images/diagrams/ex_diagram.svg
:alt: Pipeline diagram
:name: fig-pipeline

Pipeline flow diagram.
```

```{figure} ./_images/diagrams/ex_deinterleave.svg
:alt: Deinterleaving
:name: fig-deinterleave

Z-plane deinterleaving diagram.
```
