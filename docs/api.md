# API

Functions, broken down by category.

## Run Suite2p

```{eval-rst}

.. currentmodule:: lbm_suite2p_python

.. autofunction:: pipeline
.. autofunction:: run_plane
.. autofunction:: run_volume
.. autofunction:: default_ops
.. autofunction:: add_processing_step

```
---

## Load Results

```{eval-rst}

.. currentmodule:: lbm_suite2p_python

.. autofunction:: load_ops
.. autofunction:: load_planar_results
.. autofunction:: get_results

```
---

## Cellpose / HITL Workflow

Functions for human-in-the-loop cell detection using Cellpose.

```{eval-rst}

.. currentmodule:: lbm_suite2p_python

.. autofunction:: train_cellpose
.. autofunction:: annotate
.. autofunction:: open_in_gui
.. autofunction:: prepare_training_data

```

### Format Conversion

```{eval-rst}

.. currentmodule:: lbm_suite2p_python

.. autofunction:: export_for_gui
.. autofunction:: import_from_gui
.. autofunction:: ensure_cellpose_format
.. autofunction:: detect_format
.. autofunction:: masks_to_stat
.. autofunction:: stat_to_masks

```
---

## ROI Filtering

```{eval-rst}

.. currentmodule:: lbm_suite2p_python.postprocessing

.. autofunction:: apply_filters
.. autofunction:: filter_by_diameter
.. autofunction:: filter_by_max_diameter
.. autofunction:: filter_by_area
.. autofunction:: filter_by_eccentricity

```
---

## Post-Processing

### ΔF/F Calculation

```{eval-rst}

.. currentmodule:: lbm_suite2p_python

.. autofunction:: dff_rolling_percentile
.. autofunction:: dff_median_filter
.. autofunction:: zscore_trace
.. autofunction:: baseline_percentile_dff
.. autofunction:: dff_shot_noise

```

### Quality Scoring

```{eval-rst}

.. currentmodule:: lbm_suite2p_python.postprocessing

.. autofunction:: compute_trace_quality_score
.. autofunction:: sort_traces_by_quality
.. autofunction:: compute_event_exceptionality

```

### Utilities

```{eval-rst}

.. currentmodule:: lbm_suite2p_python.postprocessing

.. autofunction:: load_traces
.. autofunction:: normalize_traces
.. autofunction:: ops_to_json

```
---

## Grid Search

```{eval-rst}

.. currentmodule:: lbm_suite2p_python

.. autofunction:: grid_search
.. autofunction:: collect_grid_results
.. autofunction:: save_grid_results
.. autofunction:: plot_grid_metrics

```
---

## Plotting

### Planar Visualization

```{eval-rst}

.. currentmodule:: lbm_suite2p_python

.. autofunction:: plot_traces
.. autofunction:: animate_traces
.. autofunction:: plot_projection
.. autofunction:: plot_zplane_figures
.. autofunction:: plot_plane_quality_metrics
.. autofunction:: plot_plane_diagnostics
.. autofunction:: plot_trace_analysis
.. autofunction:: plot_multiplane_masks
.. autofunction:: plot_mask_comparison
.. autofunction:: plot_regional_zoom
.. autofunction:: plot_filtered_cells
.. autofunction:: plot_diameter_histogram

```

### Volume Visualization

```{eval-rst}

.. currentmodule:: lbm_suite2p_python

.. autofunction:: plot_volume_diagnostics
.. autofunction:: plot_orthoslices
.. autofunction:: plot_3d_roi_map
.. autofunction:: plot_3d_rastermap_clusters
.. autofunction:: plot_volume_signal
.. autofunction:: plot_volume_neuron_counts
.. autofunction:: consolidate_volume

```
