import importlib
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lbm_suite2p_python")
except PackageNotFoundError:
    # fallback for editable installs
    __version__ = "0.0.0"

# Lazy attribute -> submodule map (PEP 562). Importing the package no longer
# pulls in the whole pipeline stack (suite2p, cv2, matplotlib, ...); each name
# is imported from its submodule on first access. Keeps `lsp --help` / version
# fast and importable without the heavy scientific deps installed.
_LAZY_ATTRS = {
    # default_ops
    "default_ops": "default_ops",
    # run_lsp
    "pipeline": "run_lsp",
    "run_volume": "run_lsp",
    "run_plane": "run_lsp",
    "run_plane_stream": "run_lsp",
    "add_processing_step": "run_lsp",
    "generate_plane_dirname": "run_lsp",
    "compute_enhanced_mean_image": "run_lsp",
    # streaming
    "StreamingBinaryFile": "streaming",
    "apply_shifts": "streaming",
    # cellpose
    "train_cellpose": "cellpose",
    "annotate": "cellpose",
    "open_in_gui": "cellpose",
    "prepare_training_data": "cellpose",
    "masks_to_stat": "cellpose",
    "stat_to_masks": "cellpose",
    # conversion
    "export_for_gui": "conversion",
    "import_from_gui": "conversion",
    "get_results": "conversion",
    "ensure_cellpose_format": "conversion",
    "detect_format": "conversion",
    # zplane plotting
    "plot_traces": "zplane",
    "animate_traces": "zplane",
    "plot_zplane_figures": "zplane",
    "plot_plane_quality_metrics": "zplane",
    "plot_plane_diagnostics": "zplane",
    "plot_trace_analysis": "zplane",
    "plot_multiplane_masks": "zplane",
    "plot_mask_comparison": "zplane",
    "plot_regional_zoom": "zplane",
    "plot_filtered_cells": "zplane",
    "plot_diameter_histogram": "zplane",
    "plot_projection": "zplane",
    "plot_accepted_rejected_overlay": "zplane",
    "plot_volume_accepted_rejected_overlay": "zplane",
    # volume plotting
    "plot_volume_diagnostics": "volume",
    "plot_orthoslices": "volume",
    "plot_3d_roi_map": "volume",
    "plot_3d_rastermap_clusters": "volume",
    "plot_volume_trace_figures": "volume",
    "plot_volume_signal": "volume",
    "plot_volume_neuron_counts": "volume",
    "consolidate_volume": "volume",
    # grid search
    "grid_search": "grid_search",
    "collect_grid_results": "grid_search",
    "save_grid_results": "grid_search",
    "plot_grid_metrics": "grid_search",
    "get_best_parameters": "grid_search",
    "print_best_parameters": "grid_search",
    "plot_grid_distributions": "grid_search",
    "plot_grid_masks": "grid_search",
    # postprocessing
    "load_ops": "postprocessing",
    "load_planar_results": "postprocessing",
    "dff_rolling_percentile": "postprocessing",
    "dff_median_filter": "postprocessing",
    "zscore_trace": "postprocessing",
    "baseline_percentile_dff": "postprocessing",
    "dff_shot_noise": "postprocessing",
    "compute_roi_stats": "postprocessing",
}

# Re-exported submodules for easier access (lsp.cellpose, lsp.utils, ...).
_LAZY_SUBMODULES = frozenset({"cellpose", "conversion", "utils", "postprocessing"})


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    module_name = _LAZY_ATTRS.get(name)
    if module_name is not None:
        mod = importlib.import_module(f"{__name__}.{module_name}")
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    # Core API
    "pipeline",
    "run_volume",
    "run_plane",
    "run_plane_stream",
    "StreamingBinaryFile",
    "apply_shifts",
    "default_ops",
    "add_processing_step",
    "generate_plane_dirname",
    "compute_enhanced_mean_image",

    # Cellpose / HITL Workflow
    "train_cellpose",
    "annotate",
    "open_in_gui",
    "prepare_training_data",
    "masks_to_stat",
    "stat_to_masks",
    "export_for_gui",
    "import_from_gui",
    "get_results",
    "ensure_cellpose_format",
    "detect_format",

    # Modules
    "cellpose",
    "conversion",
    "utils",
    "postprocessing",

    # Grid Search
    "grid_search",
    "collect_grid_results",
    "save_grid_results",
    "plot_grid_metrics",
    "get_best_parameters",
    "print_best_parameters",
    "plot_grid_distributions",
    "plot_grid_masks",

    # Postprocessing
    "load_ops",
    "load_planar_results",
    "dff_rolling_percentile",
    "dff_median_filter",
    "zscore_trace",
    "baseline_percentile_dff",
    "dff_shot_noise",
    "compute_roi_stats",

    # Plotting (Z-Plane)
    "plot_traces",
    "animate_traces",
    "plot_zplane_figures",
    "plot_plane_quality_metrics",
    "plot_plane_diagnostics",
    "plot_trace_analysis",
    "plot_multiplane_masks",
    "plot_mask_comparison",
    "plot_regional_zoom",
    "plot_filtered_cells",
    "plot_diameter_histogram",

    # Plotting (Volume)
    "plot_volume_diagnostics",
    "plot_orthoslices",
    "plot_3d_roi_map",
    "plot_3d_rastermap_clusters",
    "plot_volume_trace_figures",
    "plot_projection",
    "plot_volume_signal",
    "plot_volume_neuron_counts",
    "consolidate_volume",
    "plot_accepted_rejected_overlay",
    "plot_volume_accepted_rejected_overlay",
]
