from importlib.metadata import version, PackageNotFoundError

from lbm_suite2p_python.default_ops import default_ops
from lbm_suite2p_python.run_lsp import (
    pipeline,
    run_volume,
    run_plane,
    add_processing_step,
    generate_plane_dirname,
)

from lbm_suite2p_python.cellpose import (
    train_cellpose,
    annotate,
    open_in_gui,
    prepare_training_data,
    masks_to_stat,
    stat_to_masks,
)
from lbm_suite2p_python.conversion import (
    export_for_gui,
    import_from_gui,
    get_results,
    ensure_cellpose_format,
    detect_format,
)

# Plotting exports
from lbm_suite2p_python.zplane import (
    plot_traces,
    animate_traces,
    plot_zplane_figures,
    plot_plane_quality_metrics,
    plot_plane_diagnostics,
    plot_trace_analysis,
    plot_multiplane_masks,
    plot_mask_comparison,
    plot_regional_zoom,
    plot_filtered_cells,
    plot_diameter_histogram,
    plot_projection,
    plot_accepted_rejected_overlay,
    plot_volume_accepted_rejected_overlay,
)

from lbm_suite2p_python.volume import (
    plot_volume_diagnostics,
    plot_orthoslices,
    plot_3d_roi_map,
    plot_3d_rastermap_clusters,
    plot_volume_signal,
    plot_volume_neuron_counts,
    consolidate_volume,
)

from lbm_suite2p_python.grid_search import (
    grid_search,
    collect_grid_results,
    save_grid_results,
    plot_grid_metrics,
    get_best_parameters,
    print_best_parameters,
    plot_grid_distributions,
    plot_grid_masks,
)

from lbm_suite2p_python.postprocessing import (
    load_ops,
    load_planar_results,
    dff_rolling_percentile,
    dff_shot_noise,
    compute_roi_stats,
)

# Re-export key modules for easier access
from lbm_suite2p_python import (
    cellpose,
    conversion,
    utils,
    postprocessing,
)

try:
    __version__ = version("lbm_suite2p_python")
except PackageNotFoundError:
    # fallback for editable installs
    __version__ = "0.0.0"

__all__ = [
    # Core API
    "pipeline",
    "run_volume",
    "run_plane",
    "default_ops",
    "add_processing_step",
    "generate_plane_dirname",

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
    "plot_projection",
    "plot_volume_signal",
    "plot_volume_neuron_counts",
    "consolidate_volume",
    "plot_accepted_rejected_overlay",
    "plot_volume_accepted_rejected_overlay",
]
