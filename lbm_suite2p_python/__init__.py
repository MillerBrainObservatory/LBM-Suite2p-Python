from importlib.metadata import version, PackageNotFoundError

from lbm_suite2p_python.default_ops import default_ops
from lbm_suite2p_python.run_lsp import (
    pipeline,
    run_volume,
    run_plane,
    add_processing_step,
)

# Cellpose workflow exports
from lbm_suite2p_python.cellpose_workflow import (
    redetect,
    enhance_summary_image,
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
)

# Re-export key modules for easier access
from lbm_suite2p_python import (
    cellpose,
    conversion,
    utils,
    postprocessing,
    normcorre,
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

    # Cellpose / HITL Workflow
    "redetect",
    "enhance_summary_image",
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
    "normcorre",

    # Grid Search
    "grid_search",
    "collect_grid_results",
    "save_grid_results",
    "plot_grid_metrics",

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
]
