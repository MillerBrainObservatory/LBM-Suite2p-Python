from importlib.metadata import version, PackageNotFoundError

from lbm_suite2p_python.default_ops import default_ops
from lbm_suite2p_python.run_lsp import *
from lbm_suite2p_python.utils import *
from lbm_suite2p_python.volume import *
from lbm_suite2p_python.zplane import *
from lbm_suite2p_python.postprocessing import *

try:
    __version__ = version("lbm_suite2p_python")
except PackageNotFoundError:
    # fallback for editable installs
    __version__ = "0.0.0"

__all__ = [
    "pipeline",
    "run_volume",
    "run_plane",
    "grid_search",
    "consolidate_volume",
    "add_processing_step",
    "plot_traces",
    "plot_masks",
    "plot_rastermap",
    "plot_traces_noise",
    "plot_volume_signal",
    "plot_volume_neuron_counts",
    "plot_volume_diagnostics",
    "plot_orthoslices",
    "plot_3d_roi_map",
    "plot_projection",
    "plot_execution_time",
    "plot_noise_distribution",
    "plot_multiplane_masks",
    "plot_plane_quality_metrics",
    "plot_plane_diagnostics",
    "plot_trace_analysis",
    "plot_zplane_figures",
    "plot_mask_comparison",
    "create_volume_summary_table",
    "dff_rolling_percentile",
    "dff_median_filter",
    "dff_shot_noise",
    "compute_trace_quality_score",
    "sort_traces_by_quality",
    "load_ops",
    "load_planar_results",
    "default_ops",
    # Image processing utilities
    "normalize99",
    "apply_hp_filter",
    "random_colors_for_mask",
    "mask_overlay",
    "stat_to_mask",
    # Filtering utilities
    "filter_by_max_diameter",
    "filter_by_diameter",
    "filter_by_area",
    "filter_by_eccentricity",
    "apply_filters",
    # Visualization
    "plot_regional_zoom",
    "plot_filtered_cells",
    "plot_diameter_histogram",
]

# Re-export with public name
from lbm_suite2p_python.run_lsp import _add_processing_step as add_processing_step
