"""
Interactive ImGui widget for viewing plane diagnostics.

Usage:
    python diagnostics_widget.py [plane_dir]

If no plane_dir is provided, uses the default example path.
"""

from pathlib import Path
import sys
import numpy as np

from imgui_bundle import imgui, implot, immapp, hello_imgui
from imgui_bundle import ImVec2, ImVec4

import lbm_suite2p_python as lsp
from lbm_suite2p_python.postprocessing import load_ops, load_planar_results


class DiagnosticsWidget:
    """Interactive diagnostics viewer using ImGui with dockable panels."""

    def __init__(self, plane_dir: str | Path):
        self.plane_dir = Path(plane_dir)
        self.loaded = False
        self.error_msg = ""

        # Data arrays
        self.stat = None
        self.iscell = None
        self.F = None
        self.Fneu = None
        self.ops = None

        # Computed metrics
        self.snr = None
        self.npix = None
        self.compactness = None
        self.skewness = None
        self.activity = None
        self.shot_noise = None
        self.dff = None

        # Masks
        self.accepted = None
        self.n_accepted = 0
        self.n_rejected = 0
        self.n_total = 0

        # UI state
        self.selected_roi = 0
        self.show_accepted_only = True

        # Load data
        self._load_data()

    def _load_data(self):
        """Load and compute all diagnostic data."""
        try:
            if not self.plane_dir.exists():
                self.error_msg = f"Directory not found: {self.plane_dir}"
                return

            # Load results
            res = load_planar_results(self.plane_dir)
            self.ops = load_ops(self.plane_dir / "ops.npy")

            self.stat = res["stat"]
            self.iscell = res["iscell"]
            self.F = res["F"]
            self.Fneu = res["Fneu"]

            self.n_total = len(self.stat)
            if self.n_total == 0:
                self.error_msg = "No ROIs detected"
                return

            # Handle iscell format
            if self.iscell.ndim == 2:
                self.accepted = self.iscell[:, 0].astype(bool)
            else:
                self.accepted = self.iscell.astype(bool)

            self.n_accepted = int(self.accepted.sum())
            self.n_rejected = int((~self.accepted).sum())

            # Compute ΔF/F
            F_corr = self.F - 0.7 * self.Fneu
            baseline = np.percentile(F_corr, 20, axis=1, keepdims=True)
            baseline = np.maximum(baseline, 1e-6)
            self.dff = (F_corr - baseline) / baseline

            # SNR calculation
            signal = np.std(self.dff, axis=1)
            noise = np.median(np.abs(np.diff(self.dff, axis=1)), axis=1) / 0.6745
            self.snr = signal / (noise + 1e-6)

            # Extract ROI properties
            self.npix = np.array([s.get("npix", 0) for s in self.stat])
            self.compactness = np.array([s.get("compact", np.nan) for s in self.stat])
            self.skewness = np.array([s.get("skew", np.nan) for s in self.stat])

            # Activity metric
            self.activity = np.sum(self.dff > 2, axis=1)

            # Shot noise
            fs = self.ops.get("fs", 30.0)
            frame_diffs = np.abs(np.diff(self.dff, axis=1))
            self.shot_noise = np.median(frame_diffs, axis=1) / np.sqrt(fs) * 100

            # Mean image for ROI visualization
            self.mean_img = self.ops.get("meanImgE", self.ops.get("meanImg"))
            if self.mean_img is None:
                self.mean_img = np.zeros((self.ops.get("Ly", 512), self.ops.get("Lx", 512)))

            # Normalize mean image for display
            vmin, vmax = np.percentile(self.mean_img, [1, 99])
            self.mean_img_norm = np.clip((self.mean_img - vmin) / (vmax - vmin + 1e-6), 0, 1)

            self.loaded = True

        except Exception as e:
            self.error_msg = f"Error loading data: {str(e)}"

    def _get_filtered_indices(self):
        """Get ROI indices based on current filter."""
        if self.show_accepted_only:
            return np.where(self.accepted)[0]
        else:
            return np.arange(self.n_total)

    def _get_current_roi_idx(self):
        """Get the actual ROI index for current selection."""
        indices = self._get_filtered_indices()
        if len(indices) == 0:
            return None
        self.selected_roi = min(self.selected_roi, len(indices) - 1)
        return indices[self.selected_roi]

    def _get_roi_bounds(self, roi_idx, pad=20):
        """Get bounding box for an ROI with padding."""
        roi_stat = self.stat[roi_idx]
        ypix = roi_stat.get("ypix", np.array([0]))
        xpix = roi_stat.get("xpix", np.array([0]))

        if len(ypix) == 0 or len(xpix) == 0:
            return 0, 50, 0, 50

        y_min = max(0, int(ypix.min()) - pad)
        y_max = min(self.mean_img.shape[0], int(ypix.max()) + pad)
        x_min = max(0, int(xpix.min()) - pad)
        x_max = min(self.mean_img.shape[1], int(xpix.max()) + pad)

        return y_min, y_max, x_min, x_max

    def render_controls(self):
        """Render ROI navigation controls."""
        if not self.loaded:
            imgui.text_colored(ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {self.error_msg}")
            return

        # Summary header
        imgui.text(f"Plane: {self.plane_dir.name}")
        imgui.same_line()
        imgui.text_colored(ImVec4(0.2, 0.8, 0.4, 1.0), f"Accepted: {self.n_accepted}")
        imgui.same_line()
        imgui.text_colored(ImVec4(0.9, 0.4, 0.3, 1.0), f"Rejected: {self.n_rejected}")

        imgui.separator()

        # Filter toggle
        _, self.show_accepted_only = imgui.checkbox("Show Accepted Only", self.show_accepted_only)

        indices = self._get_filtered_indices()
        n_filtered = len(indices)

        if n_filtered == 0:
            imgui.text("No ROIs to display")
            return

        # ROI selector
        imgui.text(f"ROI {self.selected_roi + 1} / {n_filtered}")
        imgui.same_line()

        if imgui.button("<"):
            self.selected_roi = max(0, self.selected_roi - 1)
        imgui.same_line()
        if imgui.button(">"):
            self.selected_roi = min(n_filtered - 1, self.selected_roi + 1)

        imgui.same_line()
        imgui.set_next_item_width(150)
        changed, new_val = imgui.slider_int("##roi_slider", self.selected_roi, 0, n_filtered - 1)
        if changed:
            self.selected_roi = new_val

        # Current ROI info
        roi_idx = self._get_current_roi_idx()
        if roi_idx is not None:
            imgui.separator()
            imgui.text_colored(
                ImVec4(0.18, 0.8, 0.44, 1.0) if self.accepted[roi_idx] else ImVec4(0.9, 0.3, 0.24, 1.0),
                "ACCEPTED" if self.accepted[roi_idx] else "REJECTED"
            )
            imgui.text(f"Index: {roi_idx} | Size: {self.npix[roi_idx]} px | SNR: {self.snr[roi_idx]:.2f}")
            imgui.text(f"Compact: {self.compactness[roi_idx]:.3f} | Skew: {self.skewness[roi_idx]:.3f} | Noise: {self.shot_noise[roi_idx]:.2f}")

    def render_full_image(self):
        """Render the full mean image with ROI location box."""
        if not self.loaded:
            return

        roi_idx = self._get_current_roi_idx()
        if roi_idx is None:
            return

        avail = imgui.get_content_region_avail()

        if implot.begin_plot("Mean Image (Full FOV)", ImVec2(avail.x, avail.y),
                             implot.Flags_.equal | implot.Flags_.no_legend):
            implot.setup_axes("X", "Y", implot.AxisFlags_.no_decorations,
                             implot.AxisFlags_.no_decorations | implot.AxisFlags_.invert)

            # Draw full mean image
            implot.plot_heatmap("##full_img", self.mean_img_norm.astype(np.float64),
                               scale_min=0.0, scale_max=1.0,
                               bounds_min=implot.Point(0, 0),
                               bounds_max=implot.Point(self.mean_img.shape[1], self.mean_img.shape[0]))

            # Draw red box around current ROI region
            y_min, y_max, x_min, x_max = self._get_roi_bounds(roi_idx)

            # Box corners
            box_x = np.array([x_min, x_max, x_max, x_min, x_min], dtype=np.float64)
            box_y = np.array([y_min, y_min, y_max, y_max, y_min], dtype=np.float64)

            implot.set_next_line_style(ImVec4(1.0, 0.2, 0.2, 1.0), 2.0)
            implot.plot_line("ROI Region", box_x, box_y)

            # Mark ROI centroid
            roi_stat = self.stat[roi_idx]
            ypix = roi_stat.get("ypix", np.array([0]))
            xpix = roi_stat.get("xpix", np.array([0]))
            if len(ypix) > 0:
                cy, cx = np.mean(ypix), np.mean(xpix)
                implot.set_next_marker_style(implot.Marker_.cross, 8, ImVec4(1.0, 0.2, 0.2, 1.0), 2.0)
                implot.plot_scatter("Center", np.array([cx]), np.array([cy]))

            implot.end_plot()

    def render_zoomed_roi(self):
        """Render zoomed view of current ROI."""
        if not self.loaded:
            return

        roi_idx = self._get_current_roi_idx()
        if roi_idx is None:
            return

        y_min, y_max, x_min, x_max = self._get_roi_bounds(roi_idx)
        zoom_img = self.mean_img_norm[y_min:y_max, x_min:x_max]

        if zoom_img.size == 0:
            imgui.text("No image data")
            return

        avail = imgui.get_content_region_avail()

        if implot.begin_plot("ROI Zoom", ImVec2(avail.x, avail.y),
                             implot.Flags_.equal | implot.Flags_.no_legend):
            implot.setup_axes("", "", implot.AxisFlags_.no_decorations,
                             implot.AxisFlags_.no_decorations | implot.AxisFlags_.invert)

            # Draw zoomed image
            implot.plot_heatmap("##zoom_img", zoom_img.astype(np.float64),
                               scale_min=0.0, scale_max=1.0,
                               bounds_min=implot.Point(0, 0),
                               bounds_max=implot.Point(zoom_img.shape[1], zoom_img.shape[0]))

            # Draw ROI pixels
            roi_stat = self.stat[roi_idx]
            ypix = roi_stat.get("ypix", np.array([]))
            xpix = roi_stat.get("xpix", np.array([]))

            if len(ypix) > 0:
                local_y = ypix - y_min
                local_x = xpix - x_min

                roi_color = ImVec4(0.18, 0.8, 0.44, 0.9) if self.accepted[roi_idx] else ImVec4(0.9, 0.3, 0.24, 0.9)
                implot.set_next_marker_style(implot.Marker_.square, 3, roi_color)
                implot.plot_scatter("ROI Pixels",
                                   local_x.astype(np.float64) + 0.5,
                                   local_y.astype(np.float64) + 0.5)

            implot.end_plot()

    def render_trace(self):
        """Render trace for current ROI."""
        if not self.loaded:
            return

        roi_idx = self._get_current_roi_idx()
        if roi_idx is None:
            return

        avail = imgui.get_content_region_avail()

        if implot.begin_plot(f"Trace (ROI {roi_idx})", ImVec2(avail.x, avail.y)):
            implot.setup_axes("Frame", "dF/F")

            trace = self.dff[roi_idx]
            frames = np.arange(len(trace), dtype=np.float64)

            color = ImVec4(0.18, 0.8, 0.44, 1.0) if self.accepted[roi_idx] else ImVec4(0.9, 0.3, 0.24, 1.0)
            implot.set_next_line_style(color)
            implot.plot_line("dF/F", frames, trace.astype(np.float64))

            implot.end_plot()

    def render_scatter_snr_vs_compactness(self):
        """Render SNR vs Compactness scatter with current ROI highlighted."""
        if not self.loaded or self.n_accepted == 0:
            return

        roi_idx = self._get_current_roi_idx()
        avail = imgui.get_content_region_avail()

        if implot.begin_plot("SNR vs Compactness", ImVec2(avail.x, avail.y)):
            implot.setup_axes("Compactness", "SNR")

            # All accepted ROIs
            valid = self.accepted & ~np.isnan(self.compactness)
            if valid.sum() > 0:
                implot.set_next_marker_style(implot.Marker_.circle, 3, ImVec4(0.4, 0.4, 0.4, 0.5))
                implot.plot_scatter("All",
                                   self.compactness[valid].astype(np.float64),
                                   self.snr[valid].astype(np.float64))

            # Highlight current ROI
            if roi_idx is not None and not np.isnan(self.compactness[roi_idx]):
                color = ImVec4(1.0, 0.8, 0.0, 1.0)  # Yellow highlight
                implot.set_next_marker_style(implot.Marker_.diamond, 10, color, 2.0, color)
                implot.plot_scatter("Current",
                                   np.array([self.compactness[roi_idx]], dtype=np.float64),
                                   np.array([self.snr[roi_idx]], dtype=np.float64))

            implot.end_plot()

    def render_scatter_snr_vs_skewness(self):
        """Render SNR vs Skewness scatter with current ROI highlighted."""
        if not self.loaded or self.n_accepted == 0:
            return

        roi_idx = self._get_current_roi_idx()
        avail = imgui.get_content_region_avail()

        if implot.begin_plot("SNR vs Skewness", ImVec2(avail.x, avail.y)):
            implot.setup_axes("Skewness", "SNR")

            # All accepted ROIs
            valid = self.accepted & ~np.isnan(self.skewness)
            if valid.sum() > 0:
                implot.set_next_marker_style(implot.Marker_.circle, 3, ImVec4(0.4, 0.4, 0.4, 0.5))
                implot.plot_scatter("All",
                                   self.skewness[valid].astype(np.float64),
                                   self.snr[valid].astype(np.float64))

            # Highlight current ROI
            if roi_idx is not None and not np.isnan(self.skewness[roi_idx]):
                color = ImVec4(1.0, 0.8, 0.0, 1.0)
                implot.set_next_marker_style(implot.Marker_.diamond, 10, color, 2.0, color)
                implot.plot_scatter("Current",
                                   np.array([self.skewness[roi_idx]], dtype=np.float64),
                                   np.array([self.snr[roi_idx]], dtype=np.float64))

            implot.end_plot()

    def render_scatter_snr_vs_activity(self):
        """Render SNR vs Activity scatter with current ROI highlighted."""
        if not self.loaded or self.n_accepted == 0:
            return

        roi_idx = self._get_current_roi_idx()
        avail = imgui.get_content_region_avail()

        if implot.begin_plot("SNR vs Activity", ImVec2(avail.x, avail.y)):
            implot.setup_axes("Active Frames", "SNR")

            # All accepted ROIs
            if self.n_accepted > 0:
                implot.set_next_marker_style(implot.Marker_.circle, 3, ImVec4(0.4, 0.4, 0.4, 0.5))
                implot.plot_scatter("All",
                                   self.activity[self.accepted].astype(np.float64),
                                   self.snr[self.accepted].astype(np.float64))

            # Highlight current ROI
            if roi_idx is not None:
                color = ImVec4(1.0, 0.8, 0.0, 1.0)
                implot.set_next_marker_style(implot.Marker_.diamond, 10, color, 2.0, color)
                implot.plot_scatter("Current",
                                   np.array([self.activity[roi_idx]], dtype=np.float64),
                                   np.array([self.snr[roi_idx]], dtype=np.float64))

            implot.end_plot()

    def render_scatter_size_vs_snr(self):
        """Render Size vs SNR scatter with current ROI highlighted."""
        if not self.loaded or self.n_accepted == 0:
            return

        roi_idx = self._get_current_roi_idx()
        avail = imgui.get_content_region_avail()

        if implot.begin_plot("Size vs SNR", ImVec2(avail.x, avail.y)):
            implot.setup_axes("Size (pixels)", "SNR")

            # All accepted ROIs
            if self.n_accepted > 0:
                implot.set_next_marker_style(implot.Marker_.circle, 3, ImVec4(0.4, 0.4, 0.4, 0.5))
                implot.plot_scatter("All",
                                   self.npix[self.accepted].astype(np.float64),
                                   self.snr[self.accepted].astype(np.float64))

            # Highlight current ROI
            if roi_idx is not None:
                color = ImVec4(1.0, 0.8, 0.0, 1.0)
                implot.set_next_marker_style(implot.Marker_.diamond, 10, color, 2.0, color)
                implot.plot_scatter("Current",
                                   np.array([self.npix[roi_idx]], dtype=np.float64),
                                   np.array([self.snr[roi_idx]], dtype=np.float64))

            implot.end_plot()


def main():
    # Default path or command line argument
    if len(sys.argv) > 1:
        plane_dir = sys.argv[1]
    else:
        plane_dir = r"D:\example_extraction\suite2p\plane01_stitched"

    widget = DiagnosticsWidget(plane_dir)

    # Configure ImGui runner with docking
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = f"Plane Diagnostics - {Path(plane_dir).name}"
    runner_params.app_window_params.window_geometry.size = (1600, 1000)

    # Enable docking
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )
    runner_params.imgui_window_params.enable_viewports = True

    def gui_callback():
        # Controls window (not dockable, stays at top)
        imgui.set_next_window_pos(ImVec2(10, 10), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(ImVec2(400, 180), imgui.Cond_.first_use_ever)
        imgui.begin("Controls", flags=imgui.WindowFlags_.no_collapse)
        widget.render_controls()
        imgui.end()

        # Full FOV image
        imgui.set_next_window_pos(ImVec2(10, 200), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(ImVec2(500, 400), imgui.Cond_.first_use_ever)
        imgui.begin("Full FOV", flags=imgui.WindowFlags_.no_collapse)
        widget.render_full_image()
        imgui.end()

        # Zoomed ROI
        imgui.set_next_window_pos(ImVec2(520, 200), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(ImVec2(300, 300), imgui.Cond_.first_use_ever)
        imgui.begin("ROI Zoom", flags=imgui.WindowFlags_.no_collapse)
        widget.render_zoomed_roi()
        imgui.end()

        # Trace
        imgui.set_next_window_pos(ImVec2(420, 10), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(ImVec2(800, 180), imgui.Cond_.first_use_ever)
        imgui.begin("Trace", flags=imgui.WindowFlags_.no_collapse)
        widget.render_trace()
        imgui.end()

        # Scatter plots - arrange in a row at bottom
        imgui.set_next_window_pos(ImVec2(10, 610), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(ImVec2(390, 380), imgui.Cond_.first_use_ever)
        imgui.begin("SNR vs Compactness", flags=imgui.WindowFlags_.no_collapse)
        widget.render_scatter_snr_vs_compactness()
        imgui.end()

        imgui.set_next_window_pos(ImVec2(410, 610), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(ImVec2(390, 380), imgui.Cond_.first_use_ever)
        imgui.begin("SNR vs Skewness", flags=imgui.WindowFlags_.no_collapse)
        widget.render_scatter_snr_vs_skewness()
        imgui.end()

        imgui.set_next_window_pos(ImVec2(810, 610), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(ImVec2(390, 380), imgui.Cond_.first_use_ever)
        imgui.begin("SNR vs Activity", flags=imgui.WindowFlags_.no_collapse)
        widget.render_scatter_snr_vs_activity()
        imgui.end()

        imgui.set_next_window_pos(ImVec2(1210, 610), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(ImVec2(380, 380), imgui.Cond_.first_use_ever)
        imgui.begin("Size vs SNR", flags=imgui.WindowFlags_.no_collapse)
        widget.render_scatter_size_vs_snr()
        imgui.end()

    runner_params.callbacks.show_gui = gui_callback

    # Initialize implot context
    def post_init():
        implot.create_context()

    def before_exit():
        implot.destroy_context()

    runner_params.callbacks.post_init = post_init
    runner_params.callbacks.before_exit = before_exit

    immapp.run(runner_params)


if __name__ == "__main__":
    main()
