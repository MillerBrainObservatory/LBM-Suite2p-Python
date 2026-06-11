from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import math

import matplotlib.offsetbox
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import VPacker, HPacker, DrawingArea
import matplotlib.gridspec as gridspec

from scipy.ndimage import distance_transform_edt

from mbo_utilities.metadata import get_param

from lbm_suite2p_python.postprocessing import (
    load_ops,
    load_planar_results,
    dff_rolling_percentile,
    zscore_trace,
    baseline_percentile_dff,
    dff_shot_noise,
    compute_trace_quality_score,
)
from lbm_suite2p_python.utils import (
    _resize_masks_fit_crop,
    bin1d,
)


def infer_units(f: np.ndarray) -> str:
    """
    Infer calcium imaging signal type from array values:
    - 'raw': values in hundreds or thousands
    - 'dff': unitless ΔF/F₀, typically ~0–1
    - 'dffp': ΔF/F₀ in percent, typically ~10–100
    - 'zscore': centered on 0 with negatives, unit-scale std

    Returns one of: 'raw', 'dff', 'dffp', 'zscore', 'unknown'
    """
    f = np.asarray(f)
    if np.issubdtype(f.dtype, np.integer):
        return "raw"

    p1, p50, p99 = np.nanpercentile(f, [1, 50, 99])

    if p99 > 500 or p50 > 100:
        return "raw"
    elif 5 < p1 < 30 and 20 < p50 < 60 and 40 < p99 < 100:
        return "dffp"
    elif 0.1 < p1 < 0.2 < p50 < 0.5 < p99 < 1.0:
        return "dff"
    elif p1 < 0 and abs(p50) < 1 and p99 > 0:
        return "zscore"
    else:
        return "unknown"


def format_time(t):
    """
    Format a time value in seconds to a human-readable string.

    Parameters
    ----------
    t : float
        Time in seconds.

    Returns
    -------
    str
        Formatted time string (e.g., "30 s", "5 min", "2 h").
    """
    if t < 60:
        return f"{int(np.ceil(t))} s"
    elif t < 3600:
        return f"{int(round(t / 60))} min"
    else:
        return f"{int(round(t / 3600))} h"


def get_color_permutation(n):
    """
    Generate a permutation of indices for visually distinct color ordering.

    Uses a coprime step to spread colors evenly across the color space.

    Parameters
    ----------
    n : int
        Number of items to permute.

    Returns
    -------
    list
        Permuted indices [0, n-1].
    """
    for s in range(n // 2 + 1, n):
        if math.gcd(s, n) == 1:
            return [(i * s) % n for i in range(n)]
    return list(range(n))


class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """
    create an anchored horizontal scale bar.

    parameters
    ----------
    size : float, optional
        bar length in data units (fixed; default is 1).
    label : str, optional
        text label (default is "").
    loc : int, optional
        location code (default is 2).
    ax : axes, optional
        axes to attach the bar (default uses current axes).
    pad, borderpad, ppad, sep : float, optional
        spacing parameters.
    linekw : dict, optional
        line properties.
    """

    def __init__(
        self,
        size=1,
        label="",
        loc=2,
        ax=None,
        pad=0.4,
        borderpad=0.5,
        ppad=0,
        sep=2,
        prop=None,
        frameon=True,
        linekw=None,
        **kwargs,
    ):
        if linekw is None:
            linekw = {}
        if ax is None:
            ax = plt.gca()
        trans = ax.transAxes

        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, size], [0, 0], **linekw)
        size_bar.add_artist(line)
        txt = matplotlib.offsetbox.TextArea(label)
        self.txt = txt
        self.vpac = VPacker(children=[size_bar, txt], align="center", pad=ppad, sep=sep)
        super().__init__(
            loc,  # noqa
            pad=pad,
            borderpad=borderpad,
            child=self.vpac,
            prop=prop,
            frameon=frameon,
            **kwargs,
        )


class AnchoredVScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """
    Create an anchored vertical scale bar.

    Parameters
    ----------
    height : float, optional
        Bar height in data units (default is 1).
    label : str, optional
        Text label (default is "").
    loc : int, optional
        Location code (default is 2).
    ax : axes, optional
        Axes to attach the bar (default uses current axes).
    pad, borderpad, ppad, sep : float, optional
        Spacing parameters.
    linekw : dict, optional
        Line properties.
    spacer_width : float, optional
        Width of spacer between bar and text.
    """

    def __init__(
        self,
        height=1,
        label="",
        loc=2,
        ax=None,
        pad=0.4,
        borderpad=0.5,
        ppad=0,
        sep=2,
        prop=None,
        frameon=True,
        linekw=None,
        spacer_width=6,
        **kwargs,
    ):
        if ax is None:
            ax = plt.gca()
        if linekw is None:
            linekw = {}
        trans = ax.transAxes

        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, 0], [0, height], **linekw)
        size_bar.add_artist(line)

        txt = matplotlib.offsetbox.TextArea(
            label, textprops=dict(rotation=90, ha="left", va="bottom")
        )
        self.txt = txt

        spacer = DrawingArea(spacer_width, 0, 0, 0)
        self.hpac = HPacker(
            children=[size_bar, spacer, txt], align="bottom", pad=ppad, sep=sep
        )
        super().__init__(
            loc,  # noqa
            pad=pad,
            borderpad=borderpad,
            child=self.hpac,
            prop=prop,
            frameon=frameon,
            **kwargs,
        )


def plot_traces_noise(
    dff_noise,
    colors,
    fps=17.0,
    window=220,
    savepath=None,
    title="Trace Noise",
    lw=0.5,
):
    """
    Plot stacked noise traces in the same style as plot_traces.

    Parameters
    ----------
    dff_noise : ndarray
        Noise traces, shape (n_neurons, n_timepoints).
    colors : ndarray
        Colormap array returned from plot_traces(return_color=True).
    fps : float
        Sampling rate, Hz.
    window : float
        Time window (seconds) to display.
    savepath : str or Path, optional
        If given, save to file.
    title : str
        Title for figure.
    lw : float
        Line width.
    """

    n_neurons, n_timepoints = dff_noise.shape
    data_time = np.arange(n_timepoints) / fps
    current_frame = min(int(window * fps), n_timepoints - 1)

    # auto offset based on noise traces
    p10 = np.percentile(dff_noise[:, : current_frame + 1], 10, axis=1)
    p90 = np.percentile(dff_noise[:, : current_frame + 1], 90, axis=1)
    offset = np.median(p90 - p10) * 1.2

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.tick_params(axis="x", which="both", labelbottom=False, length=0, colors="white")
    ax.tick_params(axis="y", which="both", labelleft=False, length=0, colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in reversed(range(n_neurons)):
        trace = dff_noise[i, : current_frame + 1]
        shifted_trace = trace + i * offset
        ax.plot(
            data_time[: current_frame + 1],
            shifted_trace,
            color=colors[i],
            lw=lw,
            zorder=-i,
        )

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", color="white")

    if savepath:
        plt.savefig(savepath, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()


def plot_traces(
        f,
        save_path: str | Path = "",
        cell_indices: np.ndarray | list[int] | None = None,
        fps=17.0,
        num_neurons=20,
        window=220,
        title="",
        offset=None,
        lw=0.5,
        cmap="tab10",
        scale_bar_unit: str = None,
        mask_overlap: bool = True,
        fig_text: str = None,
) -> None:
    """
    Plot stacked fluorescence traces with automatic offset and scale bars.

    Parameters
    ----------
    f : ndarray or str or Path
        2d array of fluorescence traces (n_neurons x n_timepoints),
        or path to Suite2p plane directory containing norm_traces.npy/F.npy.
    save_path : str, optional
        Path to save the output plot.
    fps : float
        Sampling rate in frames per second.
    num_neurons : int
        Number of neurons to display if cell_indices is None.
    window : float
        Time window (in seconds) to display.
    title : str
        Title of the figure.
    offset : float or None
        Vertical offset between traces; if None, computed automatically.
    lw : float
        Line width for data points.
    cmap : str
        Matplotlib colormap string.
    scale_bar_unit : str, optional
        Unit suffix for the vertical scale bar (e.g., "% ΔF/F₀", "a.u.").
        The numeric value is computed automatically based on the plot's
        vertical scale. If None, inferred from data range.
    cell_indices : array-like or None
        Specific cell indices to plot. If provided, overrides num_neurons.
    mask_overlap : bool, default True
        If True, lower traces mask (occlude) traces above them, creating
        a layered effect where each trace has a black background.
    """
    # Handle path input - load data from Suite2p directory
    if isinstance(f, (str, Path)):
        plane_dir = Path(f)
        if plane_dir.is_dir():
            # Try to load norm_traces.npy first, fall back to F.npy
            norm_path = plane_dir / "norm_traces.npy"
            f_path = plane_dir / "F.npy"
            iscell_path = plane_dir / "iscell.npy"
            ops_path = plane_dir / "ops.npy"

            if norm_path.exists():
                f = np.load(norm_path)
                if scale_bar_unit is None:
                    scale_bar_unit = (
                        "z-score" if infer_units(f) == "zscore"
                        else r"% $\Delta$F/F$_0$"
                    )
            elif f_path.exists():
                f = np.load(f_path)
                if scale_bar_unit is None:
                    scale_bar_unit = "a.u."
            else:
                raise FileNotFoundError(f"No norm_traces.npy or F.npy found in {plane_dir}")

            # Filter to accepted cells if iscell exists and no cell_indices provided
            if cell_indices is None and iscell_path.exists():
                iscell = np.load(iscell_path)
                cell_indices = iscell[:, 0].astype(bool)

            # Get fps from ops if available
            if ops_path.exists():
                ops = load_ops(ops_path)
                fps = ops.get("fs", fps)

            # Set save_path if not provided
            if not save_path:
                save_path = plane_dir / "figures" / "traces.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Path is not a directory: {f}")

    if isinstance(f, dict):
        raise ValueError("f must be a numpy array, not a dictionary")

    n_timepoints = f.shape[-1]
    data_time = np.arange(n_timepoints) / fps
    current_frame = min(int(window * fps), n_timepoints - 1)

    if cell_indices is None:
        displayed_neurons = min(num_neurons, f.shape[0])
        indices = np.arange(displayed_neurons)
    else:
        indices = np.array(cell_indices)
        if indices.dtype == bool:
            indices = np.where(indices)[0]  # convert boolean mask to int indices
        displayed_neurons = len(indices)

    if len(indices) == 0:
        return None

    if offset is None:
        p10 = np.percentile(f[indices, : current_frame + 1], 10, axis=1)
        p90 = np.percentile(f[indices, : current_frame + 1], 90, axis=1)
        offset = np.median(p90 - p10) * 1.2
        # Ensure minimum offset to prevent trace overlap
        min_offset = np.percentile(p90 - p10, 75) * 0.8
        offset = max(offset, min_offset, 1e-6)  # Absolute minimum to prevent divide-by-zero

    cmap_inst = plt.get_cmap(cmap)
    colors = cmap_inst(np.linspace(0, 1, displayed_neurons))
    perm = get_color_permutation(displayed_neurons)
    colors = colors[perm]

    # build shifted traces array (no masking — let z-order handle overlap)
    shifted_traces = np.zeros((displayed_neurons, current_frame + 1))
    for i in range(displayed_neurons):
        trace = f[indices[i], : current_frame + 1]
        baseline = np.percentile(trace, 8)
        shifted_traces[i] = (trace - baseline) + i * offset

    # Plot traces with z-ordering (lower traces on top via higher zorder)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.tick_params(axis="x", which="both", labelbottom=False, length=0, colors="white")
    ax.tick_params(axis="y", which="both", labelleft=False, length=0, colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Plot from top to bottom so lower-indexed traces appear on top
    time_slice = data_time[: current_frame + 1]
    for i in range(displayed_neurons - 1, -1, -1):
        z = displayed_neurons - i  # Lower index = higher zorder = on top
        if mask_overlap:
            # Fill below trace with black to mask traces above
            ax.fill_between(
                time_slice,
                shifted_traces[i],
                y2=shifted_traces[i].min() - offset,
                color="black",
                zorder=z - 0.5,
            )
        ax.plot(
            time_slice,
            shifted_traces[i],
            color=colors[i],
            lw=lw,
            zorder=z,
        )

    time_bar_length = 0.1 * window
    if time_bar_length < 60:
        time_label = f"{time_bar_length:.0f} s"
    elif time_bar_length < 3600:
        time_label = f"{time_bar_length / 60:.0f} min"
    else:
        time_label = f"{time_bar_length / 3600:.1f} hr"

    # Set y-limits with small padding (no extra space for scalebars - they go outside)
    y_min = np.min(shifted_traces)
    y_max = np.max(shifted_traces)
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.02, y_max + y_range * 0.02)

    # Compute vertical scale bar value (10% of y-range in data units)
    scale_bar_height_frac = 0.10  # 10% of axes height
    scale_bar_data_value = y_range * scale_bar_height_frac

    # Use provided unit or default to "a.u."
    if scale_bar_unit is None:
        scale_bar_unit = "a.u."

    # Format the scale bar label with computed value
    if scale_bar_data_value >= 100:
        scale_bar_label = f"{int(round(scale_bar_data_value, -1))} {scale_bar_unit}"
    elif scale_bar_data_value >= 10:
        scale_bar_label = f"{int(round(scale_bar_data_value))} {scale_bar_unit}"
    elif scale_bar_data_value >= 1:
        scale_bar_label = f"{scale_bar_data_value:.0f} {scale_bar_unit}"
    else:
        scale_bar_label = f"{scale_bar_data_value:.2f} {scale_bar_unit}"

    # Adjust subplot to make room for scalebars at bottom and right
    fig.subplots_adjust(bottom=0.12, right=0.88)

    linekw = dict(color="white", linewidth=3)

    # Time scale bar - use fig.text for fixed position below axes
    # Get axes position in figure coordinates
    ax_pos = ax.get_position()
    time_bar_x = ax_pos.x1 - 0.02  # right side of axes
    time_bar_y = 0.07  # fixed position just below axes

    # Draw horizontal line for time scale bar
    line_width_fig = 0.08  # width in figure coords
    fig.add_artist(plt.Line2D(
        [time_bar_x - line_width_fig, time_bar_x],
        [time_bar_y, time_bar_y],
        transform=fig.transFigure,
        color="white",
        linewidth=3,
        clip_on=False,
    ))
    # Add time label
    fig.text(
        time_bar_x - line_width_fig / 2,
        time_bar_y - 0.02,
        time_label,
        ha="center",
        va="top",
        color="white",
        fontsize=10,
        transform=fig.transFigure,
    )

    # Vertical scale bar - positioned just outside right edge, bottom aligned with x-axis
    vsb = AnchoredVScaleBar(
        height=scale_bar_height_frac,
        label=scale_bar_label,
        loc="lower right",
        frameon=False,
        pad=0.5,
        sep=4,
        linekw=linekw,
        ax=ax,
        spacer_width=0,
    )
    # Position just outside right edge of axes, bottom at y=0
    vsb.set_bbox_to_anchor((1.02, 0.0), transform=ax.transAxes)
    vsb.txt._text.set_color("white")
    ax.add_artist(vsb)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", color="white")

    ax.set_ylabel(
        f"Neuron Count: {displayed_neurons}",
        fontsize=10,
        fontweight="bold",
        color="white",
        labelpad=5,
    )

    if fig_text:
        fig.text(
            0.02, 0.02, fig_text,
            ha="left", va="bottom",
            color="#bbbbbb", fontsize=8, fontname="monospace",
            transform=fig.transFigure,
        )

    if save_path:
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()
    return None


def animate_traces(
    f,
    save_path="./traces.mp4",
    cell_indices=None,
    fps=17.0,
    num_neurons=20,
    window=30,
    title="",
    offset=None,
    lw=0.5,
    cmap="tab10",
    scale_bar_unit=None,
    mask_overlap=True,
    anim_fps=30,
    speed=1.0,
    dpi=150,
):
    """
    Animated version of plot_traces - scrolling window through time.

    Creates an mp4 video showing traces scrolling like an oscilloscope display.
    Visual style matches plot_traces exactly.

    Parameters
    ----------
    f : ndarray
        2d array of fluorescence traces (n_neurons x n_timepoints).
    save_path : str or Path, default "./traces.mp4"
        Output path for the animation.
    cell_indices : array-like or None
        Specific cell indices to plot. If provided, overrides num_neurons.
    fps : float, default 17.0
        Data frame rate in Hz.
    num_neurons : int, default 20
        Number of neurons to display if cell_indices is None.
    window : float, default 30
        Time window width in seconds.
    title : str, default ""
        Title for the animation.
    offset : float or None
        Vertical offset between traces; if None, computed automatically.
    lw : float, default 0.5
        Line width for traces.
    cmap : str, default "tab10"
        Matplotlib colormap.
    scale_bar_unit : str, optional
        Unit suffix for vertical scale bar. If None, uses "a.u.".
    mask_overlap : bool, default True
        If True, lower traces mask traces above them.
    anim_fps : int, default 30
        Animation frame rate (frames per second in output video).
    speed : float, default 1.0
        Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed).
    dpi : int, default 150
        Output video resolution.

    Returns
    -------
    str
        Path to saved animation.
    """
    if isinstance(f, dict):
        raise ValueError("f must be a numpy array, not a dictionary")

    n_total, n_timepoints = f.shape
    data_time = np.arange(n_timepoints) / fps
    total_duration = data_time[-1]

    # select neurons
    if cell_indices is None:
        displayed_neurons = min(num_neurons, n_total)
        indices = np.arange(displayed_neurons)
    else:
        indices = np.array(cell_indices)
        if indices.dtype == bool:
            indices = np.where(indices)[0]
        displayed_neurons = len(indices)

    if len(indices) == 0:
        print("No neurons to display")
        return None

    # pre-compute baselines and offset (once, not per frame)
    baselines = np.percentile(f[indices], 8, axis=1)

    if offset is None:
        p10 = np.percentile(f[indices], 10, axis=1)
        p90 = np.percentile(f[indices], 90, axis=1)
        offset = np.median(p90 - p10) * 1.2
        min_offset = np.percentile(p90 - p10, 75) * 0.8
        offset = max(offset, min_offset, 1e-6)

    # colors - use same permutation as plot_traces
    cmap_inst = plt.get_cmap(cmap)
    colors = cmap_inst(np.linspace(0, 1, displayed_neurons))
    perm = get_color_permutation(displayed_neurons)
    colors = colors[perm]

    # compute y-range based on expected stacked layout
    # each trace spans ~offset, so total height is roughly (n_neurons * offset) plus some headroom
    # use per-trace percentiles to avoid outliers
    trace_ranges = []
    for idx in range(displayed_neurons):
        trace = f[indices[idx]] - baselines[idx]
        # use 1st/99th percentile to ignore spikes
        p1, p99 = np.percentile(trace, [1, 99])
        trace_ranges.append(p99 - p1)
    median_trace_range = np.median(trace_ranges)

    # y-range: stack height plus headroom for trace fluctuations
    y_min_global = -median_trace_range * 0.5
    y_max_global = (displayed_neurons - 1) * offset + median_trace_range * 1.5
    y_range = y_max_global - y_min_global
    # ensure minimum range
    if y_range < 1e-6:
        y_range = 1.0

    # setup figure (matches plot_traces)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.tick_params(axis="x", which="both", labelbottom=False, length=0, colors="white")
    ax.tick_params(axis="y", which="both", labelleft=False, length=0, colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(bottom=0.15, right=0.85, left=0.10, top=0.92)

    # scale bar labels
    time_bar_length = 0.1 * window
    if time_bar_length < 60:
        time_label = f"{time_bar_length:.0f} s"
    elif time_bar_length < 3600:
        time_label = f"{time_bar_length / 60:.0f} min"
    else:
        time_label = f"{time_bar_length / 3600:.1f} hr"

    scale_bar_height_frac = 0.10
    scale_bar_data_value = y_range * scale_bar_height_frac
    if scale_bar_unit is None:
        scale_bar_unit = "a.u."
    if scale_bar_data_value >= 100:
        scale_bar_label = f"{int(round(scale_bar_data_value, -1))} {scale_bar_unit}"
    elif scale_bar_data_value >= 10:
        scale_bar_label = f"{int(round(scale_bar_data_value))} {scale_bar_unit}"
    elif scale_bar_data_value >= 1:
        scale_bar_label = f"{scale_bar_data_value:.0f} {scale_bar_unit}"
    else:
        scale_bar_label = f"{scale_bar_data_value:.2f} {scale_bar_unit}"

    # create line objects (lower index = higher zorder = on top)
    lines = []
    fills = []
    for i in range(displayed_neurons - 1, -1, -1):
        z = displayed_neurons - i
        if mask_overlap:
            fill, = ax.fill([], [], color="black", zorder=z - 0.5)
            fills.append((i, fill))
        line, = ax.plot([], [], color=colors[i], lw=lw, zorder=z)
        lines.append((i, line))

    # static elements
    ax.set_ylim(y_min_global - y_range * 0.02, y_max_global + y_range * 0.02)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", color="white")

    ax.set_ylabel(
        f"Neuron Count: {displayed_neurons}",
        fontsize=10, fontweight="bold", color="white", labelpad=5,
    )

    # time scale bar (static position)
    ax_pos = ax.get_position()
    time_bar_x = ax_pos.x1 - 0.02
    time_bar_y = 0.07
    line_width_fig = 0.08
    fig.add_artist(plt.Line2D(
        [time_bar_x - line_width_fig, time_bar_x],
        [time_bar_y, time_bar_y],
        transform=fig.transFigure,
        color="white", linewidth=3, clip_on=False,
    ))
    fig.text(
        time_bar_x - line_width_fig / 2, time_bar_y - 0.02,
        time_label, ha="center", va="top",
        color="white", fontsize=10, transform=fig.transFigure,
    )

    # vertical scale bar
    linekw = dict(color="white", linewidth=3)
    vsb = AnchoredVScaleBar(
        height=scale_bar_height_frac,
        label=scale_bar_label,
        loc="lower right",
        frameon=False, pad=0.5, sep=4,
        linekw=linekw, ax=ax, spacer_width=0,
    )
    vsb.set_bbox_to_anchor((1.02, 0.0), transform=ax.transAxes)
    vsb.txt._text.set_color("white")
    ax.add_artist(vsb)

    # animation frames
    # each animation frame advances by (speed / anim_fps) seconds of data
    step_seconds = speed / anim_fps
    max_start_time = total_duration - window
    n_frames = int(max_start_time / step_seconds) + 1

    def init():
        for i, line in lines:
            line.set_data([], [])
        for i, fill in fills:
            fill.set_xy(np.empty((0, 2)))
        return [line for _, line in lines] + [fill for _, fill in fills]

    def update(frame):
        t_start = frame * step_seconds
        t_end = t_start + window

        i_start = int(t_start * fps)
        i_end = min(int(t_end * fps), n_timepoints)

        time_slice = data_time[i_start:i_end]

        # compute shifted traces for this window
        shifted = np.zeros((displayed_neurons, i_end - i_start))
        for idx, neuron_idx in enumerate(indices):
            trace = f[neuron_idx, i_start:i_end]
            shifted[idx] = (trace - baselines[idx]) + idx * offset

        # update lines and fills
        for i, line in lines:
            line.set_data(time_slice, shifted[i])

        if mask_overlap:
            for i, fill in fills:
                # fill from trace down to below the lowest point
                y_data = shifted[i]
                y_bottom = y_min_global - y_range * 0.1
                # create polygon: trace forward, then bottom backward
                xy = np.column_stack([
                    np.concatenate([time_slice, time_slice[::-1]]),
                    np.concatenate([y_data, np.full(len(y_data), y_bottom)])
                ])
                fill.set_xy(xy)

        ax.set_xlim(t_start, t_end)

        return [line for _, line in lines] + [fill for _, fill in fills]

    ani = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=True, interval=1000 / anim_fps,
    )

    save_path = Path(save_path)
    print(f"Saving animation to {save_path} ({n_frames} frames at {anim_fps} fps)...")
    ani.save(str(save_path), fps=anim_fps, dpi=dpi, writer="ffmpeg")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return str(save_path)


def feather_mask(mask, max_alpha=0.75, edge_width=3):
    """
    Create a feathered alpha mask with soft edges.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary or labeled mask (non-zero = foreground).
    max_alpha : float, optional
        Maximum alpha value at mask center. Default is 0.75.
    edge_width : int, optional
        Width of the feathered edge in pixels. Default is 3.

    Returns
    -------
    numpy.ndarray
        Alpha mask with values in [0, max_alpha].
    """
    dist_out = distance_transform_edt(mask == 0)
    alpha = np.clip((edge_width - dist_out) / edge_width, 0, 1)
    return alpha * max_alpha


def plot_masks(
        img: np.ndarray,
        stat: list[dict] | dict,
        mask_idx: np.ndarray,
        savepath: str | Path = None,
        colors=None,
        title=None,
        *,
        ops: dict = None,
        proj_key: str = None,
        figsize: tuple = (6, 6),
        dpi: int = 300,
):
    """
    Draw ROI overlays onto a projection image.

    Parameters
    ----------
    img : ndarray (Ly x Lx)
        Background image to overlay on.
    stat : list[dict]
        Suite2p ROI stat dictionaries (with "ypix", "xpix", "lam") in
        full-frame coordinates.
    mask_idx : ndarray[bool]
        Boolean array selecting which ROIs to plot.
    savepath : str or Path, optional
        Path to save the figure. If None, displays with plt.show().
    colors : ndarray or list, optional
        Array/list of RGB tuples for each ROI selected.
        If None, colors are assigned via HSV colormap.
    title : str, optional
        Title string to place on the figure.
    ops : dict, optional
        Suite2p ops dictionary. When provided together with ``proj_key``,
        stat coordinates are translated into the image's space (using
        ``yrange`` / ``xrange`` for pre-cropped projections like
        ``max_proj`` / ``Vcorr``).
    proj_key : str, optional
        Projection key that ``img`` was taken from (e.g. ``"meanImg"``,
        ``"max_proj"``). Required alongside ``ops`` to select the correct
        coordinate-space offset.
    figsize : tuple, optional
        Figure size in inches. Default (6, 6) to match
        :func:`plot_projection`.
    dpi : int, optional
        Output DPI. Default 300.
    """
    # crop to the valid region when ops info is provided, and compute the
    # offset needed to move full-frame stat coordinates into image space
    stat_yoff = 0
    stat_xoff = 0
    if ops is not None and proj_key is not None:
        img, stat_yoff, stat_xoff = _crop_projection_to_valid(ops, img, proj_key)

    # percentile stretch for consistent contrast
    vmin = np.nanpercentile(img, 1)
    vmax = np.nanpercentile(img, 99)
    normalized = (img - vmin) / (vmax - vmin + 1e-6)
    normalized = np.clip(normalized, 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.0)
    canvas = np.tile(normalized, (3, 1, 1)).transpose(1, 2, 0)

    Ly, Lx = img.shape[:2]

    n_masks = mask_idx.sum()
    if colors is None:
        colors = plt.cm.hsv(np.linspace(0, 1, n_masks + 1))[:, :3]  # noqa

    c = 0
    for n, s in enumerate(stat):
        if mask_idx[n]:
            ypix = s["ypix"] - stat_yoff
            xpix = s["xpix"] - stat_xoff
            lam = s["lam"]

            valid_mask = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
            if not np.any(valid_mask):
                c += 1
                continue

            ypix = ypix[valid_mask]
            xpix = xpix[valid_mask]
            lam = lam[valid_mask]

            lam = lam / (lam.max() + 1e-10)
            col = colors[c]
            c += 1
            for k in range(3):
                canvas[ypix, xpix, k] = (
                        0.5 * canvas[ypix, xpix, k] + 0.5 * col[k] * lam
                )

    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(canvas, interpolation="nearest")
    if title is not None:
        ax.set_title(title, fontsize=10, color="white", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if savepath:
        if Path(savepath).is_dir():
            raise ValueError("savepath must be a file path, not a directory.")
        plt.savefig(savepath, dpi=dpi, facecolor="black")
        plt.close(fig)
    else:
        plt.show()


def _build_accepted_rejected_canvas(
        img: np.ndarray,
        stat,
        iscell_mask: np.ndarray,
        *,
        ops: dict = None,
        proj_key: str = None,
        accepted_color=(0.0, 1.0, 0.0),
        rejected_color=(1.0, 0.0, 0.0),
):
    """
    Build an RGB canvas of a projection with accepted/rejected ROIs blended
    on top using suite2p ``lam`` weights — the same per-pixel opacity used
    by :func:`plot_masks`.

    Returns
    -------
    canvas : ndarray (Ly, Lx, 3)
    n_accepted, n_rejected : int
    """
    stat_yoff = 0
    stat_xoff = 0
    if ops is not None and proj_key is not None:
        img, stat_yoff, stat_xoff = _crop_projection_to_valid(ops, img, proj_key)

    vmin = np.nanpercentile(img, 1)
    vmax = np.nanpercentile(img, 99)
    normalized = (img - vmin) / (vmax - vmin + 1e-6)
    normalized = np.clip(normalized, 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.0)
    canvas = np.tile(normalized, (3, 1, 1)).transpose(1, 2, 0).astype(np.float32)

    Ly, Lx = img.shape[:2]
    iscell_mask = np.asarray(iscell_mask, dtype=bool)

    accepted_color = np.asarray(accepted_color, dtype=np.float32)
    rejected_color = np.asarray(rejected_color, dtype=np.float32)

    for n, s in enumerate(stat):
        ypix = np.asarray(s.get("ypix", []), dtype=int) - stat_yoff
        xpix = np.asarray(s.get("xpix", []), dtype=int) - stat_xoff
        lam = np.asarray(s.get("lam", []), dtype=np.float32)
        if ypix.size == 0:
            continue
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        if not np.any(valid):
            continue
        ypix, xpix, lam = ypix[valid], xpix[valid], lam[valid]
        lam = lam / (lam.max() + 1e-10)
        col = accepted_color if iscell_mask[n] else rejected_color
        for k in range(3):
            canvas[ypix, xpix, k] = (
                0.5 * canvas[ypix, xpix, k] + 0.5 * col[k] * lam
            )

    n_accepted = int(iscell_mask.sum())
    n_rejected = int((~iscell_mask).sum())
    return canvas, n_accepted, n_rejected


def plot_accepted_rejected_overlay(
        img: np.ndarray,
        stat,
        iscell_mask: np.ndarray,
        savepath: str | Path = None,
        title: str = None,
        *,
        ops: dict = None,
        proj_key: str = None,
        figsize: tuple = (6, 6),
        dpi: int = 300,
):
    """
    Draw accepted (green) and rejected (red) ROI overlays on a projection.

    Uses the same lam-weighted per-pixel opacity as :func:`plot_masks` so
    feathering matches the rest of the segmentation figures. Adds dark
    formatting and ``Accepted``/``Rejected`` count labels at the top.

    Parameters
    ----------
    img : ndarray (Ly x Lx)
        Background projection image.
    stat : list[dict]
        Suite2p ROI stat dictionaries (full-frame coordinates).
    iscell_mask : ndarray[bool]
        Boolean array, True for accepted ROIs.
    savepath : str or Path, optional
        Path to save the figure. If None, displays with plt.show().
    title : str, optional
        Projection-name title rendered above the count labels.
    ops : dict, optional
        Suite2p ops dictionary. With ``proj_key``, the image is cropped to
        the valid (non-padded) region and stat coordinates are translated
        accordingly.
    proj_key : str, optional
        Projection key matching ``img`` (e.g. ``"meanImg"``, ``"max_proj"``).
    figsize : tuple, optional
        Figure size. Default (6, 6) to match other segmentation figures.
    dpi : int, optional
        Output DPI. Default 300.
    """
    canvas, n_accepted, n_rejected = _build_accepted_rejected_canvas(
        img, stat, iscell_mask, ops=ops, proj_key=proj_key,
    )

    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(canvas, interpolation="nearest")

    label_y = 1.02
    if title:
        ax.text(
            0.5, 1.02, title,
            transform=ax.transAxes,
            fontsize=12, fontweight="bold", fontname="monospace",
            color="white", ha="center", va="bottom",
        )
        label_y = 1.10
    ax.text(
        0.37, label_y,
        f"Accepted: {n_accepted:03d}",
        transform=ax.transAxes,
        fontsize=14, fontweight="bold", fontname="monospace",
        color="lime", ha="right", va="bottom",
    )
    ax.text(
        0.63, label_y,
        f"Rejected: {n_rejected:03d}",
        transform=ax.transAxes,
        fontsize=14, fontweight="bold", fontname="monospace",
        color="red", ha="left", va="bottom",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    plt.tight_layout()

    if savepath:
        if Path(savepath).is_dir():
            raise ValueError("savepath must be a file path, not a directory.")
        plt.savefig(savepath, dpi=dpi, facecolor="black")
        plt.close(fig)
    else:
        plt.show()


def plot_volume_accepted_rejected_overlay(
        ops_files,
        savepath: str | Path = None,
        *,
        ncols: int = None,
        figsize: tuple = None,
        dpi: int = 300,
):
    """
    Volumetric version of :func:`plot_accepted_rejected_overlay`.

    One subplot per z-plane showing the projection used by cellpose with
    accepted ROIs in green and rejected in red. Uses the same lam-weighted
    opacity as :func:`plot_masks`. Per-subplot titles show the plane's
    ``Accepted: N`` (lime) and ``Rejected: N`` (red) counts; an overall
    title at the very top reports total accepted/rejected across the volume.

    Parameters
    ----------
    ops_files : list of str or Path
        Paths to per-plane ``ops.npy`` files.
    savepath : str or Path, optional
        Where to save the figure. If None, calls plt.show().
    ncols : int, optional
        Number of columns in the subplot grid. Defaults to a near-square
        layout.
    figsize : tuple, optional
        Figure size. Defaults based on the grid shape.
    dpi : int, optional
        Output DPI. Default 300.
    """
    ops_files = [Path(p) for p in ops_files]
    n_planes = len(ops_files)
    if n_planes == 0:
        raise ValueError("ops_files is empty")

    if ncols is None:
        ncols = int(np.ceil(np.sqrt(n_planes)))
    nrows = int(np.ceil(n_planes / ncols))
    if figsize is None:
        # extra vertical room for two-line per-subplot title
        figsize = (4 * ncols, 4.8 * nrows)

    proj_lookup = {
        0: ("Vcorr", "Correlation Image"),
        1: ("max_proj", "Max Projection"),
        2: ("meanImg", "Mean Image"),
        3: ("meanImgE", "Enhanced Mean Image"),
        4: ("max_proj", "Max Projection"),
    }
    fallback_titles = {
        "meanImg": "Mean Image",
        "max_proj": "Max Projection",
        "meanImgE": "Enhanced Mean Image",
        "Vcorr": "Correlation Image",
    }

    def _is_valid_image(im):
        if im is None:
            return False
        if isinstance(im, (int, float)) and im == 0:
            return False
        if isinstance(im, np.ndarray) and im.size == 0:
            return False
        return True

    def _plane_num(ops, fallback):
        raw = ops.get("plane", None)
        if raw is None:
            return fallback
        if isinstance(raw, (int, np.integer)):
            return int(raw)
        digits = "".join(c for c in str(raw) if c.isdigit())
        return int(digits) if digits else fallback

    plane_entries = []
    for i, ops_file in enumerate(ops_files):
        try:
            ops = load_ops(ops_file)
        except Exception as e:
            print(f"  Warning: could not load {ops_file}: {e}")
            continue
        try:
            res = load_planar_results(ops)
        except Exception as e:
            print(f"  Warning: could not load planar results for {ops_file}: {e}")
            continue

        stat = res["stat"]
        iscell_mask = res["iscell"][:, 0].astype(bool)

        anatomical_only = int(ops.get("anatomical_only", 0) or 0)
        proj_key, proj_title = proj_lookup.get(anatomical_only, ("meanImg", "Mean Image"))
        img = ops.get(proj_key)
        if not _is_valid_image(img):
            for fk in ("meanImg", "max_proj", "meanImgE", "Vcorr"):
                if _is_valid_image(ops.get(fk)):
                    proj_key = fk
                    proj_title = fallback_titles[fk]
                    img = ops.get(fk)
                    break
        if not _is_valid_image(img):
            continue

        plane_entries.append({
            "plane": _plane_num(ops, i),
            "ops": ops,
            "img": img,
            "stat": stat,
            "iscell_mask": iscell_mask,
            "proj_key": proj_key,
            "proj_title": proj_title,
        })

    if not plane_entries:
        raise RuntimeError("No valid plane data found in ops_files")

    plane_entries.sort(key=lambda e: e["plane"])

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="black")
    axes = np.atleast_1d(axes).ravel()

    total_accepted = 0
    total_rejected = 0

    for idx, ax in enumerate(axes):
        ax.set_facecolor("black")
        if idx >= len(plane_entries):
            ax.axis("off")
            continue
        e = plane_entries[idx]
        canvas, n_acc, n_rej = _build_accepted_rejected_canvas(
            e["img"], e["stat"], e["iscell_mask"],
            ops=e["ops"], proj_key=e["proj_key"],
        )
        total_accepted += n_acc
        total_rejected += n_rej

        ax.imshow(canvas, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(
            0.5, 1.085, f"plane {e['plane']:02d}",
            transform=ax.transAxes,
            fontsize=11, fontweight="bold", fontname="monospace",
            color="white", ha="center", va="bottom",
        )
        ax.text(
            0.49, 1.015, f"Accepted: {n_acc:03d}",
            transform=ax.transAxes,
            fontsize=10, fontweight="bold", fontname="monospace",
            color="lime", ha="right", va="bottom",
        )
        ax.text(
            0.51, 1.015, f"Rejected: {n_rej:03d}",
            transform=ax.transAxes,
            fontsize=10, fontweight="bold", fontname="monospace",
            color="red", ha="left", va="bottom",
        )

    fig.text(
        0.49, 0.992,
        f"Total Accepted: {total_accepted:04d}",
        color="lime", fontsize=16, fontweight="bold", fontname="monospace",
        ha="right", va="top",
    )
    fig.text(
        0.51, 0.992,
        f"Total Rejected: {total_rejected:04d}",
        color="red", fontsize=16, fontweight="bold", fontname="monospace",
        ha="left", va="top",
    )

    # leave room for fig-level totals at top, and use hspace for two-line per-plot titles
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    plt.subplots_adjust(hspace=0.28, wspace=0.05)

    if savepath:
        if Path(savepath).is_dir():
            raise ValueError("savepath must be a file path, not a directory.")
        plt.savefig(savepath, dpi=dpi, facecolor="black")
        plt.close(fig)
    else:
        plt.show()


def plot_projection(
    ops,
    output_directory=None,
    fig_label=None,
    vmin=None,
    vmax=None,
    add_scalebar=False,
    proj="meanImg",
    display_masks=False,
    accepted_only=False,
):
    """
    Plot a projection image from ops with optional mask overlay.

    Parameters
    ----------
    ops : dict or str or Path
        Suite2p ops dictionary or path to ops.npy.
    output_directory : str or Path, optional
        Directory to save figure. If None, displays interactively.
    fig_label : str, optional
        Label for y-axis.
    vmin, vmax : float, optional
        Intensity display range. Defaults to 2nd/98th percentiles.
    add_scalebar : bool, default False
        Whether to add a scale bar.
    proj : str, default "meanImg"
        Projection type: "meanImg", "max_proj", or "meanImgE".
    display_masks : bool, default False
        Whether to overlay detected ROI masks.
    accepted_only : bool, default False
        If True, only show accepted cells when display_masks=True.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    # upstream removed `ROI` from suite2p.detection.stats; inline the only
    # method we used — stats_dicts_to_3d_array — as a local helper so this
    # plotting path keeps working under both fork and upstream suite2p.
    def _stats_dicts_to_3d_array(stat_list, Ly, Lx, label_id=True):
        arr = np.zeros((len(stat_list), Ly, Lx), dtype=np.float32)
        for i, s in enumerate(stat_list):
            ypix = np.asarray(s.get("ypix", []), dtype=int)
            xpix = np.asarray(s.get("xpix", []), dtype=int)
            if ypix.size == 0:
                continue
            valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
            ypix, xpix = ypix[valid], xpix[valid]
            arr[i, ypix, xpix] = (i + 1) if label_id else 1.0
        return arr
    if proj == "meanImg":
        txt = "Mean-Image"
    elif proj == "max_proj":
        txt = "Max-Projection"
    elif proj == "meanImgE":
        txt = "Mean-Image (Enhanced)"
    else:
        raise ValueError(
            "Unknown projection type. Options are ['meanImg', 'max_proj', 'meanImgE']"
        )

    if output_directory:
        output_directory = Path(output_directory)

    data = ops[proj]
    # max_proj/Vcorr are pre-cropped by suite2p to yrange/xrange; the
    # helper computes the offsets needed to align mask overlays with
    # full-frame stat coordinates and returns the image unchanged.
    _full_Ly, _full_Lx = data.shape[:2]
    data, _stat_yoff, _stat_xoff = _crop_projection_to_valid(ops, data, proj)
    shape = data.shape
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    vmin = np.nanpercentile(data, 2) if vmin is None else vmin
    vmax = np.nanpercentile(data, 98) if vmax is None else vmax

    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)

    # move projection title higher if masks are displayed to avoid overlap.
    proj_title_y = 1.07 if display_masks else 1.02
    ax.text(
        0.5,
        proj_title_y,
        txt,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        fontname="monospace",
        color="white",
        ha="center",
        va="bottom",
    )
    if fig_label:
        fig_label = fig_label.replace("_", " ").replace("-", " ").replace(".", " ")
        ax.set_ylabel(fig_label, color="white", fontweight="bold", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    if display_masks:
        res = load_planar_results(ops)
        stat = res["stat"]
        iscell_mask = res["iscell"][:, 0].astype(bool)
        # build overlay in full-frame space (stat ypix/xpix are full-frame
        # absolute coords from upstream), then slice the same full-frame
        # rectangle that the projection was cropped to so masks and data
        # stay aligned regardless of full-vs-cropped image space.
        _full_Ly_ops = get_param(ops, "Ly", default=_stat_yoff + shape[0])
        _full_Lx_ops = get_param(ops, "Lx", default=_stat_xoff + shape[1])
        im = _stats_dicts_to_3d_array(
            stat, Ly=_full_Ly_ops, Lx=_full_Lx_ops, label_id=True
        )
        _full_y0 = _stat_yoff
        _full_y1 = _stat_yoff + shape[0]
        _full_x0 = _stat_xoff
        _full_x1 = _stat_xoff + shape[1]
        im = im[:, _full_y0:_full_y1, _full_x0:_full_x1]
        im[im == 0] = np.nan
        accepted_cells = np.sum(iscell_mask)
        rejected_cells = np.sum(~iscell_mask)
        cell_rois = _resize_masks_fit_crop(
            np.nanmax(im[iscell_mask], axis=0) if np.any(iscell_mask) else np.zeros_like(im[0]),
            shape,
        )
        green_overlay = np.zeros((*shape, 4), dtype=np.float32)
        green_overlay[..., 3] = feather_mask(cell_rois > 0, max_alpha=0.9)
        green_overlay[..., 1] = 1
        ax.imshow(green_overlay)
        if not accepted_only:
            non_cell_rois = _resize_masks_fit_crop(
                (
                    np.nanmax(im[~iscell_mask], axis=0)
                    if np.any(~iscell_mask)
                    else np.zeros_like(im[0])
                ),
                shape,
            )
            magenta_overlay = np.zeros((*shape, 4), dtype=np.float32)
            magenta_overlay[..., 0] = 1
            magenta_overlay[..., 2] = 1
            magenta_overlay[..., 3] = (non_cell_rois > 0) * 0.5
            ax.imshow(magenta_overlay)
        ax.text(
            0.37,
            1.02,
            f"Accepted: {accepted_cells:03d}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            fontname="monospace",
            color="lime",
            ha="right",
            va="bottom",
        )
        ax.text(
            0.63,
            1.02,
            f"Rejected: {rejected_cells:03d}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            fontname="monospace",
            color="magenta",
            ha="left",
            va="bottom",
        )
    if add_scalebar and "dx" in ops:
        pixel_size = ops["dx"]
        scale_bar_length = 100 / pixel_size
        scalebar_x = shape[1] * 0.05
        scalebar_y = shape[0] * 0.90
        ax.add_patch(
            Rectangle(
                (scalebar_x, scalebar_y),
                scale_bar_length,
                5,
                edgecolor="white",
                facecolor="white",
            )
        )
        ax.text(
            scalebar_x + scale_bar_length / 2,
            scalebar_y - 10,
            "100 μm",
            color="white",
            fontsize=10,
            ha="center",
            fontweight="bold",
        )

    # remove the spines that will show up as white bars
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if output_directory:
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_directory, dpi=300, facecolor="black")
        plt.close(fig)
    else:
        plt.show()


def plot_noise_distribution(
    noise_levels: np.ndarray, output_filename=None, title="Noise Level Distribution"
):
    """
    Plots and saves the distribution of noise levels across neurons as a standardized image.

    Parameters
    ----------
    noise_levels : np.ndarray
        1D array of noise levels for each neuron.
    output_filename : str or Path, optional
        Path to save the plot. If empty, the plot will be displayed instead of saved.
    title : str, optional
        Suptitle for plot, default is "Noise Level Distribution".

    See Also
    --------
    lbm_suite2p_python.dff_shot_noise
    """
    if output_filename:
        output_filename = Path(output_filename)
        if output_filename.is_dir():
            raise AttributeError(
                f"save_path should be a fully qualified file path, not a directory: {output_filename}"
            )

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.hist(noise_levels, bins=50, color="gray", alpha=0.7, edgecolor="dimgray")

    mean_noise: float = np.mean(noise_levels)  # noqa
    ax.axvline(
        mean_noise,
        color="r",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_noise:.2f}",
    )

    ax.set_xlabel("Noise Level", fontsize=14, fontweight="bold", color="white")
    ax.set_ylabel("Number of Neurons", fontsize=14, fontweight="bold", color="white")
    ax.set_title(title, fontsize=16, fontweight="bold", color="white")
    ax.legend(fontsize=12, facecolor="black", edgecolor="gray", labelcolor="white")
    ax.tick_params(colors="white", labelsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    if output_filename:
        plt.savefig(output_filename, dpi=200, bbox_inches="tight", facecolor="black")
        plt.close(fig)
    else:
        plt.show()


def plot_rastermap(
    spks,
    model,
    neuron_bin_size=None,
    fps=17,
    vmin=0,
    vmax=0.8,
    xmin=0,
    xmax=None,
    save_path=None,
    title=None,
    title_kwargs=None,
    fig_text=None,
):
    """
    Plot rastermap visualization of neural activity sorted by embedding.

    Parameters
    ----------
    spks : ndarray
        Spike or activity matrix of shape (n_neurons, n_timepoints).
    model : rastermap.Rastermap
        Fitted rastermap model with `isort` attribute for neuron ordering.
    neuron_bin_size : int, optional
        Number of neurons to bin together. Auto-calculated if None.
    fps : float, default 17
        Frame rate for time axis scaling.
    vmin, vmax : float, default 0, 0.8
        Colormap intensity limits.
    xmin, xmax : int, optional
        Time range to display (in frames).
    save_path : str or Path, optional
        Path to save figure. If None, displays interactively.
    title : str, optional
        Figure title.
    title_kwargs : dict, optional
        Formatting kwargs for title text.
    fig_text : str, optional
        Additional text annotation.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    n_neurons, n_timepoints = spks.shape
    if title_kwargs is None:
        title_kwargs = dict(fontsize=14, fontweight="bold", color="white")

    if neuron_bin_size is None:
        neuron_bin_size = max(1, np.ceil(n_neurons // 500))
    else:
        neuron_bin_size = max(1, min(neuron_bin_size, n_neurons))

    sn = bin1d(spks[model.isort], neuron_bin_size, axis=0)
    if xmax is None or xmax < xmin or xmax > sn.shape[1]:
        xmax = sn.shape[1]
    sn = sn[:, xmin:xmax]

    current_time = np.round((xmax - xmin) / fps, 1)
    current_neurons = sn.shape[0]

    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    img = ax.imshow(sn, cmap="gray_r", vmin=vmin, vmax=vmax, aspect="auto")

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.tick_params(axis="both", labelbottom=False, labelleft=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    heatmap_pos = ax.get_position()

    scalebar_length = heatmap_pos.width * 0.1  # 10% width of heatmap
    scalebar_duration = np.round(
        current_time * 0.1  # noqa
    )  # 10% of the displayed time in heatmap

    x_start = heatmap_pos.x1 - scalebar_length
    x_end = heatmap_pos.x1
    y_position = heatmap_pos.y0

    fig.lines.append(
        plt.Line2D(
            [x_start, x_end],
            [y_position - 0.03, y_position - 0.03],
            transform=fig.transFigure,
            color="white",
            linewidth=2,
            solid_capstyle="butt",
        )
    )

    fig.text(
        x=(x_start + x_end) / 2,
        y=y_position - 0.045,  # slightly below the scalebar
        s=f"{scalebar_duration:.0f} s",
        ha="center",
        va="top",
        color="white",
        fontsize=6,
    )

    axins = fig.add_axes(
        [  # noqa
            heatmap_pos.x0,  # exactly aligned with heatmap's left edge
            heatmap_pos.y0 - 0.03,  # slightly below the heatmap
            heatmap_pos.width * 0.1,  # 20% width of heatmap
            0.015,  # height of the colorbar
        ]
    )

    cbar = fig.colorbar(img, cax=axins, orientation="horizontal", ticks=[vmin, vmax])
    cbar.ax.tick_params(labelsize=5, colors="white", pad=2)
    cbar.outline.set_edgecolor("white")  # noqa

    fig.text(
        heatmap_pos.x0,
        heatmap_pos.y0 - 0.1,  # below the colorbar with spacing
        "z-scored",
        ha="left",
        va="top",
        color="white",
        fontsize=6,
    )

    scalebar_neurons = int(0.1 * current_neurons)

    x_position = heatmap_pos.x1 + 0.01  # slightly right of heatmap
    y_start = heatmap_pos.y0
    y_end = y_start + (heatmap_pos.height * scalebar_neurons / current_neurons)

    line = plt.Line2D(
        [x_position, x_position],
        [y_start, y_end],
        transform=fig.transFigure,
        color="white",
        linewidth=2,
    )
    line.set_figure(fig)
    fig.lines.append(line)

    ntype = "neurons" if scalebar_neurons == 1 else "neurons"
    fig.text(
        x=x_position + 0.008,
        y=y_start,
        s=f"{scalebar_neurons} {ntype}",
        ha="left",
        va="bottom",
        color="white",
        fontsize=6,
        rotation=90,
    )

    if fig_text is None:
        n_clusters = getattr(model, "n_clusters", None)
        n_PCs = getattr(model, "n_PCs", None)
        locality = getattr(model, "locality", None)
        parts = [f"Neurons: {spks.shape[0]}", f"Superneurons: {sn.shape[0]}"]
        if n_clusters is not None:
            parts.append(f"n_clusters: {n_clusters}")
        if n_PCs is not None:
            parts.append(f"n_PCs: {n_PCs}")
        if locality is not None:
            parts.append(f"locality: {locality}")
        fig_text = ", ".join(parts)

    fig.text(
        x=(heatmap_pos.x0 + heatmap_pos.x1) / 2,
        y=y_start - 0.085,  # vertically between existing scalebars
        s=fig_text,
        ha="center",
        va="top",
        color="white",
        fontsize=6,
    )

    if title is not None:
        plt.suptitle(title, **title_kwargs)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, facecolor="black", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def plot_multiplane_masks(
    suite2p_path: str | Path,
    stat: np.ndarray,
    iscell: np.ndarray,
    nrows: int = None,
    ncols: int = None,
    figsize: tuple = None,
    save_path: str | Path = None,
    cmap: str = "gray",
) -> plt.Figure:
    """
    Plot ROI masks from all planes in a publication-quality grid layout.

    Creates a multi-panel figure showing detected ROIs overlaid on reference images
    for each z-plane, with accepted cells in green and rejected cells in red.
    Background image is selected based on anatomical_only setting.

    Parameters
    ----------
    suite2p_path : str or Path
        Path to suite2p directory containing plane folders (e.g., zplane01/).
    stat : np.ndarray
        Consolidated stat array with 'iplane' field indicating plane assignment.
    iscell : np.ndarray
        Cell classification array (n_rois, 2) where column 0 is binary classification.
    nrows : int, optional
        Number of rows in the figure grid. Auto-calculated if None.
    ncols : int, optional
        Number of columns in the figure grid. Auto-calculated if None.
    figsize : tuple, optional
        Figure size in inches (width, height). Auto-calculated if None.
    save_path : str or Path, optional
        If provided, save figure to this path. Otherwise display interactively.
    cmap : str, default "gray"
        Colormap for background images.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    from scipy import ndimage

    suite2p_path = Path(suite2p_path)
    # supports zplaneNN, planeNN, plane*_stitched
    plane_dirs = sorted(suite2p_path.glob("zplane*"))
    if not plane_dirs:
        plane_dirs = sorted(suite2p_path.glob("plane*"))
    nplanes = len(plane_dirs)

    if nplanes == 0:
        fig = plt.figure(figsize=(8, 6), facecolor="black")
        fig.text(0.5, 0.5, "No plane directories found", ha="center", va="center",
                fontsize=14, color="white")
        return fig

    # auto-calculate grid size
    if ncols is None:
        ncols = min(5, nplanes)
    if nrows is None:
        nrows = int(np.ceil(nplanes / ncols))

    # auto-calculate figure size (make panels ~5 inches each for better visibility)
    if figsize is None:
        figsize = (ncols * 5, nrows * 5)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, facecolor="black",
        gridspec_kw={"wspace": 0.02, "hspace": 0.08}
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, plane_dir in enumerate(plane_dirs):
        if idx >= len(axes):
            break

        ax = axes[idx]
        ax.set_facecolor("black")

        # extract plane number from directory name for display
        plane_name = plane_dir.name
        digits = "".join(filter(str.isdigit, plane_name))
        plane_num = int(digits) if digits else idx + 1

        # load plane ops
        ops_file = plane_dir / "ops.npy"
        yshift, xshift = 0, 0  # coordinate shifts for cropped images
        if ops_file.exists():
            plane_ops = np.load(ops_file, allow_pickle=True).item()
            Ly = plane_ops.get("Ly", 512)
            Lx = plane_ops.get("Lx", 512)

            # get crop ranges from registration
            yrange = plane_ops.get("yrange", [0, Ly])
            xrange = plane_ops.get("xrange", [0, Lx])

            # select background image based on anatomical_only
            anatomical_only = plane_ops.get("anatomical_only", 0)
            if anatomical_only >= 4:
                # max projection for anatomical mode (cropped space)
                img = plane_ops.get("max_proj", None)
                if img is not None:
                    yshift, xshift = int(yrange[0]), int(xrange[0])
                else:
                    img = plane_ops.get("meanImg")
            elif anatomical_only == 0:
                # Vcorr for functional imaging (cropped space)
                img = plane_ops.get("Vcorr", None)
                if img is not None:
                    yshift, xshift = int(yrange[0]), int(xrange[0])
                else:
                    img = plane_ops.get("meanImg")
            else:
                # meanImg for other modes (full space)
                img = plane_ops.get("meanImg", plane_ops.get("meanImgE"))

            if img is None:
                img = np.zeros((Ly, Lx))
        else:
            img = np.zeros((512, 512))
            Ly, Lx = 512, 512

        # display background image with proper contrast
        img_h, img_w = img.shape[:2]
        vmin, vmax = np.nanpercentile(img, [1, 99.5])
        ax.imshow(img, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

        # get ROIs for this plane (iplane is 0-indexed from enumeration)
        plane_mask = np.array([s.get("iplane", 0) == idx for s in stat])
        plane_stat = stat[plane_mask]
        plane_iscell = iscell[plane_mask]

        # create mask images for accepted and rejected cells
        accepted_mask = np.zeros((img_h, img_w), dtype=bool)
        rejected_mask = np.zeros((img_h, img_w), dtype=bool)

        accepted_idx = plane_iscell[:, 0] == 1
        rejected_idx = plane_iscell[:, 0] == 0

        for s in plane_stat[accepted_idx]:
            # shift coordinates from full to cropped space
            ypix = s["ypix"] - yshift
            xpix = s["xpix"] - xshift
            valid = (ypix >= 0) & (ypix < img_h) & (xpix >= 0) & (xpix < img_w)
            accepted_mask[ypix[valid], xpix[valid]] = True

        for s in plane_stat[rejected_idx]:
            # shift coordinates from full to cropped space
            ypix = s["ypix"] - yshift
            xpix = s["xpix"] - xshift
            valid = (ypix >= 0) & (ypix < img_h) & (xpix >= 0) & (xpix < img_w)
            rejected_mask[ypix[valid], xpix[valid]] = True

        # compute outlines
        acc_outline = ndimage.binary_dilation(accepted_mask) & ~accepted_mask
        rej_outline = ndimage.binary_dilation(rejected_mask) & ~rejected_mask

        # create rgba overlay
        overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)

        # accepted cells: green fill with outline
        overlay[accepted_mask, :] = [0.2, 0.8, 0.2, 0.3]  # green fill
        overlay[acc_outline, :] = [0.4, 1.0, 0.4, 0.9]  # bright green outline

        # rejected cells: red fill with outline
        overlay[rejected_mask, :] = [0.8, 0.2, 0.2, 0.2]  # red fill
        overlay[rej_outline, :] = [1.0, 0.3, 0.3, 0.6]  # red outline

        ax.imshow(overlay)

        n_acc = accepted_idx.sum()
        n_rej = rejected_idx.sum()

        # title with plane info (white on black)
        ax.set_title(
            f"Plane {plane_num:02d}  ({n_acc}/{n_rej})",
            fontsize=14, fontweight="bold", color="white", pad=6
        )
        ax.axis("off")

    # hide unused subplots
    for idx in range(nplanes, len(axes)):
        axes[idx].set_visible(False)

    # add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.2, 0.8, 0.2, 0.5), edgecolor=(0.4, 1.0, 0.4, 1.0),
              linewidth=2, label="Accepted"),
        Patch(facecolor=(0.8, 0.2, 0.2, 0.3), edgecolor=(1.0, 0.3, 0.3, 1.0),
              linewidth=2, label="Rejected"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=2,
        fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.01),
        labelcolor="white"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="black")
        plt.close(fig)

    return fig


def plot_plane_quality_metrics(
    stat: np.ndarray,
    iscell: np.ndarray,
    save_path: str | Path = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Generate publication-quality ROI quality metrics across all planes.

    Creates a multi-panel figure with line plots showing mean ± std:
    - Compactness vs plane
    - Skewness vs plane
    - ROI size (npix) vs plane
    - Radius vs plane

    Parameters
    ----------
    stat : np.ndarray
        Consolidated stat array with 'iplane', 'compact', 'npix' fields.
    iscell : np.ndarray
        Cell classification array (n_rois, 2).
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple, default (14, 10)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> stat = np.load("merged/stat.npy", allow_pickle=True)
    >>> iscell = np.load("merged/iscell.npy")
    >>> fig = plot_plane_quality_metrics(stat, iscell, save_path="quality.png")
    """
    # Extract metrics
    plane_nums = np.array([s.get("iplane", 0) for s in stat])
    unique_planes = np.unique(plane_nums)
    n_planes = len(unique_planes)

    compactness = np.array([s.get("compact", np.nan) for s in stat])
    skewness = np.array([s.get("skew", np.nan) for s in stat])
    npix = np.array([s.get("npix", 0) for s in stat])
    radius = np.array([s.get("radius", np.nan) for s in stat])
    accepted = iscell[:, 0] == 1

    # Dark theme colors (consistent with plot_volume_diagnostics)
    bg_color = "black"
    text_color = "white"
    colors = {
        "compactness": "#9b59b6",  # Purple
        "skewness": "#e67e22",     # Orange
        "size": "#3498db",         # Blue
        "radius": "#2ecc71",       # Green
    }

    # Compute mean and std per plane for accepted cells
    def compute_stats_per_plane(values, plane_nums, accepted, unique_planes):
        means = []
        stds = []
        for p in unique_planes:
            mask = (plane_nums == p) & accepted & ~np.isnan(values)
            if mask.sum() > 0:
                means.append(np.mean(values[mask]))
                stds.append(np.std(values[mask]))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        return np.array(means), np.array(stds)

    compact_mean, compact_std = compute_stats_per_plane(compactness, plane_nums, accepted, unique_planes)
    skew_mean, skew_std = compute_stats_per_plane(skewness, plane_nums, accepted, unique_planes)
    npix_mean, npix_std = compute_stats_per_plane(npix.astype(float), plane_nums, accepted, unique_planes)
    radius_mean, radius_std = compute_stats_per_plane(radius, plane_nums, accepted, unique_planes)

    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=bg_color)
        axes = axes.flatten()

        x = np.arange(n_planes)

        def style_axis(ax, xlabel, ylabel, title):
            ax.set_facecolor(bg_color)
            ax.set_xlabel(xlabel, fontweight="bold", fontsize=10, color=text_color)
            ax.set_ylabel(ylabel, fontweight="bold", fontsize=10, color=text_color)
            ax.set_title(title, fontweight="bold", fontsize=11, color=text_color)
            ax.tick_params(colors=text_color, labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(text_color)
            ax.spines["left"].set_color(text_color)
            # Set x-ticks to show plane numbers
            if n_planes <= 20:
                ax.set_xticks(x)
                ax.set_xticklabels([f"{int(p)}" for p in unique_planes])
            else:
                step = max(1, n_planes // 10)
                ax.set_xticks(x[::step])
                ax.set_xticklabels([f"{int(p)}" for p in unique_planes[::step]])

        # Panel 1: Compactness
        ax = axes[0]
        valid = ~np.isnan(compact_mean)
        ax.fill_between(x[valid], (compact_mean - compact_std)[valid], (compact_mean + compact_std)[valid],
                       alpha=0.3, color=colors["compactness"])
        ax.plot(x[valid], compact_mean[valid], 'o-', color=colors["compactness"], linewidth=2, markersize=5)
        style_axis(ax, "Z-Plane", "Compactness", "ROI Compactness (Accepted)")

        # Panel 2: Skewness
        ax = axes[1]
        valid = ~np.isnan(skew_mean)
        ax.fill_between(x[valid], (skew_mean - skew_std)[valid], (skew_mean + skew_std)[valid],
                       alpha=0.3, color=colors["skewness"])
        ax.plot(x[valid], skew_mean[valid], 'o-', color=colors["skewness"], linewidth=2, markersize=5)
        style_axis(ax, "Z-Plane", "Skewness", "Trace Skewness (Accepted)")

        # Panel 3: ROI Size (npix)
        ax = axes[2]
        valid = ~np.isnan(npix_mean)
        ax.fill_between(x[valid], (npix_mean - npix_std)[valid], (npix_mean + npix_std)[valid],
                       alpha=0.3, color=colors["size"])
        ax.plot(x[valid], npix_mean[valid], 'o-', color=colors["size"], linewidth=2, markersize=5)
        style_axis(ax, "Z-Plane", "Number of Pixels", "ROI Size (Accepted)")

        # Panel 4: Radius
        ax = axes[3]
        valid = ~np.isnan(radius_mean)
        ax.fill_between(x[valid], (radius_mean - radius_std)[valid], (radius_mean + radius_std)[valid],
                       alpha=0.3, color=colors["radius"])
        ax.plot(x[valid], radius_mean[valid], 'o-', color=colors["radius"], linewidth=2, markersize=5)
        style_axis(ax, "Z-Plane", "Radius (pixels)", "ROI Radius (Accepted)")

        # Main title
        total_accepted = int(accepted.sum())
        total_rois = len(stat)
        fig.suptitle(
            f"Volume Quality Metrics: {total_accepted} accepted / {total_rois} total ROIs",
            fontsize=12, fontweight="bold", color=text_color, y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg_color)
            plt.close(fig)

    return fig


def plot_trace_analysis(
    F: np.ndarray,
    Fneu: np.ndarray,
    stat: np.ndarray,
    iscell: np.ndarray,
    ops: dict,
    save_path: str | Path = None,
    figsize: tuple = (16, 14),
) -> Tuple[plt.Figure, dict]:
    """
    Generate trace analysis figure showing extreme examples by quality metrics.

    Creates a 6-panel figure showing example ΔF/F traces for:
    - Highest SNR / Lowest SNR
    - Lowest shot noise / Highest shot noise
    - Highest skewness / Lowest skewness

    Parameters
    ----------
    F : np.ndarray
        Fluorescence traces array (n_rois, n_frames).
    Fneu : np.ndarray
        Neuropil fluorescence array (n_rois, n_frames).
    stat : np.ndarray
        Stat array with 'iplane' and 'skew' fields.
    iscell : np.ndarray
        Cell classification array.
    ops : dict
        Ops dictionary with 'fs' (frame rate) field.
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple, default (16, 14)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    metrics : dict
        Dictionary containing computed metrics (snr, shot_noise, skewness, dff).

    Examples
    --------
    >>> fig, metrics = plot_trace_analysis(F, Fneu, stat, iscell, ops)
    >>> print(f"Mean SNR: {np.mean(metrics['snr']):.2f}")
    """
    accepted = iscell[:, 0] == 1
    n_accepted = int(np.sum(accepted))

    if n_accepted == 0:
        fig = plt.figure(figsize=figsize, facecolor="black")
        fig.text(0.5, 0.5, "No accepted ROIs found", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")
        return fig, {}

    F_acc = F[accepted]
    Fneu_acc = Fneu[accepted]
    stat_acc = stat[accepted]
    plane_nums = np.array([s.get("iplane", 0) for s in stat_acc])
    fs = ops.get("fs", 30.0)

    # static-baseline ΔF/F for quality metrics only (see baseline_percentile_dff)
    F_corrected = F_acc - 0.7 * Fneu_acc
    dff = baseline_percentile_dff(F_corrected)

    # Compute metrics
    # SNR: signal / noise
    signal = np.std(dff, axis=1)
    noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.6745  # MAD estimator
    snr = signal / (noise + 1e-6)

    # Shot noise: noise level (MAD of diff)
    shot_noise = noise

    # Skewness: from stat or compute from trace
    skewness = np.array([s.get("skew", np.nan) for s in stat_acc])
    # Fill NaNs with computed skewness if needed
    nan_mask = np.isnan(skewness)
    if nan_mask.any():
        from scipy.stats import skew as scipy_skew
        for i in np.where(nan_mask)[0]:
            skewness[i] = scipy_skew(dff[i])

    # Style configuration
    bg_color = "black"
    text_color = "white"

    # Colors for each metric type
    colors = {
        "snr_high": "#2ecc71",      # Green - good
        "snr_low": "#e74c3c",       # Red - bad
        "noise_low": "#3498db",     # Blue - good
        "noise_high": "#e67e22",    # Orange - bad
        "skew_high": "#9b59b6",     # Purple - high activity
        "skew_low": "#95a5a6",      # Gray - low activity
    }

    # Find indices for each category
    valid_mask = ~np.isnan(snr) & ~np.isnan(shot_noise) & ~np.isnan(skewness)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        fig = plt.figure(figsize=figsize, facecolor="black")
        fig.text(0.5, 0.5, "No valid ROIs with computed metrics", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")
        return fig, {}

    # Get indices for extremes
    snr_valid = snr[valid_mask]
    noise_valid = shot_noise[valid_mask]
    skew_valid = skewness[valid_mask]

    idx_snr_high = valid_idx[np.argmax(snr_valid)]
    idx_snr_low = valid_idx[np.argmin(snr_valid)]
    idx_noise_low = valid_idx[np.argmin(noise_valid)]
    idx_noise_high = valid_idx[np.argmax(noise_valid)]
    idx_skew_high = valid_idx[np.argmax(skew_valid)]
    idx_skew_low = valid_idx[np.argmin(skew_valid)]

    # Time axis - show up to 100s or full trace
    n_frames_show = min(int(100 * fs), dff.shape[1])
    time = np.arange(n_frames_show) / fs

    with plt.style.context("default"):
        fig = plt.figure(figsize=figsize, facecolor=bg_color)
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.25,
                              left=0.08, right=0.95, top=0.92, bottom=0.06)

        def plot_trace_panel(ax, idx, title, color, metric_name, metric_val):
            """Plot a single trace panel."""
            ax.set_facecolor(bg_color)
            trace = dff[idx, :n_frames_show]
            ax.plot(time, trace, color=color, linewidth=0.8, alpha=0.9)

            # Add zero line
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            # Get plane info
            plane = plane_nums[idx]

            # Style
            ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold", color=text_color)
            ax.set_ylabel("ΔF/F", fontsize=10, fontweight="bold", color=text_color)
            ax.set_title(f"{title}\n{metric_name}={metric_val:.2f}, Plane {plane}",
                        fontsize=11, fontweight="bold", color=text_color)
            ax.tick_params(colors=text_color, labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(text_color)
            ax.spines["left"].set_color(text_color)

            # Set reasonable y-limits
            y_max = np.percentile(trace, 99.5)
            y_min = np.percentile(trace, 0.5)
            margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)

        # Row 1: SNR extremes
        ax1 = fig.add_subplot(gs[0, 0])
        plot_trace_panel(ax1, idx_snr_high, "Highest SNR", colors["snr_high"],
                        "SNR", snr[idx_snr_high])

        ax2 = fig.add_subplot(gs[0, 1])
        plot_trace_panel(ax2, idx_snr_low, "Lowest SNR", colors["snr_low"],
                        "SNR", snr[idx_snr_low])

        # Row 2: Shot noise extremes
        ax3 = fig.add_subplot(gs[1, 0])
        plot_trace_panel(ax3, idx_noise_low, "Lowest Shot Noise", colors["noise_low"],
                        "Noise", shot_noise[idx_noise_low])

        ax4 = fig.add_subplot(gs[1, 1])
        plot_trace_panel(ax4, idx_noise_high, "Highest Shot Noise", colors["noise_high"],
                        "Noise", shot_noise[idx_noise_high])

        # Row 3: Skewness extremes
        ax5 = fig.add_subplot(gs[2, 0])
        plot_trace_panel(ax5, idx_skew_high, "Highest Skewness", colors["skew_high"],
                        "Skew", skewness[idx_skew_high])

        ax6 = fig.add_subplot(gs[2, 1])
        plot_trace_panel(ax6, idx_skew_low, "Lowest Skewness", colors["skew_low"],
                        "Skew", skewness[idx_skew_low])

        # Main title with summary stats
        fig.suptitle(
            f"Trace Quality Extremes: {n_accepted} accepted ROIs | "
            f"SNR: {np.nanmedian(snr):.1f} (median) | "
            f"Noise: {np.nanmedian(shot_noise):.3f} (median)",
            fontsize=12, fontweight="bold", color=text_color, y=0.98
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg_color)
            plt.close(fig)

    metrics = {
        "snr": snr,
        "shot_noise": shot_noise,
        "skewness": skewness,
        "dff": dff,
    }
    return fig, metrics


def create_volume_summary_table(
    stat: np.ndarray,
    iscell: np.ndarray,
    F: np.ndarray = None,
    Fneu: np.ndarray = None,
    ops: dict = None,
    save_path: str | Path = None,
) -> pd.DataFrame:
    """
    Generates per-plane and aggregate statistics including ROI counts,
    SNR metrics, and quality measures.

    Parameters
    ----------
    stat : np.ndarray
        Consolidated stat array with plane assignments.
    iscell : np.ndarray
        Cell classification array.
    F : np.ndarray, optional
        Fluorescence traces for SNR calculation.
    Fneu : np.ndarray, optional
        Neuropil traces for SNR calculation.
    ops : dict, optional
        Ops dictionary with frame rate.
    save_path : str or Path, optional
        If provided, save CSV to this path.

    Returns
    -------
    df : pd.DataFrame
        Summary statistics table.

    Examples
    --------
    >>> df = create_volume_summary_table(stat, iscell, F, Fneu, ops)
    >>> print(df.to_string())
    """
    accepted = iscell[:, 0] == 1
    plane_nums = np.array([s.get("iplane", 0) for s in stat])
    unique_planes = np.unique(plane_nums)

    # Compute SNR if traces provided
    snr = None
    mean_F_arr = None
    if F is not None and Fneu is not None:
        F_acc = F[accepted]
        Fneu_acc = Fneu[accepted]
        # static-baseline ΔF/F for quality metrics only (see baseline_percentile_dff)
        F_corrected = F_acc - 0.7 * Fneu_acc
        dff = baseline_percentile_dff(F_corrected)
        signal = np.std(dff, axis=1)
        noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.6745
        snr = signal / (noise + 1e-6)
        mean_F_arr = np.mean(F_acc, axis=1)
        plane_nums_acc = plane_nums[accepted]
    else:
        plane_nums_acc = plane_nums[accepted]

    # Extract metrics
    compactness = np.array([s.get("compact", np.nan) for s in stat])
    npix = np.array([s.get("npix", 0) for s in stat])

    summary_data = []
    for p in unique_planes:
        plane_mask = plane_nums == p
        plane_mask_acc = plane_nums_acc == p if snr is not None else plane_mask & accepted

        n_total = plane_mask.sum()
        n_accepted = (plane_mask & accepted).sum()

        row = {
            "Plane": int(p),
            "Total_ROIs": int(n_total),
            "Accepted": int(n_accepted),
            "Rejected": int(n_total - n_accepted),
            "Accept_Rate_%": f"{100 * n_accepted / max(1, n_total):.1f}",
            "Mean_Compact": f"{np.nanmean(compactness[plane_mask & accepted]):.2f}",
            "Mean_Size_px": f"{np.mean(npix[plane_mask & accepted]):.0f}",
        }

        if snr is not None and plane_mask_acc.sum() > 0:
            row["Mean_SNR"] = f"{np.mean(snr[plane_mask_acc]):.2f}"
            row["Median_SNR"] = f"{np.median(snr[plane_mask_acc]):.2f}"
            row["High_SNR_%"] = f"{100 * np.sum(snr[plane_mask_acc] > 2) / plane_mask_acc.sum():.1f}"
            row["Mean_F"] = f"{np.mean(mean_F_arr[plane_mask_acc]):.0f}"

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Add totals row
    totals = {
        "Plane": "ALL",
        "Total_ROIs": int(len(stat)),
        "Accepted": int(accepted.sum()),
        "Rejected": int((~accepted).sum()),
        "Accept_Rate_%": f"{100 * accepted.sum() / len(stat):.1f}",
        "Mean_Compact": f"{np.nanmean(compactness[accepted]):.2f}",
        "Mean_Size_px": f"{np.mean(npix[accepted]):.0f}",
    }
    if snr is not None:
        totals["Mean_SNR"] = f"{np.mean(snr):.2f}"
        totals["Median_SNR"] = f"{np.median(snr):.2f}"
        totals["High_SNR_%"] = f"{100 * np.sum(snr > 2) / len(snr):.1f}"
        totals["Mean_F"] = f"{np.mean(mean_F_arr):.0f}"

    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Summary table saved to: {save_path}")

    return df


def plot_volume_filter_summary(
    suite2p_path: str | Path,
    save_path: str | Path = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Create a volumetric summary figure showing cell filtering across all planes.

    Shows bar chart of accepted/rejected cells per plane, plus summary stats.

    Parameters
    ----------
    suite2p_path : str or Path
        Path to suite2p output directory containing plane subdirectories.
    save_path : str or Path, optional
        Path to save the figure. If None, displays with plt.show().
    figsize : tuple, default (14, 8)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    import matplotlib.pyplot as plt

    suite2p_path = Path(suite2p_path)

    # find plane directories
    plane_dirs = sorted(suite2p_path.glob("plane*"))
    if not plane_dirs:
        # single plane case
        if (suite2p_path / "stat.npy").exists():
            plane_dirs = [suite2p_path]
        else:
            raise ValueError(f"No plane directories or stat.npy found in {suite2p_path}")

    # collect stats per plane
    plane_stats = []
    for pdir in plane_dirs:
        stat_file = pdir / "stat.npy"
        iscell_file = pdir / "iscell.npy"
        iscell_s2p_file = pdir / "iscell_suite2p.npy"

        if not stat_file.exists() or not iscell_file.exists():
            continue

        stat = np.load(stat_file, allow_pickle=True)
        iscell = np.load(iscell_file, allow_pickle=True)
        if iscell.ndim == 2:
            iscell = iscell[:, 0]

        # load suite2p original if exists
        if iscell_s2p_file.exists():
            iscell_s2p = np.load(iscell_s2p_file, allow_pickle=True)
            if iscell_s2p.ndim == 2:
                iscell_s2p = iscell_s2p[:, 0]
        else:
            iscell_s2p = iscell

        n_total = len(stat)
        n_final_accepted = int(iscell.astype(bool).sum())
        n_s2p_accepted = int(iscell_s2p.astype(bool).sum())
        n_s2p_rejected = n_total - n_s2p_accepted
        n_filter_rejected = n_s2p_accepted - n_final_accepted

        # get plane number from dir name
        plane_name = pdir.name
        try:
            plane_num = int(plane_name.replace("plane", ""))
        except ValueError:
            plane_num = len(plane_stats)

        plane_stats.append({
            "plane": plane_num,
            "name": plane_name,
            "n_total": n_total,
            "n_s2p_accepted": n_s2p_accepted,
            "n_s2p_rejected": n_s2p_rejected,
            "n_filter_rejected": n_filter_rejected,
            "n_final_accepted": n_final_accepted,
        })

    if not plane_stats:
        raise ValueError("No valid plane data found")

    # sort by plane number
    plane_stats = sorted(plane_stats, key=lambda x: x["plane"])

    # create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]})

    # left panel: stacked bar chart per plane
    ax = axes[0]
    planes = [p["name"] for p in plane_stats]
    x = np.arange(len(planes))
    width = 0.7

    # stack: final_accepted (green) + filter_rejected (orange) + s2p_rejected (red)
    final_accepted = [p["n_final_accepted"] for p in plane_stats]
    filter_rejected = [p["n_filter_rejected"] for p in plane_stats]
    s2p_rejected = [p["n_s2p_rejected"] for p in plane_stats]

    ax.bar(x, final_accepted, width, label="accepted", color="#33a02c")
    ax.bar(x, filter_rejected, width, bottom=final_accepted,
           label="filter rejected", color="#ff7f00")
    ax.bar(x, s2p_rejected, width,
           bottom=[f + r for f, r in zip(final_accepted, filter_rejected)],
           label="suite2p rejected", color="#e31a1c")

    ax.set_xlabel("plane", fontsize=11)
    ax.set_ylabel("ROI count", fontsize=11)
    ax.set_title("ROI filtering per plane", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(planes, rotation=45, ha="right")
    ax.legend(loc="upper right")

    # right panel: summary pie chart and stats
    ax2 = axes[1]

    total_final = sum(final_accepted)
    total_filter_rej = sum(filter_rejected)
    total_s2p_rej = sum(s2p_rejected)
    total_all = total_final + total_filter_rej + total_s2p_rej

    # pie chart
    sizes = [total_final, total_filter_rej, total_s2p_rej]
    labels = ["accepted", "filter rejected", "suite2p rejected"]
    colors = ["#33a02c", "#ff7f00", "#e31a1c"]

    # filter out zero values for pie
    nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if nonzero:
        sizes_nz, labels_nz, colors_nz = zip(*nonzero)
        wedges, texts, autotexts = ax2.pie(
            sizes_nz, labels=labels_nz, colors=colors_nz,
            autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*total_all)})",
            startangle=90, textprops={"fontsize": 9}
        )
    ax2.set_title("overall summary", fontsize=12, fontweight="bold")

    # add text summary below pie
    summary_text = (
        f"total ROIs: {total_all}\n"
        f"final accepted: {total_final} ({100*total_final/max(1,total_all):.1f}%)\n"
        f"filter rejected: {total_filter_rej} ({100*total_filter_rej/max(1,total_all):.1f}%)\n"
        f"suite2p rejected: {total_s2p_rej} ({100*total_s2p_rej/max(1,total_all):.1f}%)\n"
        f"planes: {len(plane_stats)}"
    )
    ax2.text(0.5, -0.15, summary_text, transform=ax2.transAxes,
             ha="center", va="top", fontsize=10, family="monospace")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_plane_diagnostics(
    plane_dir: str | Path,
    save_path: str | Path = None,
    figsize: tuple = (16, 14),
    n_examples: int = 4,
) -> plt.Figure:
    """
    Generate a single-figure diagnostic summary for a processed plane.

    Creates a publication-quality figure showing:
    - ROI size distribution (accepted vs rejected)
    - SNR distribution with quality threshold
    - Compactness vs SNR scatter
    - Summary statistics text
    - Zoomed ROI examples: N highest SNR and N lowest SNR cells

    Robust to low/zero cell counts - will display informative messages
    when data is insufficient for certain visualizations.

    Parameters
    ----------
    plane_dir : str or Path
        Path to the plane directory containing ops.npy, stat.npy, etc.
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple, default (16, 14)
        Figure size in inches.
    n_examples : int, default 4
        Number of high/low SNR ROI examples to show.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    plane_dir = Path(plane_dir)

    # Load results
    res = load_planar_results(plane_dir)
    ops = load_ops(plane_dir / "ops.npy")

    stat = res["stat"]
    iscell = res["iscell"]
    F = res["F"]
    Fneu = res["Fneu"]

    # Handle edge case: no ROIs at all
    n_total = len(stat)
    if n_total == 0:
        fig = plt.figure(figsize=figsize, facecolor="black")
        fig.text(0.5, 0.5, "No ROIs detected\n\nCheck detection parameters:\n- threshold_scaling\n- cellprob_threshold\n- diameter",
                ha="center", va="center", fontsize=16, fontweight="bold", color="white")
        plane_name = plane_dir.name
        fig.suptitle(f"Quality Diagnostics: {plane_name}", fontsize=14, fontweight="bold", y=0.98, color="white")
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
            plt.close(fig)
        return fig

    # iscell from load_planar_results is (n_rois, 2): [:, 0] is 0/1, [:, 1] is probability
    accepted = iscell[:, 0].astype(bool)
    n_accepted = int(accepted.sum())
    n_rejected = int((~accepted).sum())

    # Compute metrics for ALL ROIs (not just accepted)
    # static-baseline ΔF/F for quality metrics only (see baseline_percentile_dff)
    F_corr = F - 0.7 * Fneu
    dff = baseline_percentile_dff(F_corr)

    # SNR calculation for all ROIs
    signal = np.std(dff, axis=1)
    noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.6745
    snr = signal / (noise + 1e-6)

    # Extract ROI properties
    npix = np.array([s.get("npix", 0) for s in stat])
    compactness = np.array([s.get("compact", np.nan) for s in stat])
    skewness = np.array([s.get("skew", np.nan) for s in stat])
    fs = ops.get("fs", 30.0)

    # Compute stats with safe defaults
    snr_acc = snr[accepted] if n_accepted > 0 else np.array([np.nan])
    median_snr = np.nanmedian(snr_acc) if n_accepted > 0 else 0.0

    # Get mean image for ROI zoom panels
    mean_img = ops.get("meanImgE", ops.get("meanImg"))

    # Create figure with custom layout - dark background like consolidate.ipynb
    # Row 0: Size dist, SNR dist, SNR vs Compactness, Activity vs SNR (4 panels)
    # Row 1: High SNR ROI zooms (n_examples panels)
    # Row 2: High SNR ROI traces
    # Row 3: Low SNR ROI zooms (n_examples panels)
    # Row 4: Low SNR ROI traces
    fig = plt.figure(figsize=(figsize[0], figsize[1] + 2), facecolor="black")

    # use nested gridspec: top row has more spacing, bottom rows are tight
    # Increased gap between top plots (bottom=0.62) and ROI images (top=0.48)
    # Added 5th row as spacer between high and low SNR groups
    gs_top = gridspec.GridSpec(1, 4, figure=fig, left=0.06, right=0.98, top=0.95, bottom=0.62,
                                wspace=0.35)
    gs_bottom = gridspec.GridSpec(5, max(4, n_examples), figure=fig, left=0.02, right=0.98,
                                   top=0.48, bottom=0.02, hspace=0.02, wspace=0.08,
                                   height_ratios=[1, 0.4, 0.15, 1, 0.4])

    # compute activity metric: number of transients (peaks above 2 std)
    if n_accepted > 0:
        dff_acc = dff[accepted]
        activity = np.sum(dff_acc > 2, axis=1)  # count frames above 2 std
    else:
        activity = np.array([])

    # compute shot noise per ROI (standardized noise metric)
    # shot_noise = median(|diff(dff)|) / sqrt(fs) * 100 (in %/sqrt(Hz))
    frame_diffs = np.abs(np.diff(dff, axis=1))
    shot_noise = np.median(frame_diffs, axis=1) / np.sqrt(fs) * 100

    # Panel 1: ROI size distribution - use step histogram for clarity
    ax_size = fig.add_subplot(gs_top[0, 0])
    ax_size.set_facecolor("black")

    all_npix = npix[npix > 0]
    if len(all_npix) > 0:
        bins = np.linspace(0, np.percentile(all_npix, 99), 40)
        # Use step histograms with distinct line styles for clear separation
        if n_accepted > 0:
            ax_size.hist(npix[accepted], bins=bins, histtype="stepfilled", alpha=0.7,
                        color="#2ecc71", edgecolor="#2ecc71", linewidth=1.5,
                        label=f"Accepted ({n_accepted})")
        if n_rejected > 0:
            ax_size.hist(npix[~accepted], bins=bins, histtype="step",
                        color="#e74c3c", linewidth=2, linestyle="-",
                        label=f"Rejected ({n_rejected})")
        ax_size.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="white", labelcolor="white", loc="upper right")
    else:
        ax_size.text(0.5, 0.5, "No ROI data", ha="center", va="center", fontsize=12, color="white")

    ax_size.set_xlabel("Size (pixels)", fontweight="bold", fontsize=9, color="white")
    ax_size.set_ylabel("Count", fontweight="bold", fontsize=9, color="white")
    ax_size.set_title("ROI Size", fontweight="bold", fontsize=10, color="white")
    ax_size.tick_params(colors="white", labelsize=8)
    ax_size.spines["top"].set_visible(False)
    ax_size.spines["right"].set_visible(False)
    ax_size.spines["bottom"].set_color("white")
    ax_size.spines["left"].set_color("white")

    # Panel 2: SNR distribution - use step histogram for clarity
    ax_snr = fig.add_subplot(gs_top[0, 1])
    ax_snr.set_facecolor("black")

    all_snr = snr[~np.isnan(snr)]
    if len(all_snr) > 0:
        bins = np.linspace(0, np.percentile(all_snr, 99), 40)

        # Filled for accepted, outline for rejected - no overlap confusion
        if n_accepted > 0:
            ax_snr.hist(snr[accepted], bins=bins, histtype="stepfilled", alpha=0.7,
                       color="#2ecc71", edgecolor="#2ecc71", linewidth=1.5,
                       label=f"Accepted ({n_accepted})")
            ax_snr.axvline(median_snr, color="#ffe66d", linestyle="-", linewidth=2,
                        label=f"Median={median_snr:.1f}")
        if n_rejected > 0:
            ax_snr.hist(snr[~accepted], bins=bins, histtype="step",
                       color="#e74c3c", linewidth=2, linestyle="-",
                       label=f"Rejected ({n_rejected})")

        ax_snr.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="white", labelcolor="white", loc="upper right")
    else:
        ax_snr.text(0.5, 0.5, "No SNR data", ha="center", va="center", fontsize=12, color="white")

    ax_snr.set_xlabel("SNR", fontweight="bold", fontsize=9, color="white")
    ax_snr.set_ylabel("Count", fontweight="bold", fontsize=9, color="white")
    ax_snr.set_title("SNR Distribution", fontweight="bold", fontsize=10, color="white")
    ax_snr.tick_params(colors="white", labelsize=8)
    ax_snr.spines["top"].set_visible(False)
    ax_snr.spines["right"].set_visible(False)
    ax_snr.spines["bottom"].set_color("white")
    ax_snr.spines["left"].set_color("white")

    # Panels 3 & 4: Compactness vs SNR and Activity vs SNR (shared Y-axis = SNR)
    # Color by skewness (activity pattern quality metric)
    ax_compact = fig.add_subplot(gs_top[0, 2])
    ax_activity = fig.add_subplot(gs_top[0, 3], sharey=ax_compact)
    ax_compact.set_facecolor("black")
    ax_activity.set_facecolor("black")

    has_scatter_data = False
    if n_accepted > 0:
        valid_compact = accepted & ~np.isnan(compactness) & ~np.isnan(skewness)
        valid_activity = accepted & ~np.isnan(skewness)
        snr_acc = snr[accepted]
        skew_acc = skewness[accepted]

        # Get shared color limits from skewness (more informative than SNR for color)
        valid_skew = skew_acc[~np.isnan(skew_acc)]
        if len(valid_skew) > 0:
            vmin, vmax = np.nanpercentile(valid_skew, [5, 95])
        else:
            vmin, vmax = 0, 1

        if valid_compact.sum() > 0:
            # Panel 3: Compactness vs SNR (SNR on y-axis)
            ax_compact.scatter(compactness[valid_compact], snr[valid_compact],
                           c=skewness[valid_compact], cmap="plasma", alpha=0.7, s=20,
                           vmin=vmin, vmax=vmax)
            has_scatter_data = True

        if len(activity) > 0 and valid_activity.sum() > 0:
            # Panel 4: Activity vs SNR (SNR on y-axis)
            sc2 = ax_activity.scatter(activity, snr_acc, c=skew_acc, cmap="plasma",
                           alpha=0.7, s=20, vmin=vmin, vmax=vmax)

            # Add single colorbar for both plots (attached to activity plot)
            cbar = plt.colorbar(sc2, ax=ax_activity, shrink=0.8)
            cbar.set_label("Skewness", fontsize=8, color="white")
            cbar.ax.yaxis.set_tick_params(color="white")
            cbar.outline.set_edgecolor("white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    if not has_scatter_data:
        ax_compact.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12, color="white")
        ax_activity.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12, color="white")

    ax_compact.set_xlabel("Compactness", fontweight="bold", fontsize=9, color="white")
    ax_compact.set_ylabel("SNR", fontweight="bold", fontsize=9, color="white")
    ax_compact.set_title("Compactness vs SNR", fontweight="bold", fontsize=10, color="white")
    ax_compact.tick_params(colors="white", labelsize=8)
    ax_compact.spines["top"].set_visible(False)
    ax_compact.spines["right"].set_visible(False)
    ax_compact.spines["bottom"].set_color("white")
    ax_compact.spines["left"].set_color("white")

    ax_activity.set_xlabel("Active Frames", fontweight="bold", fontsize=9, color="white")
    ax_activity.set_ylabel("SNR", fontweight="bold", fontsize=9, color="white")
    ax_activity.set_title("Activity vs SNR", fontweight="bold", fontsize=10, color="white")
    ax_activity.tick_params(colors="white", labelsize=8)
    # Hide y-axis labels on right plot since it shares y-axis with left
    plt.setp(ax_activity.get_yticklabels(), visible=False)
    ax_activity.spines["top"].set_visible(False)
    ax_activity.spines["right"].set_visible(False)
    ax_activity.spines["bottom"].set_color("white")
    ax_activity.spines["left"].set_color("white")

    # Helper function to plot zoomed ROI
    def plot_roi_zoom(ax, roi_idx, img, stat_entry, snr_val, noise_val, color):
        """Plot a zoomed view of a single ROI with SNR and shot noise."""
        ax.set_facecolor("black")
        ypix = stat_entry["ypix"]
        xpix = stat_entry["xpix"]

        # Calculate bounding box with padding
        pad = 15
        y_min, y_max = max(0, ypix.min() - pad), min(img.shape[0], ypix.max() + pad)
        x_min, x_max = max(0, xpix.min() - pad), min(img.shape[1], xpix.max() + pad)

        # Extract ROI region
        roi_img = img[y_min:y_max, x_min:x_max]

        if roi_img.size == 0:
            ax.text(0.5, 0.5, "No image", ha="center", va="center", fontsize=10, color="white")
            ax.axis("off")
            return

        vmin, vmax = np.nanpercentile(roi_img, [1, 99])
        ax.imshow(roi_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

        # Draw ROI outline (shifted to local coordinates)
        local_y = ypix - y_min
        local_x = xpix - x_min
        ax.scatter(local_x, local_y, c=color, s=3, alpha=0.7, linewidths=0)

        # Title with SNR and shot noise
        ax.set_title(f"#{roi_idx} SNR={snr_val:.1f} σ={noise_val:.2f}", fontsize=7, fontweight="bold", color=color)
        ax.axis("off")

    # Helper function to plot a trace snippet
    def plot_roi_trace(ax, trace, color, window_frames=500):
        """Plot a short trace snippet with shrunk Y axis."""
        ax.set_facecolor("black")
        # show first N frames or all if shorter
        n_show = min(window_frames, len(trace))
        trace_segment = trace[:n_show]
        ax.plot(trace_segment, color=color, linewidth=0.8, alpha=0.9)
        ax.set_xlim(0, n_show)
        # Shrink Y axis to 5th-95th percentile to reduce whitespace
        if len(trace_segment) > 0:
            y_lo, y_hi = np.nanpercentile(trace_segment, [5, 95])
            y_range = y_hi - y_lo
            if y_range > 0:
                ax.set_ylim(y_lo - 0.1 * y_range, y_hi + 0.1 * y_range)
        ax.axis("off")

    # Row 0-1: High SNR ROI examples with traces
    if n_accepted > 0 and mean_img is not None:
        accepted_idx = np.where(accepted)[0]
        snr_accepted = snr[accepted]
        n_show = min(n_examples, n_accepted)

        # Get indices of highest SNR cells
        top_snr_order = np.argsort(snr_accepted)[::-1][:n_show]

        for i in range(n_examples):
            # ROI image
            ax = fig.add_subplot(gs_bottom[0, i])
            if i < n_show:
                local_idx = top_snr_order[i]
                global_idx = accepted_idx[local_idx]
                plot_roi_zoom(ax, global_idx, mean_img, stat[global_idx],
                             snr[global_idx], shot_noise[global_idx], "#2ecc71")
                # trace below
                ax_trace = fig.add_subplot(gs_bottom[1, i])
                plot_roi_trace(ax_trace, dff[global_idx], "#2ecc71")
            else:
                ax.set_facecolor("black")
                ax.axis("off")
                ax_trace = fig.add_subplot(gs_bottom[1, i])
                ax_trace.set_facecolor("black")
                ax_trace.axis("off")

        # Row 2 is spacer (empty)

        # Row 3-4: Low SNR ROI examples with traces
        bottom_snr_order = np.argsort(snr_accepted)[:n_show]

        for i in range(n_examples):
            # ROI image
            ax = fig.add_subplot(gs_bottom[3, i])
            if i < n_show:
                local_idx = bottom_snr_order[i]
                global_idx = accepted_idx[local_idx]
                plot_roi_zoom(ax, global_idx, mean_img, stat[global_idx],
                             snr[global_idx], shot_noise[global_idx], "#ff6b6b")
                # trace below
                ax_trace = fig.add_subplot(gs_bottom[4, i])
                plot_roi_trace(ax_trace, dff[global_idx], "#ff6b6b")
            else:
                ax.set_facecolor("black")
                ax.axis("off")
                ax_trace = fig.add_subplot(gs_bottom[4, i])
                ax_trace.set_facecolor("black")
                ax_trace.axis("off")

    elif n_rejected > 0 and mean_img is not None:
        # Show rejected ROIs for diagnostics
        rejected_idx = np.where(~accepted)[0]
        snr_rejected = snr[~accepted]
        n_show = min(n_examples, n_rejected)

        # High SNR rejected
        top_snr_order = np.argsort(snr_rejected)[::-1][:n_show]
        for i in range(n_examples):
            ax = fig.add_subplot(gs_bottom[0, i])
            if i < n_show:
                local_idx = top_snr_order[i]
                global_idx = rejected_idx[local_idx]
                plot_roi_zoom(ax, global_idx, mean_img, stat[global_idx],
                             snr[global_idx], shot_noise[global_idx], "#ff6b6b")
                ax_trace = fig.add_subplot(gs_bottom[1, i])
                plot_roi_trace(ax_trace, dff[global_idx], "#ff6b6b")
            else:
                ax.set_facecolor("black")
                ax.axis("off")
                ax_trace = fig.add_subplot(gs_bottom[1, i])
                ax_trace.set_facecolor("black")
                ax_trace.axis("off")

        # Row 2 is spacer

        # Low SNR rejected
        bottom_snr_order = np.argsort(snr_rejected)[:n_show]
        for i in range(n_examples):
            ax = fig.add_subplot(gs_bottom[3, i])
            if i < n_show:
                local_idx = bottom_snr_order[i]
                global_idx = rejected_idx[local_idx]
                plot_roi_zoom(ax, global_idx, mean_img, stat[global_idx],
                             snr[global_idx], shot_noise[global_idx], "#ff6b6b")
                ax_trace = fig.add_subplot(gs_bottom[4, i])
                plot_roi_trace(ax_trace, dff[global_idx], "#ff6b6b")
            else:
                ax.set_facecolor("black")
                ax.axis("off")
                ax_trace = fig.add_subplot(gs_bottom[4, i])
                ax_trace.set_facecolor("black")
                ax_trace.axis("off")
    else:
        # No image available
        for row in [0, 1, 3, 4]:  # Skip spacer row 2
            for i in range(n_examples):
                ax = fig.add_subplot(gs_bottom[row, i])
                ax.set_facecolor("black")
                if row in [0, 3]:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8, color="white")
                ax.axis("off")

    # Main title
    plane_name = plane_dir.name
    fig.suptitle(f"Quality Diagnostics: {plane_name}", fontsize=14, fontweight="bold", y=0.98, color="white")

    # No tight_layout - we use manual GridSpec positioning for precise control

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
    else:
        plt.show()

    return fig


def mask_dead_zones_in_ops(ops, threshold=0.01):
    """
    Mask out dead zones from registration shifts in ops image arrays.

    Dead zones appear as very dark regions (near zero intensity) at the edges
    of images after suite3D alignment shifts are applied.

    Parameters
    ----------
    ops : dict
        Suite2p ops dictionary containing image arrays
    threshold : float
        Fraction of max intensity to use as cutoff (default 0.01 = 1%)

    Returns
    -------
    ops : dict
        Modified ops with dead zones set to NaN in image arrays
    """
    if "meanImg" not in ops:
        return ops

    # Use meanImg to identify valid regions
    mean_img = ops["meanImg"]
    valid_mask = mean_img > (mean_img.max() * threshold)
    n_invalid = (~valid_mask).sum()

    if n_invalid > 0:
        pct_invalid = 100 * n_invalid / valid_mask.size
        print(f"[mask_dead_zones] Masking {n_invalid} ({pct_invalid:.1f}%) dead zone pixels")

        # Mask all image arrays in ops
        for key in ["meanImg", "meanImgE", "max_proj", "Vcorr"]:
            if key in ops and isinstance(ops[key], np.ndarray):
                img = ops[key]
                # Only apply mask if shapes match
                if img.shape == valid_mask.shape:
                    # Convert to float and set invalid regions to NaN
                    ops[key] = img.astype(float)
                    ops[key][~valid_mask] = np.nan
                else:
                    print(f"[mask_dead_zones] Skipping {key}: shape {img.shape} != meanImg shape {valid_mask.shape}")

    return ops


def plot_zplane_figures(
    plane_dir, dff_percentile=20, dff_window_size=None, dff_smooth_window=None,
    norm_method="zscore", correct_neuropil=True, run_rastermap=False,
    rastermap_kwargs=None, **kwargs
):
    """
    Re-generate Suite2p figures for a merged plane.

    Parameters
    ----------
    plane_dir : Path
        Path to the planeXX output directory (with ops.npy, stat.npy, etc.).
    dff_percentile : int, optional
        Percentile used for ΔF/F baseline (when norm_method="dff").
    dff_window_size : int, optional
        Window size for ΔF/F rolling baseline. If None, auto-calculated
        as ~10 × tau × fs based on ops values.
    dff_smooth_window : int, optional
        Temporal smoothing window for dF/F traces (in frames).
        If None, auto-calculated as ~0.5 × tau × fs to emphasize
        transients while reducing noise. Set to 1 to disable.
    norm_method : str, default "zscore"
        Normalization used for the norm-trace plots: "dff" or "zscore".
    run_rastermap : bool, optional
        If True, compute and plot rastermap sorting of cells.
    rastermap_kwargs : dict, optional
        Override defaults passed to rastermap.Rastermap(). Merged over
        the cell-count-aware defaults (n_clusters, n_PCs, locality,
        time_lag_window, grid_upsample). Ignored if run_rastermap=False.
    kwargs : dict
        Extra keyword args (e.g. fig_label).
    """
    plane_dir = Path(plane_dir)

    # File naming convention: numbered prefixes ensure proper alphabetical ordering
    # 01_correlation -> 02_max_projection -> 03_mean -> 04_mean_enhanced
    # each image immediately followed by its _segmentation variant
    expected_files = {
        "ops": plane_dir / "ops.npy",
        "stat": plane_dir / "stat.npy",
        "iscell": plane_dir / "iscell.npy",
        # Summary images with segmentation overlays - numbered for proper ordering
        "correlation_image": plane_dir / "01_correlation.png",
        "correlation_segmentation": plane_dir / "01_correlation_segmentation.png",
        "max_proj": plane_dir / "02_max_projection.png",
        "max_proj_segmentation": plane_dir / "02_max_projection_segmentation.png",
        "meanImg": plane_dir / "03_mean.png",
        "meanImg_segmentation": plane_dir / "03_mean_segmentation.png",
        "meanImgE": plane_dir / "04_mean_enhanced.png",
        "meanImgE_segmentation": plane_dir / "04_mean_enhanced_segmentation.png",
        # rejected-cell overlay on the projection used for cellpose detection
        "rejected_segmentation": plane_dir / "04b_rejected_segmentation.png",
        # Diagnostics and analysis
        "quality_diagnostics": plane_dir / "05_quality_diagnostics.png",
        "registration": plane_dir / "06_registration.png",
        # Traces - multiple cell counts
        "traces_raw_20": plane_dir / "07a_traces_raw_20.png",
        "traces_raw_50": plane_dir / "07b_traces_raw_50.png",
        "traces_raw_100": plane_dir / "07c_traces_raw_100.png",
        "traces_norm_20": plane_dir / "08a_traces_norm_20.png",
        "traces_norm_50": plane_dir / "08b_traces_norm_50.png",
        "traces_norm_100": plane_dir / "08c_traces_norm_100.png",
        "traces_rejected": plane_dir / "09_traces_rejected.png",
        # Noise distributions
        "noise_acc": plane_dir / "10_shot_noise_accepted.png",
        "noise_rej": plane_dir / "11_shot_noise_rejected.png",
        # Rastermap
        "model": plane_dir / "model.npy",
        "rastermap": plane_dir / "12_rastermap.png",
        # Regional zoom
        "regional_zoom": plane_dir / "13_regional_zoom.png",
    }

    output_ops = load_ops(expected_files["ops"])

    # force remake of all figures (including segmentation overlays)
    for key in [
        "correlation_image",
        "correlation_segmentation",
        "max_proj",
        "max_proj_segmentation",
        "meanImg",
        "meanImg_segmentation",
        "meanImgE",
        "meanImgE_segmentation",
        "rejected_segmentation",
        "quality_diagnostics",
        "registration",
        "traces_raw_20",
        "traces_raw_50",
        "traces_raw_100",
        "traces_norm_20",
        "traces_norm_50",
        "traces_norm_100",
        "traces_rejected",
        "noise_acc",
        "noise_rej",
        "rastermap",
        "regional_zoom",
    ]:
        if key in expected_files:
            if expected_files[key].exists():
                try:
                    expected_files[key].unlink()
                except PermissionError:
                    print(f"Error: Cannot delete {expected_files[key]}, it's open elsewhere.")

    if expected_files["stat"].is_file():

        res = load_planar_results(plane_dir)
        # iscell is (n_rois, 2): [:, 0] is 0/1, [:, 1] is classifier probability
        iscell_mask = res["iscell"][:, 0].astype(bool)

        spks = res["spks"]
        F = res["F"]
        Fneu = res.get("Fneu") if isinstance(res, dict) else None

        # F (raw) feeds the "raw" trace plots and compute_trace_quality_score
        # (which optionally subtracts Fneu internally). F_for_dff is the input
        # to the norm-trace recompute — neuropil-corrected only when the
        # toggle is on, matching norm_traces.npy.
        if correct_neuropil and Fneu is not None:
            F_for_dff = F - 0.7 * Fneu
        else:
            F_for_dff = F

        # Split by accepted/rejected — F is raw, F_for_dff feeds dff
        F_accepted = F[iscell_mask] if iscell_mask.sum() > 0 else np.zeros((0, F.shape[1]))
        F_rejected = F[~iscell_mask] if (~iscell_mask).sum() > 0 else np.zeros((0, F.shape[1]))
        F_dff_accepted = F_for_dff[iscell_mask] if iscell_mask.sum() > 0 else np.zeros((0, F.shape[1]))
        F_dff_rejected = F_for_dff[~iscell_mask] if (~iscell_mask).sum() > 0 else np.zeros((0, F.shape[1]))
        spks_cells = spks[iscell_mask] if iscell_mask.sum() > 0 else np.zeros((0, spks.shape[1]))

        n_accepted = F_accepted.shape[0]
        n_rejected = F_rejected.shape[0]
        print(f"Plotting results for {n_accepted} accepted / {n_rejected} rejected ROIs")

        # segmentation overlays — generated first so failures below don't block them
        # suite2p stores images in two coordinate systems:
        # - FULL space: refImg, meanImg, meanImgE (same size as original Ly x Lx)
        # - CROPPED space: max_proj, Vcorr (size determined by yrange/xrange after registration)
        # stat coordinates are in FULL image space.

        stat_full = res["stat"]

        def _is_valid_image(img):
            if img is None:
                return False
            if isinstance(img, (int, float)) and img == 0:
                return False
            if isinstance(img, np.ndarray) and img.size == 0:
                return False
            return True

        # plot_masks receives full-frame stat coords and ops+proj_key; it
        # will crop the image to the valid (non-padded) region and map
        # stat coords into the cropped image space. This matches
        # plot_projection's cropping so mask/no-mask figures are identical
        # in size and layout.
        segmentation_images = [
            ("meanImg", "Mean Image", expected_files["meanImg_segmentation"]),
            ("meanImgE", "Enhanced Mean Image", expected_files["meanImgE_segmentation"]),
            ("max_proj", "Max Projection", expected_files["max_proj_segmentation"]),
        ]

        for img_key, title_name, save_file in segmentation_images:
            try:
                img = output_ops.get(img_key)
                if _is_valid_image(img):
                    if n_accepted > 0:
                        plot_masks(
                            img=img,
                            stat=stat_full,
                            mask_idx=iscell_mask,
                            savepath=save_file,
                            title=f"{title_name} - Accepted ROIs (n={n_accepted})",
                            ops=output_ops,
                            proj_key=img_key,
                        )
                    else:
                        plot_projection(
                            output_ops,
                            save_file,
                            fig_label=kwargs.get("fig_label", plane_dir.stem),
                            display_masks=False,
                            add_scalebar=True,
                            proj=img_key,
                        )
            except Exception as e:
                print(f"  Warning: {img_key} segmentation failed: {e}")

        # rejected-cell overlay on the projection actually used for cellpose
        # detection. anatomical_only mapping (from default_ops.py):
        #   0 -> Vcorr (functional sparse mode)
        #   1 -> max_proj / meanImg (combined; we display max_proj as the
        #        closest visualizable proxy since the ratio isn't stored)
        #   2 -> meanImg
        #   3 -> meanImgE
        #   4 -> max_proj
        try:
            if n_rejected > 0:
                anatomical_only = int(output_ops.get("anatomical_only", 0) or 0)
                proj_lookup = {
                    0: ("Vcorr", "Correlation Image"),
                    1: ("max_proj", "Max Projection (max_proj / meanImg)"),
                    2: ("meanImg", "Mean Image"),
                    3: ("meanImgE", "Enhanced Mean Image"),
                    4: ("max_proj", "Max Projection"),
                }
                rej_key, rej_title = proj_lookup.get(anatomical_only, ("meanImg", "Mean Image"))
                rej_img = output_ops.get(rej_key)
                # fallback chain if the chosen projection isn't available
                if not _is_valid_image(rej_img):
                    for fk in ("meanImg", "max_proj", "meanImgE", "Vcorr"):
                        if _is_valid_image(output_ops.get(fk)):
                            rej_key = fk
                            rej_title = {
                                "meanImg": "Mean Image",
                                "max_proj": "Max Projection",
                                "meanImgE": "Enhanced Mean Image",
                                "Vcorr": "Correlation Image",
                            }[fk]
                            rej_img = output_ops.get(fk)
                            break
                if _is_valid_image(rej_img):
                    plot_accepted_rejected_overlay(
                        img=rej_img,
                        stat=stat_full,
                        iscell_mask=iscell_mask,
                        savepath=expected_files["rejected_segmentation"],
                        title=rej_title,
                        ops=output_ops,
                        proj_key=rej_key,
                    )
        except Exception as e:
            print(f"  Warning: rejected segmentation failed: {e}")

        # correlation image (Vcorr) — already cropped by suite2p to yrange/xrange.
        try:
            vcorr = output_ops.get("Vcorr")
            if _is_valid_image(vcorr):
                fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
                ax.set_facecolor("black")
                ax.imshow(vcorr, cmap="gray")
                ax.set_title("Correlation Image", color="white", fontweight="bold")
                ax.axis("off")
                plt.tight_layout()
                plt.savefig(expected_files["correlation_image"], dpi=300, facecolor="black")
                plt.close(fig)

                if n_accepted > 0:
                    plot_masks(
                        img=vcorr,
                        stat=stat_full,
                        mask_idx=iscell_mask,
                        savepath=expected_files["correlation_segmentation"],
                        title=f"Correlation Image - Accepted ROIs (n={n_accepted})",
                        ops=output_ops,
                        proj_key="Vcorr",
                    )
        except Exception as e:
            print(f"  Warning: correlation image failed: {e}")

        # rastermap (optional)
        model = None
        if run_rastermap and n_accepted >= 2:
            try:
                from lbm_suite2p_python.zplane import plot_rastermap
                import rastermap
                has_rastermap = True
            except ImportError:
                print("rastermap not found. Install via: pip install rastermap")
                print("  or: pip install mbo_utilities[rastermap]")
                has_rastermap = False
                rastermap, plot_rastermap = None, None

            if has_rastermap:
                import json
                model_file = expected_files["model"]
                plot_file = expected_files["rastermap"]
                params_file = model_file.with_suffix(".params.json")
                need_recompute = True

                # resolved params include the cell-count-aware defaults plus
                # any user override; cache reuse requires both n_accepted AND
                # this dict to match the cached fit. lets the user re-run
                # with new n_clusters / n_PCs / locality without manually
                # deleting model.npy.
                resolved_params = {
                    "n_clusters": 100 if n_accepted >= 200 else None,
                    "n_PCs": min(128, max(2, n_accepted - 1)),
                    "locality": 0.0 if n_accepted >= 200 else 0.1,
                    "time_lag_window": 15,
                    "grid_upsample": 10 if n_accepted >= 200 else 0,
                }
                if rastermap_kwargs:
                    resolved_params.update(rastermap_kwargs)

                if model_file.is_file():
                    try:
                        cached_model = np.load(model_file, allow_pickle=True).item()
                        if hasattr(cached_model, "isort"):
                            cached_isort = cached_model.isort
                        elif isinstance(cached_model, dict) and "isort" in cached_model:
                            cached_isort = cached_model["isort"]
                        else:
                            cached_isort = None

                        cached_params = None
                        if params_file.is_file():
                            try:
                                with open(params_file) as f:
                                    cached_params = json.load(f)
                            except (OSError, json.JSONDecodeError):
                                cached_params = None

                        if cached_isort is None or len(cached_isort) != n_accepted:
                            cached_len = len(cached_isort) if cached_isort is not None else "?"
                            print(f"  Rastermap model stale (cached {cached_len} vs current {n_accepted} cells), recomputing...")
                            model_file.unlink()
                            params_file.unlink(missing_ok=True)
                        elif cached_params != resolved_params:
                            # params drift → recompute and remake the figure
                            # so the plot reflects the new fit.
                            print(f"  Rastermap params changed (cached={cached_params}, current={resolved_params}); recomputing.")
                            model_file.unlink()
                            params_file.unlink(missing_ok=True)
                        else:
                            model = cached_model
                            need_recompute = False
                            print(f"  Using cached rastermap model ({n_accepted} cells, params match)")
                    except Exception as e:
                        print(f"  Failed to load cached rastermap model: {e}, recomputing...")
                        model_file.unlink(missing_ok=True)
                        params_file.unlink(missing_ok=True)

                if need_recompute:
                    print(f"  Computing rastermap model for {n_accepted} cells with params {resolved_params}...")
                    model = rastermap.Rastermap(**resolved_params).fit(spks_cells)
                    np.save(model_file, model)
                    try:
                        with open(params_file, "w") as f:
                            json.dump(resolved_params, f, indent=2)
                    except OSError as e:
                        print(f"  Warning: could not write rastermap params sidecar: {e}")
                    # force remake of the rastermap PNG so the figure
                    # reflects the new fit (the unconditional unlink at the
                    # top of plot_zplane_figures already covers fresh runs;
                    # this guards a stale png from a prior aborted run that
                    # left model.npy out of sync).
                    if plot_file.is_file():
                        try:
                            plot_file.unlink()
                        except OSError:
                            pass

                if model is not None and not plot_file.is_file():
                    plot_rastermap(
                        spks_cells,
                        model,
                        neuron_bin_size=0,
                        save_path=plot_file,
                        title_kwargs={"fontsize": 8, "y": 0.95},
                        title="Rastermap Sorted Activity",
                    )

                if model is not None:
                    if hasattr(model, "isort"):
                        isort = model.isort
                    elif isinstance(model, dict) and "isort" in model:
                        isort = model["isort"]
                    else:
                        isort = None

                    if isort is not None:
                        isort_global = np.where(iscell_mask)[0][isort]
                        output_ops["isort"] = isort_global
                        F_accepted = F_accepted[isort]

        # trace plots and noise distributions
        try:
            fs = output_ops.get("fs", 1.0)
            tau = output_ops.get("tau", 1.0)

            # resolve auto-calculated window sizes the same way the trace
            # functions do, so the param footer reflects the values used
            _resolved_window = (
                int(dff_window_size) if dff_window_size is not None
                else (int(10 * tau * fs) if (fs and tau) else 300)
            )
            _resolved_smooth = (
                int(dff_smooth_window) if dff_smooth_window is not None
                else (max(1, int(0.5 * tau * fs)) if (fs and tau) else 1)
            )
            # method-aware labels for the norm-trace plots
            if norm_method == "zscore":
                norm_unit = "z-score"
                norm_label = "Z-Score"
                norm_param_text = (
                    f"norm=zscore  "
                    f"smooth={_resolved_smooth}f ({_resolved_smooth / fs:.2f}s)  "
                    f"fs={fs:.2f}Hz  tau={tau:.2f}s  "
                    f"neuropil={'on' if correct_neuropil else 'off'}"
                )
            else:
                norm_unit = r"% $\Delta$F/F$_0$"
                norm_label = r"$\Delta$F/F"
                norm_param_text = (
                    f"dff_percentile={dff_percentile}  "
                    f"window={_resolved_window}f ({_resolved_window / fs:.1f}s)  "
                    f"smooth={_resolved_smooth}f ({_resolved_smooth / fs:.2f}s)  "
                    f"fs={fs:.2f}Hz  tau={tau:.2f}s  "
                    f"neuropil={'on' if correct_neuropil else 'off'}"
                )

            def _norm_for_display(F_in):
                # display trace, follows norm_method
                if F_in.shape[0] == 0:
                    return np.zeros((0, F.shape[1]))
                if norm_method == "zscore":
                    return zscore_trace(
                        F_in, smooth_window=dff_smooth_window, fs=fs, tau=tau
                    )
                return dff_rolling_percentile(
                    F_in,
                    percentile=dff_percentile,
                    window_size=dff_window_size,
                    smooth_window=dff_smooth_window,
                    fs=fs,
                    tau=tau,
                ) * 100

            def _dff_unsmoothed(F_in):
                # always ΔF/F (%): shot noise is defined on ΔF/F regardless
                # of the selected norm_method
                if F_in.shape[0] == 0:
                    return np.zeros((0, F.shape[1]))
                return dff_rolling_percentile(
                    F_in,
                    percentile=dff_percentile,
                    window_size=dff_window_size,
                    smooth_window=1,
                    fs=fs,
                    tau=tau,
                ) * 100

            norm_acc = _norm_for_display(F_dff_accepted)
            norm_rej = _norm_for_display(F_dff_rejected)
            dff_acc_unsmoothed = _dff_unsmoothed(F_dff_accepted)
            dff_rej_unsmoothed = _dff_unsmoothed(F_dff_rejected)

            if n_accepted > 0:
                stat_accepted = [s for s, m in zip(res["stat"], iscell_mask) if m]
                # quality score subtracts Fneu internally; only pass it when
                # neuropil correction is enabled.
                quality = compute_trace_quality_score(
                    F_accepted,
                    Fneu=(res["Fneu"][iscell_mask] if (correct_neuropil and "Fneu" in res) else None),
                    stat=stat_accepted,
                    fs=fs,
                )
                quality_sort_idx = quality["sort_idx"]
                norm_acc_sorted = norm_acc[quality_sort_idx]
                F_accepted_sorted = F_accepted[quality_sort_idx]

                cell_counts = [20, 50, 100]
                for n_cells in cell_counts:
                    if n_accepted >= n_cells:
                        plot_traces(
                            norm_acc_sorted,
                            save_path=expected_files[f"traces_norm_{n_cells}"],
                            num_neurons=n_cells,
                            scale_bar_unit=norm_unit,
                            title=rf"Top {n_cells} {norm_label} Traces by Quality (n={n_accepted} total)",
                            fig_text=norm_param_text,
                        )
                        plot_traces(
                            F_accepted_sorted,
                            save_path=expected_files[f"traces_raw_{n_cells}"],
                            num_neurons=n_cells,
                            scale_bar_unit="a.u.",
                            title=f"Top {n_cells} Raw Traces by Quality (n={n_accepted} total)",
                        )
                    elif n_cells == 20:
                        plot_traces(
                            norm_acc_sorted,
                            save_path=expected_files["traces_norm_20"],
                            num_neurons=min(20, n_accepted),
                            scale_bar_unit=norm_unit,
                            title=rf"Top {min(20, n_accepted)} {norm_label} Traces by Quality (n={n_accepted} total)",
                            fig_text=norm_param_text,
                        )
                        plot_traces(
                            F_accepted_sorted,
                            save_path=expected_files["traces_raw_20"],
                            num_neurons=min(20, n_accepted),
                            scale_bar_unit="a.u.",
                            title=f"Top {min(20, n_accepted)} Raw Traces by Quality (n={n_accepted} total)",
                        )
            else:
                print("  No accepted cells - skipping accepted trace plots")

            if n_rejected > 0:
                plot_traces(
                    norm_rej,
                    save_path=expected_files["traces_rejected"],
                    num_neurons=min(20, n_rejected),
                    scale_bar_unit=norm_unit,
                    title=rf"{norm_label} Traces - Rejected ROIs (n={n_rejected})",
                    fig_text=norm_param_text,
                )
            else:
                print("  No rejected ROIs - skipping rejected trace plots")

            # noise distributions
            if n_accepted > 0:
                dff_noise_acc = dff_shot_noise(dff_acc_unsmoothed, fs)
                plot_noise_distribution(
                    dff_noise_acc,
                    output_filename=expected_files["noise_acc"],
                    title=f"Shot-Noise Distribution (Accepted, n={n_accepted})",
                )

            if n_rejected > 0:
                dff_noise_rej = dff_shot_noise(dff_rej_unsmoothed, fs)
                plot_noise_distribution(
                    dff_noise_rej,
                    output_filename=expected_files["noise_rej"],
                    title=f"Shot-Noise Distribution (Rejected, n={n_rejected})",
                )
        except Exception as e:
            print(f"  Warning: trace/noise plots failed: {e}")

    # summary images (no masks) - always generated
    fig_label = kwargs.get("fig_label", plane_dir.stem)
    for key in ["meanImg", "max_proj", "meanImgE"]:
        if key in output_ops and output_ops[key] is not None:
            try:
                plot_projection(
                    output_ops,
                    expected_files[key],
                    fig_label=fig_label,
                    display_masks=False,
                    add_scalebar=True,
                    proj=key,
                )
            except Exception as e:
                print(f"  Failed to plot {key}: {e}")

    # quality diagnostics
    try:
        plot_plane_diagnostics(plane_dir, save_path=expected_files["quality_diagnostics"])
    except Exception as e:
        print(f"  Failed to generate quality diagnostics: {e}")

    # regional zoom
    try:
        plot_regional_zoom(
            plane_dir,
            zoom_size=150,
            img_key="meanImgE",
            save_path=expected_files["regional_zoom"],
        )
    except Exception as e:
        print(f"  Failed to generate regional zoom: {e}")

    return output_ops


def normalize99(img):
    """
    Normalize image using 1st and 99th percentile values.

    This is a robust normalization that clips outliers and scales the image
    to the [0, 1] range based on the 1st and 99th percentile values.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array of any shape.

    Returns
    -------
    numpy.ndarray
        Normalized image with values clipped to [0, 1].

    Examples
    --------
    >>> img = np.random.rand(100, 100) * 1000
    >>> normalized = normalize99(img)
    >>> assert 0 <= normalized.min() <= normalized.max() <= 1
    """
    p1, p99 = np.percentile(img, [1, 99])
    return np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)


def apply_hp_filter(img, diameter, spatial_hp_cp):
    """
    Apply high-pass filter to image for Cellpose preprocessing.

    This replicates Suite2p's anatomical detection preprocessing, which
    normalizes the image and then subtracts a Gaussian-smoothed version
    to enhance cell boundaries.

    Parameters
    ----------
    img : numpy.ndarray
        Input 2D image (e.g., mean image or max projection).
    diameter : int or float
        Expected cell diameter in pixels. Used to calculate the Gaussian
        sigma as ``diameter * spatial_hp_cp``.
    spatial_hp_cp : float
        High-pass filter strength multiplier. Common values:
        - 0: No filtering (return normalized image)
        - 0.5: LBM default, mild filtering
        - 2.0: Strong filtering, enhances small features

    Returns
    -------
    numpy.ndarray
        High-pass filtered image.

    See Also
    --------
    normalize99 : Used internally for percentile normalization.

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter
    >>> img = np.random.rand(256, 256)
    >>> filtered = apply_hp_filter(img, diameter=6, spatial_hp_cp=0.5)
    """
    from scipy.ndimage import gaussian_filter

    img_norm = normalize99(img)
    if spatial_hp_cp > 0:
        sigma = diameter * spatial_hp_cp
        img_hp = img_norm - gaussian_filter(img_norm, sigma)
    else:
        img_hp = img_norm
    return img_hp


def random_colors_for_mask(mask, seed=42):
    """
    Generate random distinct colors for each cell ID in a mask.

    Uses HSV color space to generate visually distinct colors for each
    unique cell label in the mask.

    Parameters
    ----------
    mask : numpy.ndarray
        2D integer array where each unique positive value represents a
        different cell. Background should be 0.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    numpy.ndarray
        RGB image of shape ``(Ly, Lx, 3)`` with float32 values in [0, 1].

    See Also
    --------
    mask_overlay : Uses this function to colorize masks.
    stat_to_mask : Converts Suite2p stat to mask array.

    Examples
    --------
    >>> mask = np.zeros((100, 100), dtype=np.int32)
    >>> mask[10:20, 10:20] = 1
    >>> mask[30:40, 30:40] = 2
    >>> colors = random_colors_for_mask(mask)
    >>> assert colors.shape == (100, 100, 3)
    """
    from matplotlib.colors import hsv_to_rgb

    n_cells = mask.max()
    if n_cells == 0:
        return np.zeros((*mask.shape, 3), dtype=np.float32)

    # Generate random colors using HSV for better distinction
    np.random.seed(seed)
    hues = np.random.rand(n_cells + 1)
    saturations = 0.7 + 0.3 * np.random.rand(n_cells + 1)
    values = 0.8 + 0.2 * np.random.rand(n_cells + 1)

    # Convert HSV to RGB
    colors = np.zeros((n_cells + 1, 3))
    for i in range(1, n_cells + 1):
        colors[i] = hsv_to_rgb([hues[i], saturations[i], values[i]])

    # Map colors to mask
    rgb = colors[mask]
    return rgb.astype(np.float32)


def mask_overlay(img, mask, alpha=0.5):
    """
    Overlay colored masks on a grayscale image.

    Creates a visualization where detected cells are shown as colored
    regions blended with the underlying grayscale image.

    Parameters
    ----------
    img : numpy.ndarray
        2D grayscale background image (e.g., mean image, max projection).
    mask : numpy.ndarray
        2D integer mask where each positive value represents a different
        cell. Background should be 0.
    alpha : float, optional
        Blending factor for mask overlay. 0 = fully transparent,
        1 = fully opaque. Default is 0.5.

    Returns
    -------
    numpy.ndarray
        RGB image of shape ``(Ly, Lx, 3)`` with float32 values in [0, 1].

    See Also
    --------
    random_colors_for_mask : Generates colors for each cell.
    stat_to_mask : Converts Suite2p stat to mask.
    plot_mask_comparison : Uses this for multi-panel visualization.

    Examples
    --------
    >>> img = np.random.rand(256, 256)
    >>> mask = np.zeros((256, 256), dtype=np.int32)
    >>> mask[50:100, 50:100] = 1
    >>> overlay = mask_overlay(img, mask, alpha=0.5)
    >>> assert overlay.shape == (256, 256, 3)
    """
    img_norm = normalize99(img)
    rgb = np.stack([img_norm] * 3, axis=-1).astype(np.float32)

    if mask.max() > 0:
        colors = random_colors_for_mask(mask)
        mask_px = mask > 0
        rgb[mask_px] = (1 - alpha) * rgb[mask_px] + alpha * colors[mask_px]

    return rgb


def get_background_image(ops, img_key="max_proj"):
    """
    Get background image and coordinate offsets from ops.

    Handles the coordinate space difference between full-FOV images
    (meanImg, meanImgE, refImg) and cropped images (max_proj, Vcorr).

    Parameters
    ----------
    ops : dict
        Suite2p ops dictionary.
    img_key : str
        Key for desired image: 'max_proj', 'Vcorr', 'meanImg', 'meanImgE'.

    Returns
    -------
    img : np.ndarray
        Background image.
    yoff : int
        Y offset to subtract from stat coordinates.
    xoff : int
        X offset to subtract from stat coordinates.
    """
    Ly = ops.get("Ly", 512)
    Lx = ops.get("Lx", 512)
    yrange = ops.get("yrange", [0, Ly])
    xrange = ops.get("xrange", [0, Lx])

    # cropped images need coordinate adjustment
    cropped_keys = {"max_proj", "Vcorr"}

    if img_key in ops:
        img = ops[img_key]
        if img_key in cropped_keys:
            yoff, xoff = int(yrange[0]), int(xrange[0])
        else:
            yoff, xoff = 0, 0
    else:
        # fallback to meanImg (full space)
        img = ops.get("meanImg", np.zeros((Ly, Lx)))
        yoff, xoff = 0, 0

    return img, yoff, xoff


def _crop_projection_to_valid(ops, img, proj_key):
    """
    Resolve stat-coordinate offsets for a projection image.

    Suite2p stores projections in two coordinate systems: ``max_proj`` and
    ``Vcorr`` are already sliced by suite2p to ``yrange`` / ``xrange``;
    ``meanImg`` / ``meanImgE`` / ``refImg`` are in the full frame. The
    image itself is returned unchanged; only the offsets needed to
    translate full-frame stat coordinates into the image's space are
    computed.

    Parameters
    ----------
    ops : dict
        Suite2p ops dictionary.
    img : np.ndarray
        The projection image.
    proj_key : str
        The key in ``ops`` the image came from.

    Returns
    -------
    img : np.ndarray
        The input image, unchanged.
    stat_yoff, stat_xoff : int
        Offsets to subtract from full-frame ``ypix`` / ``xpix`` stat
        coordinates to land them in the image's coordinate space.
    """
    full_Ly, full_Lx = img.shape[:2]
    cropped_keys = {"max_proj", "Vcorr"}
    if proj_key in cropped_keys:
        stat_yoff = int(ops.get("yrange", [0, full_Ly])[0])
        stat_xoff = int(ops.get("xrange", [0, full_Lx])[0])
    else:
        stat_yoff = 0
        stat_xoff = 0
    return img, stat_yoff, stat_xoff


def stat_to_mask(stat, Ly, Lx, yoff=0, xoff=0):
    """
    Convert Suite2p stat array to a 2D labeled mask.

    Each cell is assigned a unique integer label starting from 1.
    Background pixels are 0.

    Parameters
    ----------
    stat : numpy.ndarray or list
        Array of Suite2p stat dictionaries, each containing 'ypix' and
        'xpix' keys with pixel coordinates.
    Ly : int
        Image height in pixels.
    Lx : int
        Image width in pixels.
    yoff : int, optional
        Y offset to subtract from stat coordinates (for cropped images).
    xoff : int, optional
        X offset to subtract from stat coordinates (for cropped images).

    Returns
    -------
    numpy.ndarray
        2D mask of shape ``(Ly, Lx)`` with dtype uint16. Each cell has
        a unique integer label, background is 0.

    See Also
    --------
    mask_overlay : Uses masks for visualization.

    Examples
    --------
    >>> stat = [{'ypix': np.array([10, 11]), 'xpix': np.array([20, 21])}]
    >>> mask = stat_to_mask(stat, Ly=100, Lx=100)
    >>> assert mask[10, 20] == 1
    >>> assert mask[0, 0] == 0
    """
    mask = np.zeros((Ly, Lx), dtype=np.uint16)
    for i, s in enumerate(stat):
        ypix = s['ypix'] - yoff
        xpix = s['xpix'] - xoff
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        mask[ypix[valid], xpix[valid]] = i + 1
    return mask


def plot_mask_comparison(
    img,
    results,
    zoom_levels=None,
    zoom_center=None,
    title=None,
    save_path=None,
    figsize=None,
):
    """
    Create a multi-panel comparison of detection results with zoom views.

    Generates a grid visualization comparing different parameter combinations
    (e.g., diameters) with full-image views and progressively zoomed regions.

    Parameters
    ----------
    img : numpy.ndarray
        2D background image for overlay (e.g., max projection).
    results : dict
        Dictionary mapping names to result dicts. Each result dict should
        contain either:
        - 'masks': 2D labeled mask array, OR
        - 'stat': Suite2p stat array (will be converted to mask)
        And optionally:
        - 'n_cells': Number of cells (computed from mask if not provided)
    zoom_levels : list of int, optional
        List of zoom region sizes in pixels. Default is [400, 200, 100].
    zoom_center : tuple of (int, int), optional
        Center point (cy, cx) for zoom regions. Default is image center.
    title : str, optional
        Overall figure title.
    save_path : str or Path, optional
        Path to save the figure. If None, displays with plt.show().
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is auto-calculated.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.

    See Also
    --------
    mask_overlay : Creates individual overlays.
    stat_to_mask : Converts stat arrays to masks.

    Examples
    --------
    >>> img = ops['max_proj']
    >>> results = {
    ...     'd=2': {'masks': masks_d2, 'n_cells': 500},
    ...     'd=4': {'masks': masks_d4, 'n_cells': 350},
    ...     'd=6': {'masks': masks_d6, 'n_cells': 200},
    ... }
    >>> fig = plot_mask_comparison(img, results, zoom_levels=[200, 100])
    """
    Ly, Lx = img.shape[:2]

    # Default zoom levels
    if zoom_levels is None:
        zoom_levels = [400, 200, 100]

    # Default to image center
    if zoom_center is None:
        cy, cx = Ly // 2, Lx // 2
    else:
        cy, cx = zoom_center

    n_cols = len(results)
    n_rows = len(zoom_levels) + 1  # Full image + zoom levels

    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Color palette for zoom boxes
    box_colors = ['yellow', 'cyan', 'magenta', 'lime', 'orange']

    for col, (name, r) in enumerate(results.items()):
        # Get or create mask
        if 'masks' in r:
            mask = r['masks']
        elif 'stat' in r:
            mask = stat_to_mask(r['stat'], Ly, Lx)
        else:
            raise ValueError(f"Result '{name}' must contain 'masks' or 'stat'")

        # Get cell count
        if 'n_cells' in r:
            n_cells = r['n_cells']
        else:
            n_cells = mask.max()

        overlay = mask_overlay(img, mask)

        # Full image with zoom boxes
        axes[0, col].imshow(overlay)
        axes[0, col].set_title(f"{name}: {n_cells} cells\nFull image")
        axes[0, col].axis('off')

        # Draw zoom boxes
        for i, zs in enumerate(zoom_levels):
            color = box_colors[i % len(box_colors)]
            rect = Rectangle(
                (cx - zs // 2, cy - zs // 2), zs, zs,
                fill=False, edgecolor=color, linewidth=2
            )
            axes[0, col].add_patch(rect)

        # Zoomed views
        for row, zs in enumerate(zoom_levels):
            y1, y2 = max(0, cy - zs // 2), min(Ly, cy + zs // 2)
            x1, x2 = max(0, cx - zs // 2), min(Lx, cx + zs // 2)

            zoom_overlay = overlay[y1:y2, x1:x2]
            zoom_mask = mask[y1:y2, x1:x2]
            n_cells_zoom = len(np.unique(zoom_mask)) - 1  # Exclude background

            axes[row + 1, col].imshow(zoom_overlay)
            axes[row + 1, col].set_title(f"{zs}x{zs} zoom: {n_cells_zoom} cells")
            axes[row + 1, col].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_regional_zoom(
    plane_dir,
    zoom_size: int = 150,
    img_key: str = "max_proj",
    alpha: float = 0.5,
    save_path=None,
    figsize: tuple = (15, 10),
    accepted_only: bool = True,
):
    """
    Plot corner, edge, and center zoom views of detection results.

    Creates a 2x3 grid visualization showing the full image with region
    boxes, plus zoomed views of each corner and the center. Useful for
    checking detection quality across different parts of the field of view.

    Parameters
    ----------
    plane_dir : str or Path
        Path to a Suite2p plane directory containing ops.npy, stat.npy,
        and optionally iscell.npy.
    zoom_size : int, optional
        Size of zoom regions in pixels. Default is 150.
    img_key : str, optional
        Key in ops to use as background image. Options:
        'max_proj', 'meanImg', 'meanImgE'. Default is 'max_proj'.
    alpha : float, optional
        Blending factor for mask overlay (0-1). Default is 0.5.
    save_path : str or Path, optional
        Path to save the figure. If None, displays with plt.show().
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (15, 10).
    accepted_only : bool, optional
        If True, only show cells marked as accepted (iscell[:, 0] == 1).
        Default is True.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.

    Examples
    --------
    >>> # From pipeline results
    >>> ops_paths = lsp.pipeline(input_data=path, save_path=output_dir, ...)
    >>> for ops_path in ops_paths:
    ...     plot_regional_zoom(ops_path.parent, zoom_size=150)

    >>> # With custom settings
    >>> plot_regional_zoom(
    ...     "D:/results/plane01_vdaq0",
    ...     zoom_size=200,
    ...     img_key="meanImgE",
    ...     save_path="regional_zoom.png"
    ... )
    """
    plane_dir = Path(plane_dir)

    # load results
    res = load_planar_results(plane_dir)
    ops = load_ops(plane_dir)

    # get background image with coordinate offsets
    img, yoff, xoff = get_background_image(ops, img_key)
    img_h, img_w = img.shape[:2]

    # Get stat and optionally filter by iscell
    stat = res["stat"]
    if accepted_only and "iscell" in res:
        iscell_mask = res["iscell"][:, 0].astype(bool)
        stat = stat[iscell_mask]

    n_cells = len(stat)

    # Create mask from stat (with coordinate offset for cropped images)
    mask = stat_to_mask(stat, img_h, img_w, yoff, xoff)

    # Create overlay
    overlay = mask_overlay(img, mask, alpha=alpha)

    vy1, vy2, vx1, vx2 = 0, img_h, 0, img_w
    valid_h = vy2 - vy1
    valid_w = vx2 - vx1

    # clamp zoom size to the valid extent
    zs = min(int(zoom_size), valid_h, valid_w)
    cy, cx = (vy1 + vy2) // 2, (vx1 + vx2) // 2

    regions = {
        "Top-Left": (vy1, vy1 + zs, vx1, vx1 + zs),
        "Top-Right": (vy1, vy1 + zs, vx2 - zs, vx2),
        "Bottom-Left": (vy2 - zs, vy2, vx1, vx1 + zs),
        "Bottom-Right": (vy2 - zs, vy2, vx2 - zs, vx2),
        "Center": (cy - zs // 2, cy + zs // 2, cx - zs // 2, cx + zs // 2),
    }

    # Color palette for boxes
    box_colors = ['red', 'blue', 'green', 'orange', 'yellow']

    fig, axes = plt.subplots(2, 3, figsize=figsize, facecolor="black")
    axes = axes.flatten()

    # Full image with boxes showing regions
    ax = axes[0]
    ax.set_facecolor("black")
    ax.imshow(overlay)
    for (name, (y1, y2, x1, x2)), c in zip(regions.items(), box_colors):
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=c, linewidth=2, label=name
        )
        ax.add_patch(rect)
    ax.set_title(f"Full Image: {n_cells} cells\n(boxes show zoom regions)", color="white")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8,
              facecolor="black", edgecolor="gray", labelcolor="white")
    ax.axis('off')

    # Zoomed views
    for ax, ((name, (y1, y2, x1, x2)), c) in zip(axes[1:], zip(regions.items(), box_colors)):
        ax.set_facecolor("black")
        zoom_mask = mask[y1:y2, x1:x2]
        n_zoom = len(np.unique(zoom_mask)) - 1  # Exclude background
        ax.imshow(overlay[y1:y2, x1:x2])
        ax.set_title(f"{name}: {n_zoom} cells", color=c, fontweight='bold')
        ax.axis('off')

    # Get plane name for title
    plane_name = plane_dir.name
    diameter = ops.get("diameter", "?")
    plt.suptitle(
        f"{plane_name} - Regional Comparison ({zs}x{zs}) - d={diameter}",
        fontsize=14, fontweight='bold', color="white"
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor="black")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_filtered_cells(
    plane_dir,
    iscell_original,
    iscell_filtered,
    img_key: str = "max_proj",
    alpha: float = 0.5,
    save_path=None,
    figsize: tuple = (18, 6),
    title: str = None,
):
    """
    Plot side-by-side comparison of cells before and after filtering.

    Shows three panels: kept cells, removed cells, and both overlaid
    with different colors.

    Parameters
    ----------
    plane_dir : str or Path
        Path to a Suite2p plane directory containing ops.npy, stat.npy.
    iscell_original : np.ndarray
        Original iscell array before filtering (n_rois,) or (n_rois, 2).
    iscell_filtered : np.ndarray
        Filtered iscell array (n_rois,) or (n_rois, 2).
    img_key : str, optional
        Key in ops to use as background image. Default is 'max_proj'.
    alpha : float, optional
        Blending factor for mask overlay (0-1). Default is 0.5.
    save_path : str or Path, optional
        Path to save the figure. If None, displays with plt.show().
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (18, 6).
    title : str, optional
        Custom title for the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.

    Examples
    --------
    >>> from lbm_suite2p_python import filter_by_max_diameter, plot_filtered_cells
    >>> res = load_planar_results(plane_dir)
    >>> iscell_filtered = filter_by_max_diameter(
    ...     res["iscell"], res["stat"], max_diameter_px=15
    ... )
    >>> plot_filtered_cells(plane_dir, res["iscell"], iscell_filtered)
    """
    plane_dir = Path(plane_dir)

    # load results (using direct load to avoid unnecessarily reloading iscell.npy which may be modified)
    stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    ops = load_ops(plane_dir)

    # get background image with coordinate offsets
    img, yoff, xoff = get_background_image(ops, img_key)
    img_h, img_w = img.shape[:2]

    # normalize iscell arrays to 1D boolean
    if iscell_original.ndim == 2:
        iscell_original = iscell_original[:, 0]
    if iscell_filtered.ndim == 2:
        iscell_filtered = iscell_filtered[:, 0]

    iscell_original = iscell_original.astype(bool)
    iscell_filtered = iscell_filtered.astype(bool)

    # Identify kept and removed cells
    kept_mask = iscell_filtered
    removed_mask = iscell_original & ~iscell_filtered

    n_kept = kept_mask.sum()
    n_removed = removed_mask.sum()
    n_original = iscell_original.sum()

    # Create masks for visualization (with coordinate offset for cropped images)
    mask_kept = stat_to_mask(stat[kept_mask], img_h, img_w, yoff, xoff)
    mask_removed = stat_to_mask(stat[removed_mask], img_h, img_w, yoff, xoff)

    # Normalize image
    img_norm = normalize99(img)
    img_rgb = np.stack([img_norm] * 3, axis=-1).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="black")

    # Panel 1: Kept cells (green)
    ax = axes[0]
    ax.set_facecolor("black")
    overlay_kept = img_rgb.copy()
    if mask_kept.max() > 0:
        mask_px = mask_kept > 0
        overlay_kept[mask_px] = (1 - alpha) * overlay_kept[mask_px] + alpha * np.array([0, 1, 0])
    ax.imshow(overlay_kept)
    ax.set_title(f"Kept: {n_kept} cells", fontsize=12, fontweight='bold', color='lime')
    ax.axis('off')

    # Panel 2: Removed cells (red)
    ax = axes[1]
    ax.set_facecolor("black")
    overlay_removed = img_rgb.copy()
    if mask_removed.max() > 0:
        mask_px = mask_removed > 0
        overlay_removed[mask_px] = (1 - alpha) * overlay_removed[mask_px] + alpha * np.array([1, 0, 0])
    ax.imshow(overlay_removed)
    ax.set_title(f"Removed: {n_removed} cells", fontsize=12, fontweight='bold', color='red')
    ax.axis('off')

    # Panel 3: Both overlaid
    ax = axes[2]
    ax.set_facecolor("black")
    overlay_both = img_rgb.copy()
    if mask_kept.max() > 0:
        mask_px = mask_kept > 0
        overlay_both[mask_px] = (1 - alpha) * overlay_both[mask_px] + alpha * np.array([0, 1, 0])
    if mask_removed.max() > 0:
        mask_px = mask_removed > 0
        overlay_both[mask_px] = (1 - alpha) * overlay_both[mask_px] + alpha * np.array([1, 0, 0])
    ax.imshow(overlay_both)
    ax.set_title(f"Combined: {n_kept} kept / {n_removed} removed", fontsize=12, fontweight='bold', color="white")
    ax.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label=f'Kept ({n_kept})'),
        Patch(facecolor='red', alpha=0.7, label=f'Removed ({n_removed})'),
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=10,
                   facecolor="black", edgecolor="gray", labelcolor="white")

    # Title
    if title is None:
        plane_name = plane_dir.name
        title = f"{plane_name}: {n_original} -> {n_kept} cells ({n_removed} removed)"

    plt.suptitle(title, fontsize=14, fontweight='bold', color="white")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor="black")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_filter_exclusions(
    plane_dir,
    iscell_filtered,
    filter_results: list,
    stat=None,
    ops=None,
    img_key: str = "max_proj",
    alpha: float = 0.5,
    save_dir=None,
    figsize: tuple = (12, 6),
):
    """
    Create one PNG per filter showing cells it excluded.

    For each filter that rejected cells, creates a visualization with:
    - Accepted cells (green)
    - Cells rejected by this specific filter (red)
    - Title with filter name and parameters

    Parameters
    ----------
    plane_dir : str or Path
        Path to Suite2p plane directory.
    iscell_filtered : np.ndarray
        Final filtered iscell array (n_rois,) or (n_rois, 2).
    filter_results : list of dict
        Results from apply_filters(), each dict has 'name', 'removed_mask', 'info'.
    stat : np.ndarray, optional
        Suite2p stat array. If None, loads from plane_dir.
    ops : dict, optional
        Suite2p ops dict. If None, loads from plane_dir.
    img_key : str, default "max_proj"
        Key in ops for background image.
    alpha : float, default 0.5
        Overlay transparency.
    save_dir : str or Path, optional
        Directory to save PNGs. Defaults to plane_dir.
    figsize : tuple, default (12, 6)
        Figure size.

    Returns
    -------
    dict
        Filter metadata: {filter_name: {params, n_rejected, n_remaining}}
    """
    plane_dir = Path(plane_dir)
    save_dir = Path(save_dir) if save_dir else plane_dir

    # purge stale filter PNGs (both legacy `14_filter_*.png` and any prior
    # letter-suffixed run with a different filter set), so the directory
    # only contains figures for the current apply_filters call.
    for old in save_dir.glob("14*_filter_*.png"):
        try:
            old.unlink()
        except OSError:
            pass

    # load data if needed
    if stat is None:
        stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    if ops is None:
        ops = load_ops(plane_dir)

    # pick the same projection used by 04b_rejected_segmentation.png so
    # filter overlays match the rest of the segmentation figures.
    # anatomical_only mapping: 0->Vcorr, 1->max_proj, 2->meanImg, 3->meanImgE, 4->max_proj
    anatomical_only = int(ops.get("anatomical_only", 0) or 0)
    proj_lookup = {
        0: ("Vcorr", "Correlation Image"),
        1: ("max_proj", "Max Projection"),
        2: ("meanImg", "Mean Image"),
        3: ("meanImgE", "Enhanced Mean Image"),
        4: ("max_proj", "Max Projection"),
    }
    fallback_titles = {
        "meanImg": "Mean Image",
        "max_proj": "Max Projection",
        "meanImgE": "Enhanced Mean Image",
        "Vcorr": "Correlation Image",
    }

    def _is_valid_image(im):
        if im is None:
            return False
        if isinstance(im, (int, float)) and im == 0:
            return False
        if isinstance(im, np.ndarray) and im.size == 0:
            return False
        return True

    proj_key, proj_title = proj_lookup.get(anatomical_only, ("meanImg", "Mean Image"))
    proj_img = ops.get(proj_key)
    if not _is_valid_image(proj_img):
        for fk in ("meanImg", "max_proj", "meanImgE", "Vcorr"):
            if _is_valid_image(ops.get(fk)):
                proj_key = fk
                proj_title = fallback_titles[fk]
                proj_img = ops.get(fk)
                break

    # last-ditch fallback to the explicit img_key (preserves prior contract)
    if not _is_valid_image(proj_img):
        bg, _yoff, _xoff = get_background_image(ops, img_key)
        proj_img = bg
        proj_key = None
        proj_title = img_key

    # normalize iscell
    if iscell_filtered.ndim == 2:
        iscell_filtered = iscell_filtered[:, 0]
    accepted_mask = iscell_filtered.astype(bool)
    n_accepted_total = int(accepted_mask.sum())

    filter_metadata = {}

    for idx, result in enumerate(filter_results):
        name = result["name"]
        removed_mask = result["removed_mask"]
        info = result["info"]
        config = result.get("config", {})
        n_rejected = int(removed_mask.sum())

        if n_rejected == 0:
            continue

        # build params from user config first (more meaningful), then computed info
        params = {}
        for key in ["min_diameter_um", "max_diameter_um", "min_diameter_px", "max_diameter_px",
                    "min_area_px", "max_area_px", "min_mult", "max_mult", "max_ratio",
                    "correct_neuropil", "percentile", "window_size",
                    "reject_negative_F0", "min_F0_abs", "min_F0_rel"]:
            if key in config and config[key] is not None:
                val = config[key]
                params[key] = round(val, 3) if isinstance(val, float) else val
        if not params:
            for key in ["min_px", "max_px", "min_ratio", "max_ratio", "lower_px", "upper_px"]:
                if key in info and info[key] is not None:
                    params[key] = round(info[key], 1)

        # baseline filters: short reason tag so the title states the
        # specific criterion being applied (cleaner than reading params).
        reason_str = None
        if name == "negative_baseline":
            reason_str = "rolling F0 < 0"
        elif name == "min_baseline_abs" and info.get("min_F0_abs") is not None:
            reason_str = f"median F0 < {info['min_F0_abs']:g}"
        elif name == "min_baseline_rel" and info.get("min_F0_rel") is not None:
            reason_str = f"min F0 < {info['min_F0_rel']:g} × median(F_raw)"

        # subset stat to accepted + this-filter-excluded so the canvas
        # only renders cells relevant to *this* filter step. Cells rejected
        # by suite2p or by other filters are intentionally invisible.
        keep_idx = accepted_mask | removed_mask
        if not keep_idx.any():
            continue
        stat_subset = stat[keep_idx]
        subset_iscell = accepted_mask[keep_idx]  # True=kept, False=removed-by-this-filter

        # lam-weighted overlay matching plot_accepted_rejected_overlay
        canvas, _, _ = _build_accepted_rejected_canvas(
            proj_img,
            stat_subset,
            subset_iscell,
            ops=ops if proj_key else None,
            proj_key=proj_key,
        )

        fig, ax = plt.subplots(figsize=figsize, facecolor="black")
        ax.set_facecolor("black")
        ax.imshow(canvas, interpolation="nearest")

        # header: filter name + projection title; below: Accepted / Excluded counts
        header_y = 1.20
        ax.text(
            0.5, header_y,
            f"{name} — {proj_title}",
            transform=ax.transAxes,
            fontsize=12, fontweight="bold", fontname="monospace",
            color="white", ha="center", va="bottom",
        )
        ax.text(
            0.37, 1.10,
            f"Accepted: {n_accepted_total:03d}",
            transform=ax.transAxes,
            fontsize=14, fontweight="bold", fontname="monospace",
            color="lime", ha="right", va="bottom",
        )
        ax.text(
            0.63, 1.10,
            f"Excluded: {n_rejected:03d}",
            transform=ax.transAxes,
            fontsize=14, fontweight="bold", fontname="monospace",
            color="red", ha="left", va="bottom",
        )
        # subtitle: per-criterion reasons + params
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        sub_lines = []
        if reason_str:
            sub_lines.append(reason_str)
        if params_str:
            sub_lines.append(params_str)
        if sub_lines:
            ax.text(
                0.5, 1.02, "\n".join(sub_lines),
                transform=ax.transAxes,
                fontsize=8, fontname="monospace",
                color="white", ha="center", va="bottom",
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        plt.tight_layout()

        # 14a_filter_<name>.png, 14b_..., one per filter in apply order.
        # 'a'..'z' covers 26 filters; falls back to numeric for the unlikely
        # 27th onward.
        suffix = chr(ord("a") + idx) if idx < 26 else f"_{idx + 1}"
        save_path = save_dir / f"14{suffix}_filter_{name}.png"
        plt.savefig(save_path, dpi=300, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {save_path.name}")

        filter_metadata[name] = {
            "params": params,
            "n_rejected": n_rejected,
            "n_remaining": n_accepted_total,
            "figure": save_path.name,
        }

    return filter_metadata


def plot_cell_filter_summary(
    plane_dir,
    iscell_suite2p=None,
    iscell_final=None,
    filter_results: list = None,
    stat=None,
    ops=None,
    img_key: str = "max_proj",
    alpha: float = 0.5,
    save_path=None,
    figsize: tuple = (16, 10),
):
    """
    Create a summary figure showing all filtering stages for a plane.

    Shows suite2p classification, each filter's effect, and final result
    in a single well-formatted figure.

    Parameters
    ----------
    plane_dir : str or Path
        Path to Suite2p plane directory.
    iscell_suite2p : np.ndarray, optional
        Original suite2p iscell (before accept_all_cells). Loads from
        iscell_suite2p.npy if exists, otherwise uses iscell.npy.
    iscell_final : np.ndarray, optional
        Final iscell after all filters. If None, loads from iscell.npy.
    filter_results : list of dict, optional
        Results from apply_filters(). If None, attempts to reconstruct
        from ops['filter_metadata'].
    stat : np.ndarray, optional
        Suite2p stat array. If None, loads from plane_dir.
    ops : dict, optional
        Suite2p ops dict. If None, loads from plane_dir.
    img_key : str, default "max_proj"
        Key in ops for background image.
    alpha : float, default 0.5
        Overlay transparency.
    save_path : str or Path, optional
        Path to save the figure. If None, displays with plt.show().
    figsize : tuple, default (16, 10)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plane_dir = Path(plane_dir)

    # load data
    if stat is None:
        stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    if ops is None:
        ops = load_ops(plane_dir)

    # load iscell arrays
    if iscell_suite2p is None:
        s2p_file = plane_dir / "iscell_suite2p.npy"
        if s2p_file.exists():
            iscell_suite2p = np.load(s2p_file, allow_pickle=True)
        else:
            iscell_suite2p = np.load(plane_dir / "iscell.npy", allow_pickle=True)

    if iscell_final is None:
        iscell_final = np.load(plane_dir / "iscell.npy", allow_pickle=True)

    # normalize to 1d
    if iscell_suite2p.ndim == 2:
        iscell_suite2p = iscell_suite2p[:, 0]
    if iscell_final.ndim == 2:
        iscell_final = iscell_final[:, 0]

    # get filter metadata from ops if not provided
    filter_metadata = ops.get("filter_metadata", {})

    # get background image
    img, yoff, xoff = get_background_image(ops, img_key)
    img_h, img_w = img.shape[:2]
    img_norm = normalize99(img)
    img_rgb = np.stack([img_norm] * 3, axis=-1).astype(np.float32)

    # compute masks
    n_rois = len(stat)
    suite2p_accepted = iscell_suite2p.astype(bool)
    suite2p_rejected = ~suite2p_accepted
    final_accepted = iscell_final.astype(bool)

    # determine what suite2p rejected vs what filters rejected
    n_suite2p_rejected = suite2p_rejected.sum()
    n_final_accepted = final_accepted.sum()

    # build panels: suite2p classification + filters + final
    panels = []

    # panel 1: suite2p classification
    panels.append({
        "title": "suite2p classification",
        "accepted_mask": suite2p_accepted,
        "rejected_mask": suite2p_rejected,
        "n_accepted": int(suite2p_accepted.sum()),
        "n_rejected": int(n_suite2p_rejected),
        "subtitle": f"{n_suite2p_rejected} rejected by suite2p",
    })

    # panels for each filter that rejected cells
    if filter_metadata:
        for name, meta in filter_metadata.items():
            n_rejected = meta.get("n_rejected", 0)
            if n_rejected > 0:
                params = meta.get("params", {})
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                panels.append({
                    "title": f"filter: {name}",
                    "filter_name": name,
                    "n_rejected": n_rejected,
                    "subtitle": f"{n_rejected} excluded" + (f" ({params_str})" if params_str else ""),
                })

    # final panel: result
    panels.append({
        "title": "final result",
        "accepted_mask": final_accepted,
        "rejected_mask": ~final_accepted,
        "n_accepted": int(n_final_accepted),
        "n_rejected": int(n_rois - n_final_accepted),
        "subtitle": f"{n_final_accepted} accepted, {n_rois - n_final_accepted} rejected",
    })

    # create figure
    n_panels = len(panels)
    if n_panels <= 2:
        ncols = n_panels
        nrows = 1
    elif n_panels <= 4:
        ncols = 2
        nrows = 2
    else:
        ncols = 3
        nrows = (n_panels + 2) // 3

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="black")
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # hide unused axes
    for i in range(n_panels, len(axes)):
        axes[i].set_facecolor("black")
        axes[i].axis("off")

    # color scheme
    color_accepted = np.array([0.2, 0.8, 0.2])  # green
    color_rejected = np.array([0.9, 0.2, 0.2])  # red

    for i, panel in enumerate(panels):
        ax = axes[i]
        overlay = img_rgb.copy()

        if "accepted_mask" in panel:
            # draw accepted cells
            accepted_mask = panel["accepted_mask"]
            if accepted_mask.sum() > 0:
                mask_px = stat_to_mask(stat[accepted_mask], img_h, img_w, yoff, xoff)
                if mask_px.max() > 0:
                    px = mask_px > 0
                    overlay[px] = (1 - alpha) * overlay[px] + alpha * color_accepted

            # draw rejected cells
            rejected_mask = panel["rejected_mask"]
            if rejected_mask.sum() > 0:
                mask_px = stat_to_mask(stat[rejected_mask], img_h, img_w, yoff, xoff)
                if mask_px.max() > 0:
                    px = mask_px > 0
                    overlay[px] = (1 - alpha) * overlay[px] + alpha * color_rejected

        elif "filter_name" in panel:
            # for filter panels, show accepted (green) and this filter's rejected (orange)
            # reconstruct which cells this filter rejected
            # we show: final accepted (green) + cells rejected by this filter (orange)
            if final_accepted.sum() > 0:
                mask_px = stat_to_mask(stat[final_accepted], img_h, img_w, yoff, xoff)
                if mask_px.max() > 0:
                    px = mask_px > 0
                    overlay[px] = (1 - alpha) * overlay[px] + alpha * color_accepted

            # for filter panels without explicit mask, just show the count in subtitle
            # (we don't have the exact removed_mask saved, just metadata)

        ax.set_facecolor("black")
        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(panel["title"], fontsize=11, fontweight="bold", color="white")

        # add subtitle
        if "subtitle" in panel:
            ax.text(
                0.5, -0.02, panel["subtitle"],
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=9, color="lightgray"
            )

    # add legend to last panel
    legend_elements = [
        Patch(facecolor=color_accepted, alpha=0.7, label="accepted"),
        Patch(facecolor=color_rejected, alpha=0.7, label="rejected"),
    ]
    axes[n_panels - 1].legend(
        handles=legend_elements, loc="upper right", fontsize=9,
        facecolor="black", edgecolor="gray", labelcolor="white"
    )

    # overall title
    fig.suptitle(
        f"Cell Filter Summary: {n_rois} total ROIs -> {n_final_accepted} accepted",
        fontsize=13, fontweight="bold", color="white", y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_diameter_histogram(
    stat,
    iscell=None,
    max_diameter_px: float = None,
    pixel_size_um: float = None,
    bins: int = 50,
    save_path=None,
    figsize: tuple = (10, 6),
):
    """
    Plot histogram of cell diameters with optional threshold line.

    Parameters
    ----------
    stat : np.ndarray or list
        Suite2p stat array with ROI statistics.
    iscell : np.ndarray, optional
        Cell classification array. If provided, only plots accepted cells.
    max_diameter_px : float, optional
        Threshold diameter in pixels to show as vertical line.
    pixel_size_um : float, optional
        Pixel size in microns. If provided, adds micron scale to x-axis.
    bins : int, optional
        Number of histogram bins. Default is 50.
    save_path : str or Path, optional
        Path to save the figure.
    figsize : tuple, optional
        Figure size. Default is (10, 6).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    # Filter by iscell if provided
    if iscell is not None:
        if iscell.ndim == 2:
            iscell = iscell[:, 0]
        iscell = iscell.astype(bool)
        stat = stat[iscell]

    # Get radii
    if len(stat) == 0:
        print("No cells to plot")
        return None

    if "radius" not in stat[0]:
        radii = np.array([np.sqrt(len(s["xpix"]) / np.pi) for s in stat])
    else:
        radii = np.array([s["radius"] for s in stat])

    diameters_px = 2 * radii

    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    counts, bin_edges, patches = ax.hist(
        diameters_px, bins=bins, color='steelblue',
        edgecolor='white', alpha=0.7
    )

    # Color bars above threshold red
    if max_diameter_px is not None:
        for patch, left_edge in zip(patches, bin_edges[:-1]):
            if left_edge >= max_diameter_px:
                patch.set_facecolor('red')
                patch.set_alpha(0.7)

        # Add threshold line
        ax.axvline(max_diameter_px, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {max_diameter_px:.1f} px')

        # Count cells above threshold
        n_above = (diameters_px > max_diameter_px).sum()
        n_total = len(diameters_px)
        ax.legend(title=f'{n_above}/{n_total} cells above threshold')

    ax.set_xlabel('Diameter (pixels)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Cell Diameter Distribution (n={len(diameters_px)})', fontsize=14)

    # Add micron scale if pixel size provided
    if pixel_size_um is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()[0] * pixel_size_um, ax.get_xlim()[1] * pixel_size_um)
        ax2.set_xlabel('Diameter (µm)', fontsize=12)
        if max_diameter_px is not None:
            max_um = max_diameter_px * pixel_size_um
            ax2.axvline(max_um, color='red', linestyle='--', linewidth=2, alpha=0.5)

    # Add statistics
    median_d = np.median(diameters_px)
    mean_d = np.mean(diameters_px)
    stats_text = f'Median: {median_d:.1f} px\nMean: {mean_d:.1f} px'
    if pixel_size_um:
        stats_text += f'\n({median_d * pixel_size_um:.1f} / {mean_d * pixel_size_um:.1f} µm)'

    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig
