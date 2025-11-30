from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tifffile
import math

import matplotlib.offsetbox
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import VPacker, HPacker, DrawingArea
import matplotlib.gridspec as gridspec

from scipy.ndimage import distance_transform_edt

from lbm_suite2p_python.postprocessing import (
    load_ops,
    load_planar_results,
    dff_rolling_percentile,
    dff_shot_noise,
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
    - 'dff-percentile': ΔF/F₀ in percent, typically ~10–100

    Returns one of: 'raw', 'dff', 'dff-percentile'
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
    else:
        return "unknown"


def format_time(t):
    if t < 60:
        # make sure we dont show 0 seconds
        return f"{int(np.ceil(t))} s"
    elif t < 3600:
        return f"{int(round(t / 60))} min"
    else:
        return f"{int(round(t / 3600))} h"


def get_color_permutation(n):
    # choose a step from n//2+1 up to n-1 that is coprime with n
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
        # trans = ax.get_xaxis_transform()
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
        scale_bar_label: str = r"50% $\Delta$F/F$_0$",
) -> None:
    """
    Plot stacked fluorescence traces with automatic offset and scale bars.

    Parameters
    ----------
    f : ndarray
        2d array of fluorescence traces (n_neurons x n_timepoints).
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
    scale_bar_label : str, default "50% ΔF/F₀"
        User-supplied label for the vertical scale bar. The scale bar is always
        10% of the figure height; this label describes what that height represents.
    cell_indices : array-like or None
        Specific cell indices to plot. If provided, overrides num_neurons.
    """
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

    # fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    # ax.set_facecolor("black")

    # Build shifted traces array (no masking - let z-order handle overlap)
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
    for i in range(displayed_neurons - 1, -1, -1):
        ax.plot(
            data_time[: current_frame + 1],
            shifted_traces[i],
            color=colors[i],
            lw=lw,
            zorder=displayed_neurons - i,  # Lower index = higher zorder = on top
        )

    # Set y-limits based on shifted traces with minimal padding
    y_min = np.min(shifted_traces)
    y_max = np.max(shifted_traces)
    y_padding = (y_max - y_min) * 0.02
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    time_bar_length = 0.1 * window
    if time_bar_length < 60:
        time_label = f"{time_bar_length:.0f} s"
    elif time_bar_length < 3600:
        time_label = f"{time_bar_length / 60:.0f} min"
    else:
        time_label = f"{time_bar_length / 3600:.1f} hr"

    linekw = dict(color="white", linewidth=3)
    hsb = AnchoredHScaleBar(
        size=0.1,
        label=time_label,
        loc=4,
        frameon=False,
        pad=0.6,
        sep=4,
        linekw=linekw,
        ax=ax,
    )
    hsb.set_bbox_to_anchor((0.9, -0.05), transform=ax.transAxes)  # noqa
    hsb.txt._text.set_color("white")  # noqa

    ax.add_artist(hsb)

    # Vertical scale bar is 10% of figure height with user-supplied label
    vsb = AnchoredVScaleBar(
        height=0.10,
        label=scale_bar_label,
        loc="lower right",
        frameon=False,
        pad=-0.1,
        sep=4,
        linekw=linekw,
        ax=ax,
        spacer_width=0,
    )
    vsb.set_bbox_to_anchor((1.00, 0.05), transform=ax.transAxes)
    vsb.txt._text.set_color("white")
    ax.add_artist(vsb)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", color="white")

    ax.set_ylabel(
        f"Neuron Count: {displayed_neurons}",
        fontsize=8,
        fontweight="bold",
        color="white",
        labelpad=2,
    )

    if save_path:
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()
    return None

def animate_traces(
    f,
    save_path="./scrolling.mp4",
    fps=17.0,
    start_neurons=20,
    window=120,
    title="",
    gap=None,
    lw=0.5,
    cmap="tab10",
    anim_fps=60,
    expand_after=5,
    speed_factor=1.0,
    expansion_factor=2.0,
    smooth_factor=1,
):
    """WIP"""
    n_neurons, n_timepoints = f.shape
    data_time = np.arange(n_timepoints) / fps
    T_data = data_time[-1]
    current_frame = min(int(window * fps), n_timepoints - 1)
    t_f_local = (T_data - window + expansion_factor * expand_after) / (
        1 + expansion_factor
    )

    if gap is None:
        p10 = np.percentile(f[:start_neurons, : current_frame + 1], 10, axis=1)
        p90 = np.percentile(f[:start_neurons, : current_frame + 1], 90, axis=1)
        gap = np.median(p90 - p10) * 1.2

    cmap_inst = plt.get_cmap(cmap)
    colors = cmap_inst(np.linspace(0, 1, n_neurons))
    perm = np.random.permutation(n_neurons)
    colors = colors[perm]

    all_shifted = []
    for i in range(start_neurons):
        trace = f[i, : current_frame + 1]
        baseline = np.percentile(trace, 8)
        shifted = (trace - baseline) + i * gap
        all_shifted.append(shifted)

    all_y = np.concatenate(all_shifted)
    y_min = np.min(all_y)
    y_max = np.max(all_y)

    rounded_dff = np.round(y_max - y_min) * 0.1
    dff_label = f"{rounded_dff:.0f} % ΔF/F₀"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.tick_params(axis="y", labelleft=False, length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fills = []
    linekw = dict(color="white", linewidth=3)
    hsb = AnchoredHScaleBar(
        size=0.1,
        label=format_time(0.1 * window),
        loc=4,
        frameon=False,
        pad=0.6,
        sep=4,
        linekw=linekw,
        ax=ax,
    )

    hsb.set_bbox_to_anchor((0.97, -0.1), transform=ax.transAxes)  # noqa

    ax.add_artist(hsb)

    vsb = AnchoredVScaleBar(
        height=0.1,
        label=dff_label,
        loc="lower right",  # noqa
        frameon=False,
        pad=0,
        sep=4,
        linekw=linekw,
        ax=ax,
        spacer_width=0,
    )
    ax.add_artist(vsb)

    lines = []
    for i in range(n_neurons):
        (line,) = ax.plot([], [], color=colors[i], lw=lw, zorder=-i)
        lines.append(line)

    def init():
        for ix in range(n_neurons):
            if ix < start_neurons:
                _trace = f[ix, : current_frame + 1]
                _baseline = np.percentile(_trace, 8)
                _shifted = (_trace - _baseline) + ix * gap
                lines[ix].set_data(data_time[: current_frame + 1], _shifted)
            else:
                lines[ix].set_data([], [])
        extra = 0.05 * window
        ax.set_xlim(0, window + extra)
        ax.set_ylim(y_min - 0.05 * abs(y_min), y_max + 0.05 * abs(y_max))
        return lines + [hsb, vsb]

    def update(frame):
        t = speed_factor * frame / anim_fps

        if t < expand_after:
            x_min = t
            x_max = t + window
            n_visible = start_neurons
        else:
            u = min(1.0, (t - expand_after) / (t_f_local - expand_after))
            ease = 3 * u**2 - 2 * u**3  # smoothstep easing
            x_min = t

            window_start = window
            window_end = window + expansion_factor * (T_data - window - expand_after)
            current_window = window_start + (window_end - window_start) * ease

            x_max = x_min + current_window

            n_visible = start_neurons + int((n_neurons - start_neurons) * ease)
            n_visible = min(n_neurons, n_visible)

        i_lower = int(x_min * fps)
        i_upper = int(x_max * fps)
        i_upper = max(i_upper, i_lower + 1)

        for ix in range(n_neurons):
            if ix < n_visible:
                _trace = f[ix, i_lower:i_upper]
                _baseline = np.percentile(_trace, 8)
                _shifted = (_trace - _baseline) + ix * gap
                lines[ix].set_data(data_time[i_lower:i_upper], _shifted)
            else:
                lines[ix].set_data([], [])

        for fill in fills:
            fill.remove()
        fills.clear()

        for ix in range(n_visible - 1):
            trace1 = f[ix, i_lower:i_upper]
            baseline1 = np.percentile(trace1, 8)
            shifted1 = (trace1 - baseline1) + ix * gap

            trace2 = f[ix + 1, i_lower:i_upper]
            baseline2 = np.percentile(trace2, 8)
            shifted2 = (trace2 - baseline2) + (ix + 1) * gap

            fill = ax.fill_between(
                data_time[i_lower:i_upper],
                shifted1,
                shifted2,
                where=shifted1 > shifted2,
                color="black",
                zorder=-ix - 1,
            )
            fills.append(fill)

        _all_shifted = [
            (f[ix, i_lower:i_upper] - np.percentile(f[ix, i_lower:i_upper], 8))
            + ix * gap
            for ix in range(n_visible)
        ]
        _all_y = np.concatenate(_all_shifted)
        y_min_new, y_max_new = np.min(_all_y), np.max(_all_y)

        extra_axis = 0.05 * (x_max - x_min)
        ax.set_xlim(x_min, x_max + extra_axis)
        ax.set_ylim(
            y_min_new - 0.05 * abs(y_min_new), y_max_new + 0.05 * abs(y_max_new)
        )

        if title:
            ax.set_title(title, fontsize=16, fontweight="bold", color="white")

        _dff_rounded = np.round(y_max_new - y_min_new) * 0.1

        if _dff_rounded > 300:
            vsb.set_visible(False)
        else:
            _dff_label = f"{_dff_rounded:.0f} % ΔF/F₀"
            vsb.txt.set_text(_dff_label)
        hsb.txt.set_text(format_time(0.1 * (x_max - x_min)))
        ax.set_ylabel(
            f"Neuron Count: {n_visible}", fontsize=8, fontweight="bold", labelpad=2
        )

        return lines + [hsb, vsb] + fills

    effective_anim_fps = anim_fps * smooth_factor
    total_frames = int(np.ceil((T_data / speed_factor)))

    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        interval=1000 / effective_anim_fps,
        blit=True,
    )
    ani.save(save_path, fps=anim_fps)
    plt.show()


def feather_mask(mask, max_alpha=0.75, edge_width=3):
    # mask alpha using distance transform
    dist_out = distance_transform_edt(mask == 0)
    alpha = np.clip((edge_width - dist_out) / edge_width, 0, 1)
    return alpha * max_alpha


def plot_masks(
        img: np.ndarray,
        stat: list[dict] | dict,
        mask_idx: np.ndarray,
        savepath: str | Path,
        colors=None,
        title=None,
):
    """
    Draw ROI overlays onto the mean image.

    Parameters
    ----------
    stat : list[dict]
        Suite2p ROI stat dictionaries (with "ypix", "xpix", "lam").
    img : ndarray (Ly x Lx)
        Background image to overlay on.
    mask_idx : ndarray[bool]
        Boolean array selecting which ROIs to plot.
    savepath : str or Path
        Fully qualified path to save the figure.
    colors : ndarray or list, optional
        Array/list of RGB tuples for each ROI selected.
        If None, colors are assigned via HSV colormap.
    title : str, optional
        Title string to place on the figure.
    """

    # Normalize background image using percentile stretch for better contrast
    # this prevents dark images when min/max are extreme outliers
    vmin = np.nanpercentile(img, 1)
    vmax = np.nanpercentile(img, 99)
    normalized = (img - vmin) / (vmax - vmin + 1e-6)
    normalized = np.clip(normalized, 0, 1)
    # Set NaN regions to 0 (black background)
    normalized = np.nan_to_num(normalized, nan=0.0)
    canvas = np.tile(normalized, (3, 1, 1)).transpose(1, 2, 0)

    # Get image dimensions for bounds checking
    Ly, Lx = img.shape[:2]

    # Assign colors if not provided
    n_masks = mask_idx.sum()
    if colors is None:
        colors = plt.cm.hsv(np.linspace(0, 1, n_masks + 1))[:, :3]  # noqa

    c = 0
    for n, s in enumerate(stat):
        if mask_idx[n]:
            ypix, xpix, lam = s["ypix"], s["xpix"], s["lam"]

            # Bounds checking - only keep pixels within image dimensions
            valid_mask = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
            if not np.any(valid_mask):
                c += 1
                continue  # Skip ROI if no valid pixels

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

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(canvas, interpolation="nearest")
    if title is not None:
        ax.set_title(title, fontsize=10, color="white", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if savepath:
        if Path(savepath).is_dir():
            raise ValueError("savepath must be a file path, not a directory.")
        plt.savefig(savepath, dpi=300, facecolor="black")
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
    from suite2p.detection.stats import ROI
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
        fontname="Courier New",
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
        im = ROI.stats_dicts_to_3d_array(
            stat, Ly=ops["Ly"], Lx=ops["Lx"], label_id=True
        )
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
            fontname="Courier New",
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
            fontname="Courier New",
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

    fig = plt.figure(figsize=(8, 5))
    plt.hist(noise_levels, bins=50, color="gray", alpha=0.7, edgecolor="black")

    mean_noise: float = np.mean(noise_levels)  # noqa
    plt.axvline(
        mean_noise,
        color="r",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_noise:.2f}",
    )

    plt.xlabel("Noise Level", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Neurons", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if output_filename:
        plt.savefig(output_filename, dpi=200, bbox_inches="tight")
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
        fig_text = f"Neurons: {spks.shape[0]}, Superneurons: {sn.shape[0]}, n_clusters: {model.n_PCs}, n_PCs: {model.n_clusters}, locality: {model.locality}"

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


def save_pc_panels_and_metrics(ops, savepath, pcs=(0, 1, 2, 3)):
    """
    Save PC metrics in two forms:
    1. Alternating TIFF (PC Low/High side-by-side per frame, press play in ImageJ to flip).
    2. Panel TIFF (static figures for PC1/2 and PC3/4).
    Also saves summary metrics as CSV.

    Parameters
    ----------
    ops : dict or str or Path
        Suite2p ops dict or path to ops.npy. Must contain "regPC" and "regDX".
    savepath : str or Path
        Output file stem (without extension).
    pcs : tuple of int
        PCs to include (default first four).
    """
    if not isinstance(ops, dict):
        ops = np.load(ops, allow_pickle=True).item()

    if "nframes" in ops and ops["nframes"] < 1500:
        print(
            f"1500 frames needed for registration metrics, found {ops['nframes']}. Skipping PC metrics."
        )
        return {}
    elif "regPC" not in ops or "regDX" not in ops:
        print("regPC or regDX not found in ops, skipping PC metrics.")
        return {}
    elif len(pcs) != 4 or any(p < 0 for p in pcs):
        raise ValueError(
            "pcs must be a tuple of four non-negative integers."
            " E.g., (0, 1, 2, 3) for the first four PCs."
            f" Got: {pcs}"
        )

    regPC = ops["regPC"]  # shape (2, nPC, Ly, Lx)
    regDX = ops["regDX"]  # shape (nPC, 3)
    savepath = Path(savepath)

    alt_frames = []
    alt_labels = []
    for view, view_name in zip([0, 1], ["Low", "High"]):
        # side-by-side: PC1 | PC2
        left = regPC[view, pcs[0]]
        right = regPC[view, pcs[1]]
        combined = np.hstack([left, right])
        alt_frames.append(combined.astype(np.float32))
        alt_labels.append(f"PC{pcs[0] + 1}/{pcs[1] + 1} {view_name}")

        # side-by-side: PC3 | PC4
        left = regPC[view, pcs[2]]
        right = regPC[view, pcs[3]]
        combined = np.hstack([left, right])
        alt_frames.append(combined.astype(np.float32))
        alt_labels.append(f"PC{pcs[2] + 1}/{pcs[3] + 1} {view_name}")

    panel_frames = []
    panel_labels = []
    for left, right in [(pcs[0], pcs[1]), (pcs[2], pcs[3])]:
        for view, view_name in zip([0, 1], ["Low", "High"]):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(regPC[view, left], cmap="gray")
            axes[0].set_title(f"PC{left + 1} {view_name}")
            axes[0].axis("off")
            axes[1].imshow(regPC[view, right], cmap="gray")
            axes[1].set_title(f"PC{right + 1} {view_name}")
            axes[1].axis("off")
            fig.tight_layout()
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # noqa
            w, h = fig.canvas.get_width_height()
            img = img.reshape((h, w, 4))[..., :3]
            panel_frames.append(img)
            panel_labels.append(f"PC{left + 1}/{right + 1} {view_name}")
            plt.close(fig)

    panel_tiff = savepath.with_name(savepath.stem + "_panels.tif")
    tifffile.imwrite(
        panel_tiff,
        np.stack(panel_frames, axis=0),
        imagej=True,
        metadata={"Labels": panel_labels},
    )
    print(f"Saved panel TIFF to {panel_tiff}")

    df = pd.DataFrame(regDX, columns=["Rigid", "Avg_NR", "Max_NR"])
    metrics = {
        "Avg_Rigid": df["Rigid"].mean(),
        "Avg_Average_NR": df["Avg_NR"].mean(),
        "Avg_Max_NR": df["Max_NR"].mean(),
        "Max_Rigid": df["Rigid"].max(),
        "Max_Average_NR": df["Avg_NR"].max(),
        "Max_Max_NR": df["Max_NR"].max(),
    }
    csv_path = savepath.with_suffix(".csv")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    print(f"Saved metrics CSV to {csv_path}")
    print(df.head())

    return {
        "panel_tiff": panel_tiff,
        "metrics_csv": csv_path,
    }



def plot_multiplane_masks(
    suite2p_path: str | Path,
    stat: np.ndarray,
    iscell: np.ndarray,
    nrows: int = 3,
    ncols: int = 5,
    figsize: tuple = (20, 12),
    save_path: str | Path = None,
    cmap: str = "gray",
) -> plt.Figure:
    """
    Plot ROI masks from all planes in a publication-quality grid layout.

    Creates a multi-panel figure showing detected ROIs overlaid on mean images
    for each z-plane, with accepted cells in green and rejected cells in red.

    Parameters
    ----------
    suite2p_path : str or Path
        Path to suite2p directory containing plane folders (e.g., plane01_stitched/).
    stat : np.ndarray
        Consolidated stat array with 'iplane' field indicating plane assignment.
    iscell : np.ndarray
        Cell classification array (n_rois, 2) where column 0 is binary classification.
    nrows : int, default 3
        Number of rows in the figure grid.
    ncols : int, default 5
        Number of columns in the figure grid.
    figsize : tuple, default (20, 12)
        Figure size in inches (width, height).
    save_path : str or Path, optional
        If provided, save figure to this path. Otherwise display interactively.
    cmap : str, default "gray"
        Colormap for background images.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> stat = np.load("merged/stat.npy", allow_pickle=True)
    >>> iscell = np.load("merged/iscell.npy")
    >>> fig = plot_multiplane_masks("path/to/suite2p", stat, iscell)
    """
    suite2p_path = Path(suite2p_path)
    plane_dirs = sorted(suite2p_path.glob("plane*_stitched"))
    if not plane_dirs:
        plane_dirs = sorted(suite2p_path.glob("plane*"))
    nplanes = len(plane_dirs)

    # Use a clean, publication-ready style
    with plt.style.context("default"):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, facecolor="white",
            gridspec_kw={"wspace": 0.05, "hspace": 0.15}
        )
        axes = axes.flatten()

        for idx, plane_dir in enumerate(plane_dirs):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Extract plane number from directory name
            plane_name = plane_dir.name
            digits = "".join(filter(str.isdigit, plane_name))
            plane_num = int(digits) if digits else idx + 1

            # Load plane ops for mean image
            ops_file = plane_dir / "ops.npy"
            if ops_file.exists():
                plane_ops = np.load(ops_file, allow_pickle=True).item()
                img = plane_ops.get("meanImg", plane_ops.get("meanImgE"))
                if img is None:
                    img = np.zeros((plane_ops.get("Ly", 512), plane_ops.get("Lx", 512)))
            else:
                img = np.zeros((512, 512))

            # Display background image with proper contrast
            vmin, vmax = np.nanpercentile(img, [1, 99])
            ax.imshow(img, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

            # Get ROIs for this plane
            plane_mask = np.array([s.get("iplane", 0) == plane_num for s in stat])
            plane_stat = stat[plane_mask]
            plane_iscell = iscell[plane_mask]

            # Draw accepted cells (green)
            accepted_idx = plane_iscell[:, 0] == 1
            for s in plane_stat[accepted_idx]:
                ypix, xpix = s["ypix"], s["xpix"]
                ax.scatter(xpix, ypix, c="lime", s=0.3, alpha=0.7, linewidths=0)

            # Draw rejected cells (red, more transparent)
            rejected_idx = plane_iscell[:, 0] == 0
            for s in plane_stat[rejected_idx]:
                ypix, xpix = s["ypix"], s["xpix"]
                ax.scatter(xpix, ypix, c="red", s=0.2, alpha=0.4, linewidths=0)

            n_acc = accepted_idx.sum()
            n_rej = rejected_idx.sum()

            # Clean title with plane info
            ax.set_title(
                f"Plane {plane_num:02d}\n{n_acc} / {n_rej}",
                fontsize=9, fontweight="bold", pad=3
            )
            ax.axis("off")

        # Hide unused subplots
        for idx in range(nplanes, len(axes)):
            axes[idx].axis("off")

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
                   markersize=8, label="Accepted"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=8, label="Rejected"),
        ]
        fig.legend(
            handles=legend_elements, loc="lower center", ncol=2,
            fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.02)
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        else:
            plt.show()

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
    mean_line_color = "#e74c3c"  # Red for mean markers

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

    # Compute ΔF/F
    F_corrected = F_acc - 0.7 * Fneu_acc
    baseline = np.percentile(F_corrected, 20, axis=1, keepdims=True)
    baseline = np.maximum(baseline, 1e-6)
    dff = (F_corrected - baseline) / baseline

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
        F_corrected = F_acc - 0.7 * Fneu_acc
        baseline = np.percentile(F_corrected, 20, axis=1, keepdims=True)
        baseline = np.maximum(baseline, 1e-6)
        dff = (F_corrected - baseline) / baseline
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
    cell_prob = iscell[:, 1]  # classifier probability for each ROI
    n_accepted = int(accepted.sum())
    n_rejected = int((~accepted).sum())

    # Compute metrics for ALL ROIs (not just accepted)
    F_corr = F - 0.7 * Fneu
    baseline = np.percentile(F_corr, 20, axis=1, keepdims=True)
    baseline = np.maximum(baseline, 1e-6)
    dff = (F_corr - baseline) / baseline

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
    npix_acc = npix[accepted] if n_accepted > 0 else np.array([0])
    mean_snr = np.nanmean(snr_acc) if n_accepted > 0 else 0.0
    median_snr = np.nanmedian(snr_acc) if n_accepted > 0 else 0.0
    high_snr_pct = 100 * np.sum(snr_acc > 2) / max(1, len(snr_acc)) if n_accepted > 0 else 0.0
    mean_size = np.mean(npix_acc) if n_accepted > 0 else 0.0

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
            sc1 = ax_compact.scatter(compactness[valid_compact], snr[valid_compact],
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
    plane_dir, dff_percentile=8, dff_window_size=101, run_rastermap=False, **kwargs
):
    """
    Re-generate Suite2p figures for a merged plane.

    Parameters
    ----------
    plane_dir : Path
        Path to the planeXX output directory (with ops.npy, stat.npy, etc.).
    dff_percentile : int, optional
        Percentile used for ΔF/F baseline.
    dff_window_size : int, optional
        Window size for ΔF/F rolling baseline.
    run_rastermap : bool, optional
        If True, compute and plot rastermap sorting of cells.
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
        # Diagnostics and analysis
        "quality_diagnostics": plane_dir / "05_quality_diagnostics.png",
        "registration": plane_dir / "06_registration.png",
        # Traces
        "traces_raw": plane_dir / "07_traces_raw.png",
        "traces_dff": plane_dir / "08_traces_dff.png",
        "traces_noise": plane_dir / "09_traces_noise.png",
        # Noise distributions
        "noise_acc": plane_dir / "10_noise_accepted.png",
        "noise_rej": plane_dir / "11_noise_rejected.png",
        # Rastermap
        "model": plane_dir / "model.npy",
        "rastermap": plane_dir / "12_rastermap.png",
    }

    output_ops = load_ops(expected_files["ops"])

    # Dead zones are now handled via yrange/xrange cropping in run_lsp.py
    # so we don't need to mask them here anymore
    # output_ops = mask_dead_zones_in_ops(output_ops)

    # force remake of the heavy figures
    for key in [
        "registration",
        "traces_raw",
        "traces_dff",
        "traces_noise",
        "noise_acc",
        "noise_rej",
        "rastermap",
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
        cell_prob = res["iscell"][:, 1]

        spks = res["spks"]
        F = res["F"]

        # Split by accepted/rejected
        F_accepted = F[iscell_mask] if iscell_mask.sum() > 0 else np.zeros((0, F.shape[1]))
        F_rejected = F[~iscell_mask] if (~iscell_mask).sum() > 0 else np.zeros((0, F.shape[1]))
        spks_cells = spks[iscell_mask] if iscell_mask.sum() > 0 else np.zeros((0, spks.shape[1]))

        n_accepted = F_accepted.shape[0]
        n_rejected = F_rejected.shape[0]
        print(f"Plotting results for {n_accepted} accepted / {n_rejected} rejected ROIs")

        # --- RASTERMAP (only for sufficient cell counts) ---
        # rastermap sorts neurons by activity similarity for visualization
        # we cache the model to avoid recomputing, but validate it matches current data
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
                model_file = expected_files["model"]
                plot_file = expected_files["rastermap"]
                need_recompute = True

                # check if cached model exists and is valid for current cell count
                if model_file.is_file():
                    try:
                        cached_model = np.load(model_file, allow_pickle=True).item()
                        if hasattr(cached_model, "isort") and len(cached_model.isort) == n_accepted:
                            model = cached_model
                            need_recompute = False
                        else:
                            # stale model - cell count changed since last run
                            print(f"  Rastermap model stale (cached {len(cached_model.isort) if hasattr(cached_model, 'isort') else '?'} vs current {n_accepted} cells), recomputing...")
                            model_file.unlink()
                    except Exception as e:
                        print(f"  Failed to load cached rastermap model: {e}, recomputing...")
                        model_file.unlink(missing_ok=True)

                # fit new model if needed
                if need_recompute:
                    params = {
                        "n_clusters": 100 if n_accepted >= 200 else None,
                        "n_PCs": min(128, max(2, n_accepted - 1)),
                        "locality": 0.0 if n_accepted >= 200 else 0.1,
                        "time_lag_window": 15,
                        "grid_upsample": 10 if n_accepted >= 200 else 0,
                    }
                    model = rastermap.Rastermap(**params).fit(spks_cells)
                    np.save(model_file, model)

                # regenerate plot if missing (even if model was cached)
                if model is not None and not plot_file.is_file():
                    plot_rastermap(
                        spks_cells,
                        model,
                        neuron_bin_size=0,
                        save_path=plot_file,
                        title_kwargs={"fontsize": 8, "y": 0.95},
                        title="Rastermap Sorted Activity",
                    )

                # apply sorting to traces for downstream plots
                if model is not None:
                    isort_global = np.where(iscell_mask)[0][model.isort]
                    output_ops["isort"] = isort_global
                    F_accepted = F_accepted[model.isort]

        # --- COMPUTE ΔF/F ---
        fs = output_ops.get("fs", 1.0)

        if n_accepted > 0:
            dffp_acc = dff_rolling_percentile(
                F_accepted, percentile=dff_percentile, window_size=dff_window_size
            ) * 100
        else:
            dffp_acc = np.zeros((0, F.shape[1]))

        if n_rejected > 0:
            dffp_rej = dff_rolling_percentile(
                F_rejected, percentile=dff_percentile, window_size=dff_window_size
            ) * 100
        else:
            dffp_rej = np.zeros((0, F.shape[1]))

        # --- TRACE PLOTS (robust to any cell count >= 1) ---
        plot_n_traces = output_ops.get("plot_n_traces", 30)

        if n_accepted > 0:
            plot_traces(
                dffp_acc,
                save_path=expected_files["traces_dff"],
                num_neurons=min(plot_n_traces, n_accepted),
                scale_bar_label=r"50% $\Delta$F/F$_0$",
                title=r"$\Delta$F/F Traces - Accepted Cells (n={})".format(n_accepted),
            )
            plot_traces(
                F_accepted,
                save_path=expected_files["traces_raw"],
                num_neurons=min(plot_n_traces, n_accepted),
                scale_bar_label="a.u.",
                title=f"Raw Traces - Accepted Cells (n={n_accepted})",
            )
        else:
            print("  No accepted cells - skipping accepted trace plots")

        if n_rejected > 0:
            plot_traces(
                dffp_rej,
                save_path=expected_files["traces_noise"],
                num_neurons=min(plot_n_traces, n_rejected),
                scale_bar_label=r"50% $\Delta$F/F$_0$",
                title=r"$\Delta$F/F Traces - Rejected ROIs (n={})".format(n_rejected),
            )
        else:
            print("  No rejected ROIs - skipping rejected trace plots")

        # --- NOISE DISTRIBUTIONS (robust to any cell count >= 1) ---
        if n_accepted > 0:
            dff_noise_acc = dff_shot_noise(dffp_acc, fs)
            plot_noise_distribution(dff_noise_acc, output_filename=expected_files["noise_acc"])

        if n_rejected > 0:
            dff_noise_rej = dff_shot_noise(dffp_rej, fs)
            plot_noise_distribution(dff_noise_rej, output_filename=expected_files["noise_rej"])

        # --- SEGMENTATION OVERLAYS ---
        # Suite2p stores images in two coordinate systems:
        # - FULL space: refImg, meanImg, meanImgE (same size as original Ly x Lx)
        # - CROPPED space: max_proj, Vcorr (size determined by yrange/xrange after registration)
        # The stat coordinates are in FULL image space.

        stat_full = res["stat"]  # stat coordinates in full image space

        # Helper to check if image is valid
        def _is_valid_image(img):
            if img is None:
                return False
            if isinstance(img, (int, float)) and img == 0:
                return False
            if isinstance(img, np.ndarray) and img.size == 0:
                return False
            return True

        # Get crop parameters for images in cropped space
        yrange = output_ops.get("yrange", [0, output_ops.get("Ly", 512)])
        xrange = output_ops.get("xrange", [0, output_ops.get("Lx", 512)])
        ymin, xmin = int(yrange[0]), int(xrange[0])

        # Create stat with adjusted coordinates for cropped image space
        if ymin > 0 or xmin > 0:
            stat_cropped = []
            for s in stat_full:
                s_adj = s.copy()
                s_adj["ypix"] = s["ypix"] - ymin
                s_adj["xpix"] = s["xpix"] - xmin
                stat_cropped.append(s_adj)
        else:
            stat_cropped = stat_full

        # Images in FULL space - use stat_full
        full_space_images = {
            "meanImg": ("Mean Image", expected_files["meanImg_segmentation"]),
            "meanImgE": ("Enhanced Mean Image", expected_files["meanImgE_segmentation"]),
        }

        for img_key, (title_name, save_file) in full_space_images.items():
            img = output_ops.get(img_key)
            if _is_valid_image(img):
                if n_accepted > 0:
                    plot_masks(
                        img=img,
                        stat=stat_full,
                        mask_idx=iscell_mask,
                        savepath=save_file,
                        title=f"{title_name} - Accepted ROIs (n={n_accepted})"
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

        # Images in CROPPED space - use stat_cropped
        cropped_space_images = {
            "max_proj": ("Max Projection", expected_files["max_proj_segmentation"]),
        }

        for img_key, (title_name, save_file) in cropped_space_images.items():
            img = output_ops.get(img_key)
            if _is_valid_image(img):
                if n_accepted > 0:
                    plot_masks(
                        img=img,
                        stat=stat_cropped,
                        mask_idx=iscell_mask,
                        savepath=save_file,
                        title=f"{title_name} - Accepted ROIs (n={n_accepted})"
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

        # Correlation image (Vcorr) - in CROPPED space
        vcorr = output_ops.get("Vcorr")
        if _is_valid_image(vcorr):
            # Save correlation image without masks
            fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
            ax.set_facecolor("black")
            ax.imshow(vcorr, cmap="gray")
            ax.set_title("Correlation Image", color="white", fontweight="bold")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(expected_files["correlation_image"], dpi=150, facecolor="black")
            plt.close(fig)

            # Correlation image with segmentation
            if n_accepted > 0:
                plot_masks(
                    img=vcorr,
                    stat=stat_cropped,
                    mask_idx=iscell_mask,
                    savepath=expected_files["correlation_segmentation"],
                    title=f"Correlation Image - Accepted ROIs (n={n_accepted})"
                )

    # --- SUMMARY IMAGES (no masks) - always generated ---
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

    # --- QUALITY DIAGNOSTICS ---
    try:
        plot_plane_diagnostics(plane_dir, save_path=expected_files["quality_diagnostics"])
        print(f"  Saved: {expected_files['quality_diagnostics'].name}")
    except Exception as e:
        print(f"  Failed to generate quality diagnostics: {e}")

    return output_ops
