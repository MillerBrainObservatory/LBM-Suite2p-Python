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
        scale_bar_label: str = "50% ΔF/F₀",
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

    # Normalize background image (handle NaN values from dead zone masking)
    img_min = np.nanmin(img)
    img_ptp = np.nanmax(img) - img_min
    normalized = (img - img_min) / (img_ptp + 1e-6)
    # Set NaN regions to 0 (black background)
    normalized = np.nan_to_num(normalized, nan=0.0)
    canvas = np.tile(normalized, (3, 1, 1)).transpose(1, 2, 0)

    # Assign colors if not provided
    n_masks = mask_idx.sum()
    if colors is None:
        colors = plt.cm.hsv(np.linspace(0, 1, n_masks + 1))[:, :3]  # noqa

    c = 0
    for n, s in enumerate(stat):
        if mask_idx[n]:
            ypix, xpix, lam = s["ypix"], s["xpix"], s["lam"]
            lam = lam / lam.max()
            col = colors[c]
            c += 1
            for k in range(3):
                canvas[ypix, xpix, k] = (
                        0.5 * canvas[ypix, xpix, k] + 0.5 * col[k] * lam
                )

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, interpolation="nearest")
    if title is not None:
        plt.title(title, fontsize=10)
    plt.axis("off")
    plt.tight_layout()

    if savepath:
        if Path(savepath).is_dir():
            raise ValueError("savepath must be a file path, not a directory.")
        plt.savefig(savepath, dpi=300)
        plt.close()
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
        iscell = res["iscell"]
        im = ROI.stats_dicts_to_3d_array(
            stat, Ly=ops["Ly"], Lx=ops["Lx"], label_id=True
        )
        im[im == 0] = np.nan
        accepted_cells = np.sum(iscell)
        rejected_cells = np.sum(~iscell)
        cell_rois = _resize_masks_fit_crop(
            np.nanmax(im[iscell], axis=0) if np.any(iscell) else np.zeros_like(im[0]),
            shape,
        )
        green_overlay = np.zeros((*shape, 4), dtype=np.float32)
        green_overlay[..., 3] = feather_mask(cell_rois > 0, max_alpha=0.9)
        green_overlay[..., 1] = 1
        ax.imshow(green_overlay)
        if not accepted_only:
            non_cell_rois = _resize_masks_fit_crop(
                (
                    np.nanmax(im[~iscell], axis=0)
                    if np.any(~iscell)
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


# =============================================================================
# Publication-Quality Volumetric Figures
# =============================================================================


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
    figsize: tuple = (16, 10),
    style: str = "publication",
) -> plt.Figure:
    """
    Generate publication-quality ROI quality metrics across all planes.

    Creates a multi-panel figure showing:
    - ROI counts per plane (stacked bar: accepted/rejected)
    - Compactness distribution by plane (violin plot)
    - ROI size distribution by plane (violin plot)
    - Summary statistics table

    Parameters
    ----------
    stat : np.ndarray
        Consolidated stat array with 'iplane', 'compact', 'npix' fields.
    iscell : np.ndarray
        Cell classification array (n_rois, 2).
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple, default (16, 10)
        Figure size in inches.
    style : str, default "publication"
        Style preset: "publication" (white bg) or "dark" (black bg).

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
    accepted = iscell[:, 0] == 1

    # Style configuration
    if style == "dark":
        bg_color, text_color = "black", "white"
        accent_colors = {"accepted": "lime", "rejected": "orangered",
                        "violin1": "cyan", "violin2": "magenta"}
    else:
        bg_color, text_color = "white", "black"
        accent_colors = {"accepted": "#2ecc71", "rejected": "#e74c3c",
                        "violin1": "#3498db", "violin2": "#9b59b6"}

    with plt.style.context("default"):
        fig = plt.figure(figsize=figsize, facecolor=bg_color)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Panel 1: ROI counts per plane (stacked bar)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(bg_color)
        counts_acc = [np.sum((plane_nums == p) & accepted) for p in unique_planes]
        counts_rej = [np.sum((plane_nums == p) & ~accepted) for p in unique_planes]

        x = np.arange(n_planes)
        width = 0.7
        ax1.bar(x, counts_acc, width, label="Accepted", color=accent_colors["accepted"], alpha=0.85)
        ax1.bar(x, counts_rej, width, bottom=counts_acc, label="Rejected",
                color=accent_colors["rejected"], alpha=0.85)

        ax1.set_xlabel("Plane", fontweight="bold", fontsize=11, color=text_color)
        ax1.set_ylabel("Number of ROIs", fontweight="bold", fontsize=11, color=text_color)
        ax1.set_title("ROI Counts per Plane", fontweight="bold", fontsize=12, color=text_color)
        ax1.set_xticks(x[::max(1, n_planes // 10)])
        ax1.set_xticklabels([f"{int(p)}" for p in unique_planes[::max(1, n_planes // 10)]])
        ax1.tick_params(colors=text_color)
        ax1.legend(fontsize=9, framealpha=0.9)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        for spine in ax1.spines.values():
            spine.set_color(text_color)

        # Panel 2: Compactness violin plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(bg_color)
        data_compact = [compactness[(plane_nums == p) & accepted & ~np.isnan(compactness)]
                       for p in unique_planes]
        # Filter out empty arrays
        valid_compact = [(i, d) for i, d in enumerate(data_compact) if len(d) > 0]
        if valid_compact:
            positions, data = zip(*valid_compact)
            parts = ax2.violinplot(data, positions=positions, widths=0.7, showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(accent_colors["violin1"])
                pc.set_alpha(0.7)
            for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
                if key in parts:
                    parts[key].set_color(text_color)

        ax2.set_xlabel("Plane", fontweight="bold", fontsize=11, color=text_color)
        ax2.set_ylabel("Compactness", fontweight="bold", fontsize=11, color=text_color)
        ax2.set_title("ROI Compactness (Accepted)", fontweight="bold", fontsize=12, color=text_color)
        ax2.tick_params(colors=text_color)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for spine in ax2.spines.values():
            spine.set_color(text_color)

        # Panel 3: Skewness violin plot
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor(bg_color)
        data_skew = [skewness[(plane_nums == p) & accepted & ~np.isnan(skewness)]
                    for p in unique_planes]
        valid_skew = [(i, d) for i, d in enumerate(data_skew) if len(d) > 0]
        if valid_skew:
            positions, data = zip(*valid_skew)
            parts = ax3.violinplot(data, positions=positions, widths=0.7, showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(accent_colors["violin2"])
                pc.set_alpha(0.7)
            for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
                if key in parts:
                    parts[key].set_color(text_color)

        ax3.set_xlabel("Plane", fontweight="bold", fontsize=11, color=text_color)
        ax3.set_ylabel("Skewness", fontweight="bold", fontsize=11, color=text_color)
        ax3.set_title("Trace Skewness (Accepted)", fontweight="bold", fontsize=12, color=text_color)
        ax3.tick_params(colors=text_color)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        for spine in ax3.spines.values():
            spine.set_color(text_color)

        # Panel 4: ROI size violin plot
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_facecolor(bg_color)
        data_npix = [npix[(plane_nums == p) & accepted] for p in unique_planes]
        valid_npix = [(i, d) for i, d in enumerate(data_npix) if len(d) > 0]
        if valid_npix:
            positions, data = zip(*valid_npix)
            parts = ax4.violinplot(data, positions=positions, widths=0.7, showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(accent_colors["violin1"])
                pc.set_alpha(0.7)
            for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
                if key in parts:
                    parts[key].set_color(text_color)

        ax4.set_xlabel("Plane", fontweight="bold", fontsize=11, color=text_color)
        ax4.set_ylabel("Number of Pixels", fontweight="bold", fontsize=11, color=text_color)
        ax4.set_title("ROI Size (Accepted)", fontweight="bold", fontsize=12, color=text_color)
        ax4.tick_params(colors=text_color)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        for spine in ax4.spines.values():
            spine.set_color(text_color)

        # Panel 5: Acceptance rate by plane
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor(bg_color)
        acceptance_rates = [100 * np.sum((plane_nums == p) & accepted) / max(1, np.sum(plane_nums == p))
                          for p in unique_planes]
        ax5.bar(x, acceptance_rates, width, color=accent_colors["accepted"], alpha=0.85)
        ax5.axhline(np.mean(acceptance_rates), color=accent_colors["rejected"],
                   linestyle="--", linewidth=2, label=f"Mean: {np.mean(acceptance_rates):.1f}%")

        ax5.set_xlabel("Plane", fontweight="bold", fontsize=11, color=text_color)
        ax5.set_ylabel("Acceptance Rate (%)", fontweight="bold", fontsize=11, color=text_color)
        ax5.set_title("Cell Classification Rate", fontweight="bold", fontsize=12, color=text_color)
        ax5.set_xticks(x[::max(1, n_planes // 10)])
        ax5.set_xticklabels([f"{int(p)}" for p in unique_planes[::max(1, n_planes // 10)]])
        ax5.tick_params(colors=text_color)
        ax5.legend(fontsize=9, framealpha=0.9)
        ax5.set_ylim(0, 100)
        ax5.spines["top"].set_visible(False)
        ax5.spines["right"].set_visible(False)
        for spine in ax5.spines.values():
            spine.set_color(text_color)

        # Panel 6: Summary statistics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor(bg_color)
        ax6.axis("off")

        total_rois = len(stat)
        total_acc = accepted.sum()
        total_rej = (~accepted).sum()
        mean_compact = np.nanmean(compactness[accepted])
        mean_npix = np.mean(npix[accepted])
        mean_skew = np.nanmean(skewness[accepted])

        summary_text = (
            f"{'Summary Statistics':^30}\n"
            f"{'─' * 30}\n"
            f"{'Total ROIs:':<20}{total_rois:>10,}\n"
            f"{'Accepted:':<20}{total_acc:>10,} ({100*total_acc/total_rois:.1f}%)\n"
            f"{'Rejected:':<20}{total_rej:>10,} ({100*total_rej/total_rois:.1f}%)\n"
            f"{'Planes:':<20}{n_planes:>10}\n"
            f"{'─' * 30}\n"
            f"{'Mean Compactness:':<20}{mean_compact:>10.2f}\n"
            f"{'Mean ROI Size:':<20}{mean_npix:>10.0f} px\n"
            f"{'Mean Skewness:':<20}{mean_skew:>10.2f}\n"
        )

        ax6.text(
            0.5, 0.5, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment="center", horizontalalignment="center",
            family="monospace", color=text_color,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color,
                     edgecolor=text_color, alpha=0.8)
        )

        # Main title
        fig.suptitle(
            "Volume Quality Metrics", fontsize=14, fontweight="bold",
            color=text_color, y=0.98
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=bg_color)
            plt.close(fig)
        else:
            plt.show()

    return fig


def plot_trace_analysis(
    F: np.ndarray,
    Fneu: np.ndarray,
    stat: np.ndarray,
    iscell: np.ndarray,
    ops: dict,
    save_path: str | Path = None,
    figsize: tuple = (18, 12),
    style: str = "publication",
) -> Tuple[plt.Figure, dict]:
    """
    Generate comprehensive trace analysis figure for volumetric data.

    Creates a multi-panel figure showing:
    - Example traces from different planes
    - SNR distribution per plane
    - Mean fluorescence per plane
    - Activity correlation matrix
    - Activity heatmap (sorted by activity level)

    Parameters
    ----------
    F : np.ndarray
        Fluorescence traces array (n_rois, n_frames).
    Fneu : np.ndarray
        Neuropil fluorescence array (n_rois, n_frames).
    stat : np.ndarray
        Stat array with 'iplane' field.
    iscell : np.ndarray
        Cell classification array.
    ops : dict
        Ops dictionary with 'fs' (frame rate) field.
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple, default (18, 12)
        Figure size in inches.
    style : str, default "publication"
        Style preset: "publication" or "dark".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    metrics : dict
        Dictionary containing computed metrics (snr, dff, mean_F).

    Examples
    --------
    >>> fig, metrics = plot_trace_analysis(F, Fneu, stat, iscell, ops)
    >>> print(f"Mean SNR: {np.mean(metrics['snr']):.2f}")
    """
    accepted = iscell[:, 0] == 1
    F_acc = F[accepted]
    Fneu_acc = Fneu[accepted]
    stat_acc = stat[accepted]
    plane_nums = np.array([s.get("iplane", 0) for s in stat_acc])
    unique_planes = np.unique(plane_nums)
    fs = ops.get("fs", 30.0)

    # Compute ΔF/F
    F_corrected = F_acc - 0.7 * Fneu_acc
    baseline = np.percentile(F_corrected, 20, axis=1, keepdims=True)
    baseline = np.maximum(baseline, 1e-6)  # Prevent division by zero
    dff = (F_corrected - baseline) / baseline

    # Compute SNR (signal-to-noise ratio)
    signal = np.std(dff, axis=1)
    noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.6745  # MAD estimator
    snr = signal / (noise + 1e-6)

    # Mean fluorescence
    mean_F = np.mean(F_acc, axis=1)

    # Style configuration
    if style == "dark":
        bg_color, text_color = "black", "white"
        cmap_traces = "tab10"
        cmap_heatmap = "magma"
        cmap_corr = "RdBu_r"
    else:
        bg_color, text_color = "white", "black"
        cmap_traces = "tab10"
        cmap_heatmap = "viridis"
        cmap_corr = "RdBu_r"

    with plt.style.context("default"):
        fig = plt.figure(figsize=figsize, facecolor=bg_color)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                              height_ratios=[1, 1, 1.2])

        # Panel 1: Example traces from different planes (spans top row)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor(bg_color)

        n_examples = min(5, len(unique_planes))
        colors = plt.cm.get_cmap(cmap_traces)(np.linspace(0, 1, n_examples))

        offset = 0
        time = np.arange(min(3000, dff.shape[1])) / fs  # Show first ~100s
        for i, p in enumerate(unique_planes[:n_examples]):
            plane_mask = plane_nums == p
            if plane_mask.sum() > 0:
                # Get highest SNR cell from this plane
                plane_snr = snr[plane_mask]
                best_idx = np.where(plane_mask)[0][np.argmax(plane_snr)]
                trace = dff[best_idx, :len(time)]
                ax1.plot(time, trace + offset, color=colors[i], linewidth=0.8,
                        label=f"Plane {int(p)}")
                offset += np.percentile(np.abs(trace), 99) * 1.5

        ax1.set_xlabel("Time (s)", fontweight="bold", fontsize=11, color=text_color)
        ax1.set_ylabel("ΔF/F (offset)", fontweight="bold", fontsize=11, color=text_color)
        ax1.set_title("Example Traces (Highest SNR per Plane)", fontweight="bold",
                     fontsize=12, color=text_color)
        ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax1.tick_params(colors=text_color)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        for spine in ax1.spines.values():
            spine.set_color(text_color)

        # Panel 2: SNR distribution per plane
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor(bg_color)
        snr_by_plane = [snr[plane_nums == p] for p in unique_planes]
        valid_snr = [(i, d) for i, d in enumerate(snr_by_plane) if len(d) > 0]
        if valid_snr:
            positions, data = zip(*valid_snr)
            parts = ax2.violinplot(data, positions=positions, widths=0.7, showmeans=True)
            for pc in parts["bodies"]:
                pc.set_facecolor("#f39c12")
                pc.set_alpha(0.7)
            for key in ["cmeans", "cbars", "cmins", "cmaxes"]:
                if key in parts:
                    parts[key].set_color(text_color)

        ax2.axhline(2, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7, label="SNR=2")
        ax2.set_xlabel("Plane", fontweight="bold", fontsize=11, color=text_color)
        ax2.set_ylabel("SNR", fontweight="bold", fontsize=11, color=text_color)
        ax2.set_title("Signal-to-Noise Ratio", fontweight="bold", fontsize=12, color=text_color)
        ax2.legend(fontsize=9, framealpha=0.9)
        ax2.tick_params(colors=text_color)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for spine in ax2.spines.values():
            spine.set_color(text_color)

        # Panel 3: Mean fluorescence per plane
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor(bg_color)
        mean_F_by_plane = [mean_F[plane_nums == p] for p in unique_planes]
        valid_F = [(i, d) for i, d in enumerate(mean_F_by_plane) if len(d) > 0]
        if valid_F:
            positions, data = zip(*valid_F)
            parts = ax3.violinplot(data, positions=positions, widths=0.7, showmeans=True)
            for pc in parts["bodies"]:
                pc.set_facecolor("#3498db")
                pc.set_alpha(0.7)
            for key in ["cmeans", "cbars", "cmins", "cmaxes"]:
                if key in parts:
                    parts[key].set_color(text_color)

        ax3.set_xlabel("Plane", fontweight="bold", fontsize=11, color=text_color)
        ax3.set_ylabel("Mean Fluorescence (a.u.)", fontweight="bold", fontsize=11, color=text_color)
        ax3.set_title("Mean Fluorescence per Plane", fontweight="bold", fontsize=12, color=text_color)
        ax3.tick_params(colors=text_color)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        for spine in ax3.spines.values():
            spine.set_color(text_color)

        # Panel 4: Activity correlation matrix (sampled)
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_facecolor(bg_color)
        n_sample = min(100, len(dff))
        if n_sample > 1:
            sample_idx = np.random.choice(len(dff), n_sample, replace=False)
            corr_matrix = np.corrcoef(dff[sample_idx])
            im = ax4.imshow(corr_matrix, cmap=cmap_corr, vmin=-0.5, vmax=0.5, aspect="auto")
            cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
            cbar.set_label("Correlation", fontsize=10, color=text_color)
            cbar.ax.tick_params(colors=text_color)

        ax4.set_xlabel("Cell Index", fontweight="bold", fontsize=11, color=text_color)
        ax4.set_ylabel("Cell Index", fontweight="bold", fontsize=11, color=text_color)
        ax4.set_title(f"Correlation Matrix (n={n_sample})", fontweight="bold",
                     fontsize=12, color=text_color)
        ax4.tick_params(colors=text_color)

        # Panel 5: Activity heatmap (spans bottom row)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.set_facecolor(bg_color)

        # Sort by activity level (number of high ΔF/F events)
        activity = np.sum(dff > 1, axis=1)
        n_show = min(200, len(dff))
        sort_idx = np.argsort(activity)[::-1][:n_show]

        # Temporal downsampling for visualization
        downsample = max(1, dff.shape[1] // 2000)
        dff_plot = dff[sort_idx, ::downsample]

        im = ax5.imshow(
            dff_plot, aspect="auto", cmap=cmap_heatmap,
            vmin=0, vmax=np.percentile(dff_plot, 99),
            extent=[0, dff.shape[1] / fs, n_show, 0]
        )
        cbar = plt.colorbar(im, ax=ax5, shrink=0.6, pad=0.02)
        cbar.set_label("ΔF/F", fontsize=10, color=text_color)
        cbar.ax.tick_params(colors=text_color)

        ax5.set_xlabel("Time (s)", fontweight="bold", fontsize=11, color=text_color)
        ax5.set_ylabel("Cell (sorted by activity)", fontweight="bold", fontsize=11, color=text_color)
        ax5.set_title(f"Activity Heatmap (Top {n_show} Active Cells)", fontweight="bold",
                     fontsize=12, color=text_color)
        ax5.tick_params(colors=text_color)

        # Main title
        fig.suptitle(
            "Volumetric Trace Analysis", fontsize=14, fontweight="bold",
            color=text_color, y=0.99
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=bg_color)
            plt.close(fig)
        else:
            plt.show()

    metrics = {"snr": snr, "dff": dff, "mean_F": mean_F}
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
    Create a comprehensive summary table of volumetric processing results.

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
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    Generate a single-figure diagnostic summary for a processed plane.

    Creates a publication-quality 6-panel figure showing:
    - Mean image with ROI outlines
    - ROI size distribution (accepted vs rejected)
    - SNR distribution with quality threshold
    - Example traces (top SNR cells)
    - Compactness vs SNR scatter
    - Summary statistics text

    Parameters
    ----------
    plane_dir : str or Path
        Path to the plane directory containing ops.npy, stat.npy, etc.
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple, default (16, 12)
        Figure size in inches.

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

    accepted = iscell[:, 0] == 1
    n_total = len(stat)
    n_accepted = accepted.sum()
    n_rejected = (~accepted).sum()

    # Compute metrics
    F_corr = F - 0.7 * Fneu
    baseline = np.percentile(F_corr, 20, axis=1, keepdims=True)
    baseline = np.maximum(baseline, 1e-6)
    dff = (F_corr - baseline) / baseline

    # SNR calculation
    signal = np.std(dff, axis=1)
    noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.6745
    snr = signal / (noise + 1e-6)

    # Extract ROI properties
    npix = np.array([s.get("npix", 0) for s in stat])
    compactness = np.array([s.get("compact", np.nan) for s in stat])
    fs = ops.get("fs", 30.0)

    # Create figure
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Mean image with ROI outlines (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    mean_img = ops.get("meanImgE", ops.get("meanImg"))
    if mean_img is not None:
        vmin, vmax = np.nanpercentile(mean_img, [1, 99])
        ax1.imshow(mean_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

        # Draw accepted ROIs
        for s in stat[accepted]:
            ypix, xpix = s["ypix"], s["xpix"]
            ax1.scatter(xpix, ypix, c="lime", s=0.2, alpha=0.5, linewidths=0)

    ax1.set_title(f"ROI Detection: {n_accepted} accepted / {n_rejected} rejected",
                  fontweight="bold", fontsize=11)
    ax1.axis("off")

    # Panel 2: Summary statistics
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")

    snr_acc = snr[accepted] if n_accepted > 0 else np.array([0])
    high_snr_pct = 100 * np.sum(snr_acc > 2) / max(1, len(snr_acc))

    summary_text = (
        f"{'Plane Summary':^28}\n"
        f"{'─' * 28}\n"
        f"{'Total ROIs:':<18}{n_total:>10,}\n"
        f"{'Accepted:':<18}{n_accepted:>10,}\n"
        f"{'Rejected:':<18}{n_rejected:>10,}\n"
        f"{'Accept Rate:':<18}{100*n_accepted/max(1,n_total):>9.1f}%\n"
        f"{'─' * 28}\n"
        f"{'Mean SNR:':<18}{np.mean(snr_acc):>10.2f}\n"
        f"{'Median SNR:':<18}{np.median(snr_acc):>10.2f}\n"
        f"{'SNR > 2:':<18}{high_snr_pct:>9.1f}%\n"
        f"{'─' * 28}\n"
        f"{'Mean Size:':<18}{np.mean(npix[accepted]):>8.0f} px\n"
        f"{'Frame Rate:':<18}{fs:>9.1f} Hz\n"
        f"{'Frames:':<18}{F.shape[1]:>10,}\n"
    )

    ax2.text(
        0.5, 0.5, summary_text, transform=ax2.transAxes,
        fontsize=10, verticalalignment="center", horizontalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9)
    )

    # Panel 3: ROI size distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if n_accepted > 0:
        ax3.hist(npix[accepted], bins=30, alpha=0.7, color="#2ecc71", label="Accepted", edgecolor="white")
    if n_rejected > 0:
        ax3.hist(npix[~accepted], bins=30, alpha=0.5, color="#e74c3c", label="Rejected", edgecolor="white")
    ax3.set_xlabel("ROI Size (pixels)", fontweight="bold")
    ax3.set_ylabel("Count", fontweight="bold")
    ax3.set_title("ROI Size Distribution", fontweight="bold", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Panel 4: SNR distribution
    ax4 = fig.add_subplot(gs[1, 1])
    if n_accepted > 0:
        ax4.hist(snr[accepted], bins=30, alpha=0.7, color="#3498db", label="Accepted", edgecolor="white")
    ax4.axvline(2, color="#e74c3c", linestyle="--", linewidth=2, label="SNR=2 threshold")
    ax4.axvline(np.median(snr_acc), color="#2ecc71", linestyle="-", linewidth=2,
                label=f"Median={np.median(snr_acc):.1f}")
    ax4.set_xlabel("Signal-to-Noise Ratio", fontweight="bold")
    ax4.set_ylabel("Count", fontweight="bold")
    ax4.set_title("SNR Distribution (Accepted)", fontweight="bold", fontsize=11)
    ax4.legend(fontsize=9)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Panel 5: Compactness vs SNR scatter
    ax5 = fig.add_subplot(gs[1, 2])
    if n_accepted > 0:
        valid_mask = accepted & ~np.isnan(compactness)
        if valid_mask.sum() > 0:
            sc = ax5.scatter(compactness[valid_mask], snr[valid_mask],
                           c=npix[valid_mask], cmap="viridis", alpha=0.6, s=20)
            cbar = plt.colorbar(sc, ax=ax5, shrink=0.8)
            cbar.set_label("ROI Size (px)", fontsize=9)
    ax5.axhline(2, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7)
    ax5.set_xlabel("Compactness", fontweight="bold")
    ax5.set_ylabel("SNR", fontweight="bold")
    ax5.set_title("Quality Metrics (Accepted)", fontweight="bold", fontsize=11)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # Panel 6: Example traces (spans bottom row)
    ax6 = fig.add_subplot(gs[2, :])

    if n_accepted > 0:
        # Get top 10 SNR cells
        n_traces = min(10, n_accepted)
        top_snr_idx = np.argsort(snr[accepted])[::-1][:n_traces]
        accepted_idx = np.where(accepted)[0]
        cells_to_plot = accepted_idx[top_snr_idx]

        time = np.arange(min(3000, dff.shape[1])) / fs
        colors = plt.cm.tab10(np.linspace(0, 1, n_traces))

        offset = 0
        for i, cell_idx in enumerate(cells_to_plot):
            trace = dff[cell_idx, :len(time)]
            ax6.plot(time, trace + offset, color=colors[i], linewidth=0.8,
                    label=f"Cell {cell_idx} (SNR={snr[cell_idx]:.1f})")
            offset += np.percentile(np.abs(trace), 99) * 1.5

        ax6.set_xlabel("Time (s)", fontweight="bold")
        ax6.set_ylabel("ΔF/F (offset)", fontweight="bold")
        ax6.set_title(f"Top {n_traces} Cells by SNR", fontweight="bold", fontsize=11)
        ax6.legend(loc="upper right", fontsize=8, ncol=2)
    else:
        ax6.text(0.5, 0.5, "No accepted cells", transform=ax6.transAxes,
                ha="center", va="center", fontsize=14)

    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    # Main title
    plane_name = plane_dir.name
    fig.suptitle(f"Quality Diagnostics: {plane_name}", fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
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

    expected_files = {
        "ops": plane_dir / "ops.npy",
        "stat": plane_dir / "stat.npy",
        "iscell": plane_dir / "iscell.npy",
        "registration": plane_dir / "registration.png",
        "segmentation_accepted": plane_dir / "segmentation_accepted.png",
        "segmentation_rejected": plane_dir / "segmentation_rejected.png",
        "area_filter": plane_dir / "segmentation_rejected_area_filter.png",
        "segmentation_filtered": plane_dir / "segmentation_rejected.png",
        "max_proj": plane_dir / "max_projection_image.png",
        "meanImg": plane_dir / "mean_image.png",
        "meanImgE": plane_dir / "mean_image_enhanced.png",
        "traces_raw": plane_dir / "traces_raw.png",
        "traces_dff": plane_dir / "traces_dff.png",
        "traces_noise": plane_dir / "traces_noise.png",
        "traces_area": plane_dir / "traces_rejected_area_filter.png",
        "noise_acc": plane_dir / "shot_noise_distrubution_accepted.png",
        "noise_rej": plane_dir / "shot_noise_distrubution_rejected.png",
        "model": plane_dir / "model.npy",
        "rastermap": plane_dir / "rastermap.png",
    }

    output_ops = load_ops(expected_files["ops"])

    # Dead zones are now handled via yrange/xrange cropping in run_lsp.py
    # so we don't need to mask them here anymore
    # output_ops = mask_dead_zones_in_ops(output_ops)

    # force remake of the heavy figures
    for key in [
        "registration",
        "segmentation_accepted",
        "segmentation_rejected",
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
        iscell = res["iscell"]
        iscell_mask = (
            iscell[:, 0].astype(bool) if iscell.ndim == 2 else iscell.astype(bool)
        )

        spks = res["spks"]
        F = res["F"]

        n_neurons = F.shape[0]
        if n_neurons < 10:
            return output_ops

        # rastermap model
        F_accepted = F[iscell_mask]
        F_rejected = F[~iscell_mask]
        spks_cells = spks[iscell_mask]

        n_accepted = F_accepted.shape[0]
        n_rejected = F_rejected.shape[0]
        print(f"Plotting results for {n_accepted} accepted / {n_rejected} rejected ROIs")

        model = None
        if run_rastermap:
            try:
                from lbm_suite2p_python.zplane import plot_rastermap
                import rastermap

                has_rastermap = True
            except ImportError:
                print(
                    "rastermap package not found, skipping rastermap plotting. \n"
                    "Install via `pip install rastermap` or set run_rastermap=False \n"
                    "for run_plane(), run_volume(), or plot_rastermap() to work."
                )
                has_rastermap = False
                rastermap, plot_rastermap = None, None
            if expected_files["model"].is_file():
                model = np.load(expected_files["model"], allow_pickle=True).item()
            elif has_rastermap:
                params = {
                    "n_clusters": 100 if n_neurons >= 200 else None,
                    "n_PCs": min(128, max(2, n_neurons - 1)),
                    "locality": 0.0 if n_neurons >= 200 else 0.1,
                    "time_lag_window": 15,
                    "grid_upsample": 10 if n_neurons >= 200 else 0,
                }
                model = rastermap.Rastermap(**params).fit(spks_cells)
                np.save(expected_files["model"], model)

                plot_rastermap(
                    spks_cells,
                    model,
                    neuron_bin_size=0,
                    save_path=expected_files["rastermap"],
                    title_kwargs={"fontsize": 8, "y": 0.95},
                    title="Rastermap Sorted Activity",
                )

            if model is not None:
                # indices of cells relative to *all* ROIs
                isort_global = np.where(iscell_mask)[0][model.isort]
                output_ops["isort"] = isort_global

                # reorder just the cells
                F_accepted = F_accepted[model.isort]

        # compute dF/F
        # f_norm_acc = normalize_traces(F_accepted, mode="percentile")
        # f_norm_rej = normalize_traces(F_rejected, mode="percentile")
        f_norm_acc = F_accepted
        f_norm_rej = F_rejected

        dffp_acc = (
            dff_rolling_percentile(
                f_norm_acc, percentile=dff_percentile, window_size=dff_window_size
            )
            * 100
        )
        dffp_rej = (
            dff_rolling_percentile(
                f_norm_rej, percentile=dff_percentile, window_size=dff_window_size
            )
            * 100
        )

        # Plot traces for accepted cells (if any exist)
        if n_accepted > 0:
            plot_traces(
                dffp_acc,
                save_path=expected_files["traces_dff"],
                num_neurons=min(output_ops.get("plot_n_traces", 30), n_accepted),
                scale_bar_label="50% ΔF/F₀",
                title=f"Accepted Cells (n={n_accepted})",
            )
            plot_traces(
                f_norm_acc,
                save_path=expected_files["traces_raw"],
                num_neurons=min(output_ops.get("plot_n_traces", 30), n_accepted),
                scale_bar_label="a.u.",
                title=f"Accepted Cells (n={n_accepted})",
            )
        else:
            print(f"No accepted cells to plot traces for")

        # Plot traces for rejected cells (if any exist)
        if n_rejected > 0:
            plot_traces(
                dffp_rej,
                save_path=expected_files["traces_noise"],
                num_neurons=min(output_ops.get("plot_n_traces", 30), n_rejected),
                scale_bar_label="50% ΔF/F₀",
                title=f"Rejected Cells (n={n_rejected})",
            )
        else:
            print(f"No rejected cells to plot traces for")

        fs = output_ops.get("fs", 1.0)
        dff_noise_acc = dff_shot_noise(dffp_acc, fs) if n_accepted > 0 else np.array([])
        dff_noise_rej = dff_shot_noise(dffp_rej, fs) if n_rejected > 0 else np.array([])

        if n_accepted > 0:
            plot_noise_distribution(
                dff_noise_acc, output_filename=expected_files["noise_acc"]
            )
        if n_rejected > 0:
            plot_noise_distribution(
                dff_noise_rej, output_filename=expected_files["noise_rej"]
            )
        # Use the image that was actually used for detection
        # For anatomical: ops["Vcorr"] contains the detection image (in CROPPED space)
        # stat coordinates are in FULL image space (after yrange/xrange adjustment in detect.py)
        # So we need to adjust coordinates back to cropped space to match Vcorr

        # Prefer Vcorr (actual detection image) when available
        detection_img = output_ops.get("Vcorr")
        stat_to_plot = res["stat"]

        # Check if Vcorr is valid and not a placeholder
        if detection_img is None or (isinstance(detection_img, (int, float)) and detection_img == 0):
            # Fallback to full images that match stat coordinate space
            detection_img = output_ops.get("meanImgE")
            if detection_img is None:
                detection_img = output_ops.get("meanImg")
        else:
            # Vcorr is in cropped space - need to offset stat coordinates
            ymin = int(output_ops.get("yrange", [0])[0])
            xmin = int(output_ops.get("xrange", [0])[0])

            # Create temporary stat with adjusted coordinates for cropped space
            stat_to_plot = []
            for s in res["stat"]:
                s_adj = s.copy()
                s_adj["ypix"] = s["ypix"] - ymin
                s_adj["xpix"] = s["xpix"] - xmin
                stat_to_plot.append(s_adj)

        if detection_img is not None:
            plot_masks(
                img=detection_img,
                stat=stat_to_plot,
                mask_idx=iscell_mask,
                savepath=expected_files["segmentation_accepted"],
                title="Accepted ROIs"
            )
        else:
            print("WARNING: No valid background image found for mask overlay")

        # iscell_area = filter_by_area(iscell_mask, res["stat"])
        # eliminated_area = iscell_mask & ~iscell_area
        # plot_masks(
        #     img=output_ops.get("meanImgE"),
        #     stat=res["stat"],
        #     mask_idx=eliminated_area,
        #     savepath=expected_files["area_filter"],
        #     title="Cells Rejected: Area filter"
        # )
        # plot_traces(
        #     F,
        #     save_path=expected_files["traces_area"],
        #     cell_indices=eliminated_area,
        #     title="Traces eliminated by Area filter",
        #     fps=output_ops["fs"],
        # )

    fig_label = kwargs.get("fig_label", plane_dir.stem)
    for key in ["meanImg", "max_proj", "meanImgE"]:
        if key in output_ops:
            plot_projection(
                output_ops,
                expected_files[key],
                fig_label=fig_label,
                display_masks=False,
                add_scalebar=True,
                proj=key,
            )

    # Generate single-figure diagnostic summary
    try:
        diagnostics_path = plane_dir / "quality_diagnostics.png"
        plot_plane_diagnostics(plane_dir, save_path=diagnostics_path)
        print(f"  Saved: quality_diagnostics.png")
    except Exception as e:
        print(f"  Failed to generate quality diagnostics: {e}")

    return output_ops
