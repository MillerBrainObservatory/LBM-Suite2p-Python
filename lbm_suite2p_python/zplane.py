from pathlib import Path

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
        signal_units=None,
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
    signal_units : str, optional
        Units of fluorescence signal.
    cell_indices : array-like or None
        Specific cell indices to plot. If provided, overrides num_neurons.
    """
    if isinstance(f, dict):
        raise ValueError("f must be a numpy array, not a dictionary")

    if signal_units is None:
        signal_units = infer_units(f)

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

    cmap_inst = plt.get_cmap(cmap)
    colors = cmap_inst(np.linspace(0, 1, displayed_neurons))
    perm = get_color_permutation(displayed_neurons)
    colors = colors[perm]

    # fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    # ax.set_facecolor("black")

    # build a composite array
    # each pixel is the value of the lowest trace at that timepoint
    composite = np.full_like(f[:displayed_neurons, :current_frame + 1], np.nan)
    for i in range(displayed_neurons):
        trace = f[indices[i], : current_frame + 1]
        baseline = np.percentile(trace, 8)
        shifted = (trace - baseline) + i * offset
        if i == 0:
            composite[i] = shifted
        else:
            # keep only parts that are strictly above all lower traces
            below = np.nanmax(composite[:i], axis=0)
            masked = np.where(shifted > below, shifted, np.nan)
            composite[i] = masked

    # plot only the visible parts
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.tick_params(axis="x", which="both", labelbottom=False, length=0, colors="white")
    ax.tick_params(axis="y", which="both", labelleft=False, length=0, colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(displayed_neurons):
        ax.plot(
            data_time[: current_frame + 1],
            composite[i],
            color=colors[i],
            lw=lw,
            zorder=-i,
        )

    all_shifted = [
        (f[i, : current_frame + 1] - np.percentile(f[i, : current_frame + 1], 10))
        + i * offset
        for i in range(displayed_neurons)
    ]
    all_y = np.concatenate(all_shifted)
    y_min, y_max = np.min(all_y), np.max(all_y)

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

    # how much signal change corresponds to 10% of the y-axis span
    vertical_bar_height = 0.1 * (y_max - y_min)

    # express it directly in the same units as f (no offset normalization)
    rounded_signal_units = np.round(vertical_bar_height, 2)

    if signal_units == "raw":
        dff_label = f"{rounded_signal_units:.2f} raw signal (a.u)"
    elif signal_units == "dff":
        dff_label = f"{rounded_signal_units:.2f} ΔF/F₀"
    elif signal_units == "dffp":
        dff_label = f"{rounded_signal_units:.2f} % ΔF/F₀"
    else:
        dff_label = f"{rounded_signal_units:.2f}"

    vsb = AnchoredVScaleBar(
        height=0.1,
        label=dff_label,
        loc="lower right",  # noqa
        frameon=False,
        pad=-0.1,
        sep=4,
        linekw=linekw,
        ax=ax,
        spacer_width=0,
    )
    vsb.set_bbox_to_anchor((1.00, 0.05), transform=ax.transAxes)  # noqa
    # vsb.set_bbox_to_anchor(, transform=ax.transAxes)
    vsb.txt._text.set_color("white")  # noqa
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

    # Normalize background image
    canvas = np.tile(
        (img - img.min()) / (np.ptp(img) + 1e-6), (3, 1, 1)
    ).transpose(1, 2, 0)

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

    alt_tiff = savepath.with_name(savepath.stem + "_alternating.tif")
    tifffile.imwrite(
        alt_tiff,
        np.stack(alt_frames, axis=0),
        imagej=True,
        metadata={"Labels": alt_labels},
    )
    print(f"Saved alternating TIFF to {alt_tiff}")

    # ----------------
    # 2. Panel TIFF
    # ----------------
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

    # ----------------
    # 3. CSV metrics
    # ----------------
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
        "alternating_tiff": alt_tiff,
        "panel_tiff": panel_tiff,
        "metrics_csv": csv_path,
    }


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

        if n_neurons >= 30:
            plot_traces(
                dffp_acc,
                save_path=expected_files["traces_dff"],
                num_neurons=output_ops.get("plot_n_traces", 30),
                signal_units="dffp",
            )
            plot_traces(
                f_norm_acc,
                save_path=expected_files["traces_raw"],
                num_neurons=output_ops.get("plot_n_traces", 30),
                signal_units="raw",
            )

        fs = output_ops.get("fs", 1.0)
        dff_noise_acc = dff_shot_noise(dffp_acc, fs)
        dff_noise_rej = dff_shot_noise(dffp_rej, fs)
        plot_noise_distribution(
            dff_noise_acc, output_filename=expected_files["noise_acc"]
        )
        plot_noise_distribution(
            dff_noise_rej, output_filename=expected_files["noise_rej"]
        )
        plot_masks(
            img=output_ops.get("meanImgE"),
            stat=res["stat"],
            mask_idx=iscell_mask,
            savepath=expected_files["segmentation_accepted"],
            title="Accepted ROIs"
        )

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

    return output_ops
