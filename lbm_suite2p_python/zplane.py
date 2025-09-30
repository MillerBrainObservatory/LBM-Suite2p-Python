from pathlib import Path

import numpy as np
import math

import matplotlib.offsetbox
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import VPacker, HPacker, DrawingArea
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

from scipy.ndimage import distance_transform_edt

from lbm_suite2p_python.utils import dff_rolling_percentile, _resize_masks_fit_crop, load_planar_results, bin1d, \
    load_ops
from suite2p.detection.stats import ROI
from skimage.segmentation import find_boundaries


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
        fps=17.0,
        num_neurons=20,
        window=220,
        title="",
        offset=None,
        lw=0.5,
        cmap="tab10",
        signal_units=None,
):
    """
    Plot stacked fluorescence traces with automatic offset and scale bars.

    Parameters
    ----------
    f : ndarray
        2d array of fluorescence traces (n_neurons x n_timepoints).
    save_path : str, optional
        Path to save the output plot (default is "./stacked_traces.png").
    fps : float, optional
        Sampling rate in frames per second (default is 17.0).
    num_neurons : int, optional
        Number of neurons to display (default is 20).
    window : float, optional
        Time window (in seconds) to display (default is 120).
    title : str, optional
        Title of the figure (default is "").
    offset : float or None, optional
        Vertical offset between traces; if None, computed automatically.
    lw : float, optional
        Line width for data points.
    cmap : str, optional
        Matplotlib colormap string (default is 'tab10').
    signal_units : str, optional
        Units of fluorescence signal. Options: "raw", "dff", "dffp", if None will infer from percentile,
        recommended to keep None unless units are misinterpreted.
    """
    if isinstance(f, dict):
        ops = f
        print("Loading dff (%) from ops-dict")
        res = load_planar_results(ops)
        f = res["F"]
        percentile = ops.get("dff_percentile", 20)
        window = ops.get("dff_window_size", window)
        f = dff_rolling_percentile(f, percentile=percentile, window_size=window) * 100
        signal_units = "dffp"

    if signal_units is None:
        signal_units = infer_units(f)

    displayed_neurons = min(num_neurons, f.shape[0])
    n_timepoints = f.shape[-1]
    data_time = np.arange(n_timepoints) / fps
    current_frame = min(int(window * fps), n_timepoints - 1)

    if offset is None:
        p10 = np.percentile(f[:displayed_neurons, : current_frame + 1], 10, axis=1)
        p90 = np.percentile(f[:displayed_neurons, : current_frame + 1], 90, axis=1)
        offset = np.median(p90 - p10) * 1.2

    cmap_inst = plt.get_cmap(cmap)
    colors = cmap_inst(np.linspace(0, 1, displayed_neurons))
    perm = get_color_permutation(displayed_neurons)
    colors = colors[perm]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.tick_params(axis="x", which="both", labelbottom=False, length=0, colors="white")
    ax.tick_params(axis="y", which="both", labelleft=False, length=0, colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in reversed(range(displayed_neurons)):
        trace = f[i, : current_frame + 1]
        baseline = np.percentile(trace, 8)
        shifted_trace = (trace - baseline) + i * offset

        ax.plot(
            data_time[: current_frame + 1],
            shifted_trace,
            color=colors[i],
            lw=lw,
            zorder=-i,
        )

        if i < displayed_neurons - 1:
            prev_trace = f[i + 1, : current_frame + 1]
            prev_baseline = np.percentile(prev_trace, 8)
            prev_shifted = (prev_trace - prev_baseline) + (i + 1) * offset
            mask = shifted_trace > prev_shifted
            ax.fill_between(
                data_time[: current_frame + 1],
                shifted_trace,
                prev_shifted,
                where=mask,
                color="black",
                zorder=-i - 1,
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

    dff_bar_height = 0.1 * (y_max - y_min)
    rounded_dff = round(dff_bar_height / 5) * 5

    if signal_units == "raw":
        dff_label = f"{rounded_dff:.0f} raw signal (a.u)"
    elif signal_units == "dff":
        dff_label = f"{rounded_dff:.0f} ΔF/F₀"
    elif signal_units == "dffp":
        dff_label = f"{rounded_dff:.0f} % ΔF/F₀"
    else:
        print(f"unknown label: {signal_units}")
        dff_label = "Unknown"

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

    return fig, colors


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
            ease = 3 * u ** 2 - 2 * u ** 3  # smoothstep easing
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
            (f[ix, i_lower:i_upper] - np.percentile(f[ix, i_lower:i_upper], 8)) + ix * gap
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

def _draw_masks(stat, meanImg, mask_idx, fname, out_prefix, outpath):
    canvas = np.tile(
        (meanImg - meanImg.min()) / (np.ptp(meanImg) + 1e-6),
        (3, 1, 1)
    ).transpose(1, 2, 0)  # grayscale RGB

    colors = plt.cm.hsv(np.linspace(0, 1, mask_idx.sum() + 1))  # noqa

    c = 0
    for n, s in enumerate(stat):
        if mask_idx[n]:
            ypix = s["ypix"]
            xpix = s["xpix"]
            lam = s["lam"] / s["lam"].max()
            col = colors[c][:3]  # RGB
            c += 1
            for k in range(3):
                canvas[ypix, xpix, k] = (
                        0.5 * canvas[ypix, xpix, k] + 0.5 * col[k] * lam
                )

    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)
    outfile = outpath / f"{out_prefix}_{fname}.png"

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved {outfile}")


def plot_masks(plane_dir, out_prefix="rois", output_directory=None):
    """
    Make ROI overlays like Suite2p GUI: one PNG for accepted and one for rejected.

    Parameters
    ----------
    plane_dir : str or Path
        Directory containing ops.npy, stat.npy, iscell.npy
    out_prefix : str
        Prefix for output files (saved in plane_dir by default)
    output_directory : str or Path, optional
        If given, save to this directory instead of plane_dir
    """
    plane_dir = Path(plane_dir)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    iscell = np.load(plane_dir / "iscell.npy")

    Ly, Lx = ops["Ly"], ops["Lx"]
    meanImg = ops.get("meanImgE", ops.get("meanImg", np.zeros((Ly, Lx))))

    if output_directory is None:
        output_directory = plane_dir

    _draw_masks(stat, meanImg, iscell[:, 0] == 1, "accepted", out_prefix, output_directory)
    _draw_masks(stat, meanImg, iscell[:, 0] == 0, "rejected", out_prefix, output_directory)

def suite2p_roi_overlay(
        ops,
        stat,
        iscell,
        proj=None,
        plot_indices=None,
        output_directory=None,
        color_mode="random",
        red_border=False,
        colors=None,
):
    ops = load_ops(ops)
    yr0, yr1 = ops["yrange"]
    xr0, xr1 = ops["xrange"]
    img = ops[proj]  # Already cropped by suite2p

    print("yrange, xrange:", ops["yrange"], ops["xrange"])
    print("img.shape:", img.shape, "operational Ly/Lx:", ops["Ly"], ops["Lx"])

    # Normalize for display
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    norm_img = np.clip((img - p1) / (p99 - p1), 0, 1)

    H = np.zeros_like(norm_img)
    S = np.zeros_like(norm_img)
    mask = np.zeros_like(norm_img, dtype=bool)

    iscell = np.asarray(iscell)
    cell_mask = iscell if iscell.ndim == 1 else iscell[:, 0]
    indices = np.flatnonzero(cell_mask) if plot_indices is None else plot_indices
    for i, n in enumerate(indices):
        s = stat[n]
        # Shift ROI coordinates into cropped image space
        ypix = np.array(s["ypix"]) - yr0
        xpix = np.array(s["xpix"]) - xr0

        # Filter out invalid coords (can happen near edges)
        valid = (
                (ypix >= 0)
                & (ypix < norm_img.shape[0])
                & (xpix >= 0)
                & (xpix < norm_img.shape[1])
        )
        ypix = ypix[valid]
        xpix = xpix[valid]

        mask[ypix, xpix] = True

        if colors is not None:
            hue = rgb_to_hsv(np.array([[colors[i][:3]]]))[0, 0, 0]
        elif color_mode == "random":
            hue = np.random.rand()
        elif color_mode == "uniform":
            hue = 0.6
        else:
            hue = (i / max(len(indices), 1)) % 1.0
        H[ypix, xpix] = hue
        S[ypix, xpix] = 1

    rgb = hsv_to_rgb(np.stack([H, S, norm_img], axis=-1))

    if red_border and mask.any():
        borders = find_boundaries(mask, mode="outer")
        rgb[borders] = [1, 0, 0]

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    if output_directory:
        plt.savefig(output_directory, dpi=300, bbox_inches="tight", facecolor="black")
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

    mean_noise = np.mean(noise_levels)
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

    print(f"Neuron binning factor (default): {neuron_bin_size}")
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
