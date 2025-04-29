from pathlib import Path
import numpy as np
import math

import matplotlib.offsetbox
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import VPacker, HPacker, DrawingArea
from rastermap.utils import bin1d

from lbm_suite2p_python.utils import dff_percentile
from lbm_suite2p_python.utils import _resize_masks_fit_crop
from suite2p.detection.stats import ROI

def format_time(t):
    if t < 60:
        # make sure we dont show 0 seconds
        return f"{int(np.ceil(t))} s"
    elif t < 3600:
        return f"{int(round(t/60))} min"
    else:
        return f"{int(round(t/3600))} h"


def get_color_permutation(n):
    # choose a step from n//2+1 up to n-1 that is coprime with n
    for s in range(n//2 + 1, n):
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
    def __init__(self, size=1, label="", loc=2, ax=None, pad=0.4,
                 borderpad=0.5, ppad=0, sep=2, prop=None,
                 frameon=True, linekw=None, **kwargs):
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
        self.vpac = VPacker(children=[size_bar, txt],
                            align="center", pad=ppad, sep=sep)
        super().__init__(loc, pad=pad, borderpad=borderpad,
                         child=self.vpac, prop=prop, frameon=frameon, **kwargs)

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
    def __init__(self, height=1, label="", loc=2, ax=None, pad=0.4,
                 borderpad=0.5, ppad=0, sep=2, prop=None,
                 frameon=True, linekw={}, spacer_width=6, **kwargs):
        if ax is None:
            ax = plt.gca()
        trans = ax.transAxes

        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, 0], [0, height], **linekw)
        size_bar.add_artist(line)

        txt = matplotlib.offsetbox.TextArea(label, textprops=dict(rotation=90, ha="left", va="bottom"))
        self.txt = txt

        spacer = DrawingArea(spacer_width, 0, 0, 0)
        self.hpac = HPacker(children=[size_bar, spacer, txt],
                            align="bottom", pad=ppad, sep=sep)
        super().__init__(loc, pad=pad, borderpad=borderpad,
                         child=self.hpac, prop=prop, frameon=frameon, **kwargs)

def plot_traces(
        f,
        save_path="",
        fps=17.0,
        num_neurons=20,
        window=220,
        title="",
        offset=None,
        lw=0.5,
        cmap='tab10',
        signal_units="dff"
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
        Units of fluorescence signal. Options: "DF/F0 %", "DF/F0", "raw signal" (default: "DF/F0 %").
    """
    if isinstance(f, dict):
        ops = f
        print("Loading dff (%) from ops-dict")
        res = load_planar_results(ops)
        f = res["F"]
        f = dff_percentile(f) * 100
        signal_units = "dff"

    _, n_timepoints = f.shape
    data_time = np.arange(n_timepoints) / fps
    current_frame = min(int(window * fps), n_timepoints - 1)
    displayed_neurons = num_neurons

    if offset is None:
        p10 = np.percentile(f[:displayed_neurons, :current_frame + 1], 10, axis=1)
        p90 = np.percentile(f[:displayed_neurons, :current_frame + 1], 90, axis=1)
        offset = np.median(p90 - p10) * 1.2

    cmap_inst = plt.get_cmap(cmap)
    colors = cmap_inst(np.linspace(0, 1, displayed_neurons))
    perm = get_color_permutation(displayed_neurons)
    colors = colors[perm]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    ax.set_facecolor('black')
    ax.tick_params(axis='x', which='both', labelbottom=False, length=0, colors='white')
    ax.tick_params(axis='y', which='both', labelleft=False, length=0, colors='white')
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in reversed(range(displayed_neurons)):
        trace = f[i, :current_frame + 1]
        baseline = np.percentile(trace, 8)
        shifted_trace = (trace - baseline) + i * offset

        ax.plot(data_time[:current_frame + 1], shifted_trace, color=colors[i], lw=lw, zorder=-i)

        if i < displayed_neurons - 1:
            prev_trace = f[i + 1, :current_frame + 1]
            prev_baseline = np.percentile(prev_trace, 8)
            prev_shifted = (prev_trace - prev_baseline) + (i + 1) * offset
            mask = shifted_trace > prev_shifted
            ax.fill_between(data_time[:current_frame + 1], shifted_trace, prev_shifted,
                            where=mask, color='black', zorder=-i - 1)

    all_shifted = [(f[i, :current_frame + 1] - np.percentile(f[i, :current_frame + 1], 10)) + i * offset
                   for i in range(displayed_neurons)]
    all_y = np.concatenate(all_shifted)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_range = window
    new_x_upper = window + 0.05 * x_range

    time_bar_length = 0.1 * window
    if time_bar_length < 60:
        time_label = f"{time_bar_length:.0f} s"
    elif time_bar_length < 3600:
        time_label = f"{time_bar_length / 60:.0f} min"
    else:
        time_label = f"{time_bar_length / 3600:.1f} hr"

    linekw = dict(color="white", linewidth=3)
    hsb = AnchoredHScaleBar(size=0.1, label=time_label,
                                loc=4, frameon=False, pad=0.6, sep=4, linekw=linekw, ax=ax)
    hsb.set_bbox_to_anchor((0.9, -0.05), transform=ax.transAxes)
    hsb.txt._text.set_color('white')

    ax.add_artist(hsb)

    dff_bar_height = 0.1 * (y_max - y_min)
    bottom_baseline = np.percentile(f[0, :current_frame + 1], 8)
    bottom_trace_min = np.min(f[0, :current_frame + 1] - bottom_baseline)
    rounded_dff = round(dff_bar_height / 5) * 5

    dff_label = f"{rounded_dff:.0f} % ΔF/F₀" if signal_units == "dff" else f"{rounded_dff:.0f} raw signal (a.u)"

    vsb = AnchoredVScaleBar(
        height=.1,
        label=dff_label,
        loc='lower right',
        frameon=False,
        pad=-.1,
        sep=4,
        linekw=linekw,
        ax=ax,
        spacer_width=0
    )
    vsb.set_bbox_to_anchor((1.00, 0.05), transform=ax.transAxes)
    # vsb.set_bbox_to_anchor(, transform=ax.transAxes)
    vsb.txt._text.set_color('white')
    ax.add_artist(vsb)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", color="white")

    ax.set_ylabel(f"Neuron Count: {displayed_neurons}", fontsize=8, fontweight="bold", color="white", labelpad=2)

    if save_path:
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()


def animate_traces(
        f,
        save_path="./scrolling.mp4",
        fps=17.0,
        start_neurons=20,
        window=120,
        title="",
        gap=None,
        lw=0.5,
        cmap='tab10',
        anim_fps=60,
        expand_after=5,
        speed_factor=1.0,
        expansion_factor=2.0,
        smooth_factor=1,
):
    n_neurons, n_timepoints = f.shape
    data_time = np.arange(n_timepoints) / fps
    T_data = data_time[-1]
    current_frame = min(int(window * fps), n_timepoints - 1)
    t_f_local = (T_data - window + expansion_factor * expand_after) / (1 + expansion_factor)

    if gap is None:
        p10 = np.percentile(f[:start_neurons, :current_frame+1], 10, axis=1)
        p90 = np.percentile(f[:start_neurons, :current_frame+1], 90, axis=1)
        gap = np.median(p90 - p10) * 1.2

    cmap_inst = plt.get_cmap(cmap)
    colors = cmap_inst(np.linspace(0, 1, n_neurons))
    perm = np.random.permutation(n_neurons)
    colors = colors[perm]

    all_shifted = []
    for i in range(start_neurons):
        trace = f[i, :current_frame+1]
        baseline = np.percentile(trace, 8)
        shifted = (trace - baseline) + i * gap
        all_shifted.append(shifted)

    all_y = np.concatenate(all_shifted)
    y_min = np.min(all_y)
    y_max = np.max(all_y)

    rounded_dff = np.round(y_max - y_min) * 0.1
    dff_label = f"{rounded_dff:.0f} % ΔF/F₀"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    ax.set_facecolor('black')
    ax.tick_params(axis='x', labelbottom=False, length=0)
    ax.tick_params(axis='y', labelleft=False, length=0)

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
        ax=ax
    )

    hsb.set_bbox_to_anchor((0.97, -0.1), transform=ax.transAxes)

    ax.add_artist(hsb)

    vsb = AnchoredVScaleBar(
        height=.1,
        label=dff_label,
        loc='lower right',
        frameon=False,
        pad=0,
        sep=4,
        linekw=linekw,
        ax=ax,
        spacer_width=0
    )
    ax.add_artist(vsb)

    lines = []
    for i in range(n_neurons):
        line, = ax.plot([], [], color=colors[i], lw=lw, zorder=-i)
        lines.append(line)

    def init():
        for i in range(n_neurons):
            if i < start_neurons:
                trace = f[i, :current_frame+1]
                baseline = np.percentile(trace, 8)
                shifted = (trace - baseline) + i * gap
                lines[i].set_data(data_time[:current_frame+1], shifted)
            else:
                lines[i].set_data([], [])
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

        for i in range(n_neurons):
            if i < n_visible:
                trace = f[i, i_lower:i_upper]
                baseline = np.percentile(trace, 8)
                shifted = (trace - baseline) + i * gap
                lines[i].set_data(data_time[i_lower:i_upper], shifted)
            else:
                lines[i].set_data([], [])

        for fill in fills:
            fill.remove()
        fills.clear()

        for i in range(n_visible - 1):
            trace1 = f[i, i_lower:i_upper]
            baseline1 = np.percentile(trace1, 8)
            shifted1 = (trace1 - baseline1) + i * gap

            trace2 = f[i+1, i_lower:i_upper]
            baseline2 = np.percentile(trace2, 8)
            shifted2 = (trace2 - baseline2) + (i+1) * gap

            fill = ax.fill_between(data_time[i_lower:i_upper], shifted1, shifted2,
                                   where=shifted1 > shifted2, color='black', zorder=-i-1)
            fills.append(fill)

        all_shifted = [(f[i, i_lower:i_upper] - np.percentile(f[i, i_lower:i_upper], 8)) + i * gap for i in range(n_visible)]
        all_y = np.concatenate(all_shifted)
        y_min_new, y_max_new = np.min(all_y), np.max(all_y)

        extra_axis = 0.05 * (x_max - x_min)
        ax.set_xlim(x_min, x_max + extra_axis)
        ax.set_ylim(y_min_new - 0.05 * abs(y_min_new), y_max_new + 0.05 * abs(y_max_new))

        if title:
            ax.set_title(title, fontsize=16, fontweight="bold", color="white")

        rounded_dff = np.round(y_max_new - y_min_new) * 0.1

        if rounded_dff > 300:
            vsb.set_visible(False)
        else:
            dff_label = f"{rounded_dff:.0f} % ΔF/F₀"
            vsb.txt.set_text(dff_label)
        hsb.txt.set_text(format_time(0.1 * (x_max - x_min)))
        ax.set_ylabel(f"Neuron Count: {n_visible}", fontsize=8, fontweight="bold", labelpad=2)

        return lines + [hsb, vsb] + fills

    effective_anim_fps = anim_fps * smooth_factor
    total_frames = int(np.ceil((T_data / speed_factor)))

    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, interval=1000/effective_anim_fps, blit=True)
    ani.save(save_path, fps=anim_fps)
    plt.show()


def plot_projection(
        ops,
        savepath=None,
        fig_label=None,
        vmin=None,
        vmax=None,
        add_scalebar=False,
        proj="meanImg",
        display_masks=False,
        accepted_only=False
):
    if proj == "meanImg":
        txt = "Mean-Image"
    elif proj == "max_proj":
        txt = "Max-Projection"
    elif proj == "meanImgE":
        txt = "Mean-Image (Enhanced)"
    else:
        raise ValueError("Unknown projection type. Options are ['meanImg', 'max_proj', 'meanImgE']")

    if savepath:
        savepath = Path(savepath)

    data = ops[proj]
    shape = data.shape
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    vmin = np.nanpercentile(data, 2) if vmin is None else vmin
    vmax = np.nanpercentile(data, 98) if vmax is None else vmax

    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)

    # move projection title higher if masks are displayed to avoid overlap.
    proj_title_y = 1.07 if display_masks else 1.02
    ax.text(0.5, proj_title_y, txt, transform=ax.transAxes,
            fontsize=14, fontweight='bold', fontname="Courier New",
            color='white', ha='center', va='bottom')
    if fig_label:
        fig_label = fig_label.replace("_", " ").replace("-", " ").replace(".", " ")
        ax.set_ylabel(fig_label, color='white', fontweight='bold', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    if display_masks:
        res = load_planar_results(ops)
        stat = res["stat"]
        iscell = res["iscell"]
        im = ROI.stats_dicts_to_3d_array(stat, Ly=ops['Ly'], Lx=ops['Lx'], label_id=True)
        im[im == 0] = np.nan
        accepted_cells = np.sum(iscell)
        rejected_cells = np.sum(~iscell)
        cell_rois = _resize_masks_fit_crop(
            np.nanmax(im[iscell], axis=0) if np.any(iscell) else np.zeros_like(im[0]),
            shape
        )
        green_overlay = np.zeros((*shape, 4), dtype=np.float32)
        green_overlay[..., 1] = 1
        green_overlay[..., 3] = (cell_rois > 0) * 1.0
        ax.imshow(green_overlay)
        if not accepted_only:
            non_cell_rois = _resize_masks_fit_crop(
                np.nanmax(im[~iscell], axis=0) if np.any(~iscell) else np.zeros_like(im[0]),
                shape)
            magenta_overlay = np.zeros((*shape, 4), dtype=np.float32)
            magenta_overlay[..., 0] = 1
            magenta_overlay[..., 2] = 1
            magenta_overlay[..., 3] = (non_cell_rois > 0) * 0.5
            ax.imshow(magenta_overlay)
        ax.text(0.37, 1.02, f"Accepted: {accepted_cells:03d}", transform=ax.transAxes,
                fontsize=14, fontweight='bold', fontname="Courier New",
                color='lime', ha='right', va='bottom')
        ax.text(0.63, 1.02, f"Rejected: {rejected_cells:03d}", transform=ax.transAxes,
                fontsize=14, fontweight='bold', fontname="Courier New",
                color='magenta', ha='left', va='bottom')
    if add_scalebar and 'dx' in ops:
        pixel_size = ops['dx']
        scale_bar_length = 100 / pixel_size
        scalebar_x = shape[1] * 0.05
        scalebar_y = shape[0] * 0.90
        ax.add_patch(Rectangle(
            (scalebar_x, scalebar_y), scale_bar_length, 5,
            edgecolor='white', facecolor='white'))
        ax.text(scalebar_x + scale_bar_length / 2, scalebar_y - 10,
                "100 μm", color='white', fontsize=10, ha='center', fontweight='bold')

    # remove the spines that will show up as white bars
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, facecolor='black')
        plt.close(fig)
    else:
        plt.show()


def plot_noise_distribution(noise_levels: np.ndarray, save_path=None, title="Noise Level Distribution"):
    """
    Plots and saves the distribution of noise levels across neurons as a standardized image.

    Parameters
    ----------
    noise_levels : np.ndarray
        1D array of noise levels for each neuron.
    save_path : str or Path, optional
        Path to save the plot. If empty, the plot will be displayed instead of saved.
    title : str, optional
        Suptitle for plot, default is "Noise Level Distribution".

    See Also
    --------
    lbm_suite2p_python.dff_shot_noise
    """
    if save_path:
        save_path = Path(save_path)
        if save_path.is_dir():
            raise AttributeError(f"save_path should be a fully qualified file path, not a directory: {save_path}")

    fig = plt.figure(figsize=(8, 5))
    plt.hist(noise_levels, bins=50, color="gray", alpha=0.7, edgecolor="black")

    mean_noise = np.mean(noise_levels)
    plt.axvline(mean_noise, color='r', linestyle='dashed', linewidth=2, label=f"Mean: {mean_noise:.2f}")

    plt.xlabel("Noise Level", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Neurons", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def load_planar_results(ops, z_plane=None) -> dict:
    """
    Load stat, iscell, spks files and return as a dict. Does NOT filter by valid cells, array contain both
    accepted and rejected neurons. Filter for accepted-only via f[iscell] or fneue[iscell] if needed.

    Parameters
    ----------
    ops : dict, str or Path
        Dict of or path to the ops.npy file.
    z_plane : int or None, optional
        the z-plane index for this file. If provided, it is stored in the output.

    Returns
    -------
    dict
        dictionary with keys:
        - 'F': fluorescence traces loaded from F.npy,
        - 'Fneu': neuropil fluorescence traces loaded from Fneu.npy,
        - 'spks': spike traces loaded from spks.npy,
        - 'stat': stats loaded from stat.npy,
        - 'iscell': boolean array from iscell.npy,
        - 'cellprob': cell probability from classifier.
        - 'z_plane': an array (of shape [n_neurons,]) with the provided z_plane index.

    See Also
    --------
    lbm_suite2p_python.load_ops
    lbm_suite2p_python.load_traces
    """
    output_ops = load_ops(ops)
    save_path = Path(output_ops['save_path'])

    F = np.load(save_path.joinpath('F.npy'))
    Fneu = np.load(save_path.joinpath('Fneu.npy'))
    spks = np.load(save_path.joinpath('spks.npy'))
    stat = np.load(save_path.joinpath('stat.npy'), allow_pickle=True)
    iscell = np.load(save_path.joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
    cellprob = np.load(save_path.joinpath('iscell.npy'), allow_pickle=True)[:, 1]

    n_neurons = spks.shape[0]
    if z_plane is None:
        # If not provided, assign a default of 0
        z_plane_arr = np.zeros(n_neurons, dtype=int)
    else:
        z_plane_arr = np.full(n_neurons, z_plane, dtype=int)
    return {"F": F, "Fneu": Fneu, "spks": spks, "stat": stat, "iscell": iscell, "cellprob": cellprob, 'z_plane': z_plane_arr}

def load_traces(ops):
    """
    Return (accepted-only) fluorescence traces, neuropil traces and spike traces from ops file.

    Parameters
    ----------
    ops : str, Path or dict
        Path to the ops.npy file or a dict containing the ops data.

    Returns
    -------
    tuple
        A tuple containing three arrays **filtered to contain only accepted neurons**:
        - F: Fluorescence traces (2D array, shape [n_neurons, n_timepoints])
        - Fneu: Neuropil fluorescence traces (2D array, shape [n_neurons, n_timepoints])
        - spks: Spike traces (2D array, shape [n_neurons, n_timepoints])

    See Also
    --------
    lbm_suite2p_python.load_ops
    lbm_suite2p_python.load_planar_results
    """
    output_ops = load_ops(ops)
    save_path = Path(output_ops['save_path'])

    F = np.load(save_path.joinpath('F.npy'))
    Fneu = np.load(save_path.joinpath('Fneu.npy'))
    spks = np.load(save_path.joinpath('spks.npy'))
    iscell = np.load(save_path.joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
    return F[iscell], Fneu[iscell], spks[iscell]


def load_ops(ops_input: str | Path | list[str | Path]):
    """ Simple utility load a suite2p npy file"""
    if isinstance(ops_input, (str, Path)):
        return np.load(ops_input, allow_pickle=True).item()
    elif isinstance(ops_input, dict):
        return ops_input
    print("Warning: No valid ops file provided, returning None.")
    return None


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
        title_kwargs={},
        fig_text=None
):
    n_neurons, n_timepoints = spks.shape

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
    ax.tick_params(axis='both', labelbottom=False, labelleft=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    heatmap_pos = ax.get_position()

    scalebar_length = heatmap_pos.width * 0.1         # 10% width of heatmap
    scalebar_duration = np.round(current_time * 0.1)  # 10% of the displayed time in heatmap

    x_start = heatmap_pos.x1 - scalebar_length
    x_end = heatmap_pos.x1
    y_position = heatmap_pos.y0

    fig.lines.append(plt.Line2D([x_start, x_end], [y_position - 0.03, y_position - 0.03],
                                transform=fig.transFigure, color='white', linewidth=2, solid_capstyle='butt'))

    fig.text(
        x=(x_start + x_end) / 2,
        y=y_position - 0.045,  # slightly below the scalebar
        s=f"{scalebar_duration:.0f} s",
        ha="center", va="top",
        color="white", fontsize=6
    )

    axins = fig.add_axes([
        heatmap_pos.x0,                  # exactly aligned with heatmap's left edge
        heatmap_pos.y0 - 0.03,           # slightly below the heatmap
        heatmap_pos.width * 0.1,         # 20% width of heatmap
        0.015                            # height of the colorbar
    ])

    cbar = fig.colorbar(img, cax=axins, orientation="horizontal", ticks=[vmin, vmax])
    cbar.ax.tick_params(labelsize=5, colors="white", pad=2)
    cbar.outline.set_edgecolor("white")

    fig.text(
        heatmap_pos.x0,
        heatmap_pos.y0 - 0.1,  # below the colorbar with spacing
        "z-scored",
        ha="left", va="top",
        color="white", fontsize=6
    )

    scalebar_neurons = int(0.1 * current_neurons)

    x_position = heatmap_pos.x1 + 0.01  # slightly right of heatmap
    y_start = heatmap_pos.y0
    y_end = y_start + (heatmap_pos.height * scalebar_neurons / current_neurons)

    line = plt.Line2D(
        [x_position, x_position], [y_start, y_end],
        transform=fig.transFigure, color='white', linewidth=2
    )
    line.set_figure(fig)
    fig.lines.append(line)

    ntype = "neurons" if scalebar_neurons == 1 else "neurons"
    fig.text(
        x=x_position + 0.008,
        y=y_start,
        s=f"{scalebar_neurons} {ntype}",
        ha="left", va="bottom",
        color="white", fontsize=6, rotation=90
    )

    if fig_text is None:
        fig_text = f"Neurons: {spks.shape[0]}, Superneurons: {sn.shape[0]}, n_clusters: {model.n_PCs}, n_PCs: {model.n_clusters}, locality: {model.locality}"

    fig.text(
        x=(heatmap_pos.x0 + heatmap_pos.x1) / 2,
        y=y_start - 0.085,  # vertically between existing scalebars
        s=fig_text,
        ha="center", va="top",
        color="white", fontsize=6
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
