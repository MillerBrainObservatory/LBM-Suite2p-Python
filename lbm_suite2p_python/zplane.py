import math

import matplotlib.pyplot as plt
import matplotlib.offsetbox
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.offsetbox import VPacker, HPacker, DrawingArea
import numpy as np

from lbm_suite2p_python import load_traces, dff_percentile


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
        print("Loading dff (%) from ops-dict")
        f, _, _ = load_traces(f)
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
    ax.add_artist(hsb)

    dff_bar_height = 0.1 * (y_max - y_min)
    bottom_baseline = np.percentile(f[0, :current_frame + 1], 8)
    bottom_trace_min = np.min(f[0, :current_frame + 1] - bottom_baseline)
    rounded_dff = round(dff_bar_height / 5) * 5

    dff_label = f"{rounded_dff:.0f} % ΔF/F₀" if signal_units == "dff" else f"{rounded_dff:.0f} raw signal (a.u)"

    vsb = AnchoredVScaleBar(height=0.1, label=dff_label,
                                loc='lower left', frameon=False, pad=0, sep=4,
                                linekw=linekw, ax=ax, spacer_width=0)
    hsb.txt._text.set_color('white')
    vsb.set_bbox_to_anchor((new_x_upper - x_range * 0.05, bottom_trace_min), transform=ax.transData)
    vsb.txt._text.set_color('white')
    ax.add_artist(vsb)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", color="white")

    ax.set_ylabel(f"Neuron Count: {displayed_neurons}", fontsize=8, fontweight="bold", color="white", labelpad=2)

    if save_path:
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
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



def plot_noise_distribution(noise_levels, save_path, plane_idx, title="Noise Level Distribution"):
    """
    Plots and saves the distribution of noise levels across neurons as a standardized image.

    Parameters:
    - noise_levels (numpy.ndarray): Noise levels for each neuron.
    - save_path (Path): Directory where images will be saved.
    - plane_idx (int): Index of the imaging plane.
    - title (str): Title of the plot.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(noise_levels, bins=50, color="gray", alpha=0.7, edgecolor="black")

    mean_noise = np.mean(noise_levels)
    plt.axvline(mean_noise, color='r', linestyle='dashed', linewidth=2, label=f"Mean: {mean_noise:.2f}")

    plt.xlabel("Noise Level", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Neurons", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(save_path / f"plane_{plane_idx}.png", dpi=200, bbox_inches="tight")
    plt.close()

