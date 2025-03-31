import math

import matplotlib.pyplot as plt
import matplotlib.offsetbox
from matplotlib.lines import Line2D
from matplotlib.offsetbox import VPacker, HPacker, DrawingArea
from pathlib import Path
import numpy as np


def format_time(t):
    if t < 60:
        return f"{int(round(t))} s"
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

    notes
    -----
    tweak 'size' for bar length; this value remains fixed.
    """
    def __init__(self, size=1, label="", loc=2, ax=None, pad=0.4,
                 borderpad=0.5, ppad=0, sep=2, prop=None,
                 frameon=True, linekw=None, **kwargs):
        if linekw is None:
            linekw = {}
        if ax is None:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
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
    create an anchored vertical scale bar.

    parameters
    ----------
    height : float, optional
        bar height in data units (default is 1).
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
    spacer_width : float, optional
        width of spacer between bar and text.

    notes
    -----
    tweak 'height' for bar length.
    """
    def __init__(self, height=1, label="", loc=2, ax=None, pad=0.4,
                 borderpad=0.5, ppad=0, sep=2, prop=None,
                 frameon=True, linekw={}, spacer_width=6, **kwargs):
        if ax is None:
            ax = plt.gca()
        trans = ax.get_yaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, 0], [0, height], **linekw)
        size_bar.add_artist(line)
        txt = matplotlib.offsetbox.TextArea(label, textprops=dict(rotation=90))
        self.txt = txt
        spacer = DrawingArea(spacer_width, 0, 0, 0)
        self.hpac = HPacker(children=[size_bar, spacer, txt],
                            align="center", pad=ppad, sep=sep)
        super().__init__(loc, pad=pad, borderpad=borderpad,
                         child=self.hpac, prop=prop, frameon=frameon, **kwargs)


def plot_traces(
    f,
    save_path="",
    fps=17.0,
    start_neurons=20,
    window=120,
    title="",
    offset=None,
    lw=0.5,
    cmap='tab10',
    signal_units="dff"  # New parameter
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
    start_neurons : int, optional
        Number of neurons to display (default is 20).
    window : float, optional
        Time window (in seconds) to display (default is 120).
    offset : float or None, optional
        Vertical offset between traces; if None, computed automatically.
    lw : float, optional
        Line width for data points.
    signal_units : str, optional
        Units of fluorescence signal. Options: "DF/F0 %", "DF/F0", "raw signal" (default: "DF/F0 %").

    """
    if isinstance(f, dict):
        f = np.load(Path(f["ops_path"]).parent.joinpath("F.npy"))

    _, n_timepoints = f.shape
    data_time = np.arange(n_timepoints) / fps  # x-axis in seconds
    x_lower, x_upper = 0, window
    current_frame = min(int(window * fps), n_timepoints - 1)
    displayed_neurons = start_neurons

    if offset is None:
        perc10 = np.percentile(f[:displayed_neurons, :current_frame+1], 10, axis=1)
        perc90 = np.percentile(f[:displayed_neurons, :current_frame+1], 90, axis=1)
        gap = np.median(perc90 - perc10) * 1.2
        offset = gap

    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, displayed_neurons))
    perm = get_color_permutation(displayed_neurons)
    colors = colors[perm]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    ax.tick_params(axis='x', which='both', labelbottom=False, length=0)
    ax.tick_params(axis='y', which='both', labelleft=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in reversed(range(displayed_neurons)):
        trace = f[i, :current_frame + 1]
        trace[trace < 0] = 0
        baseline = np.percentile(trace, 8)
        shifted_trace = (trace - baseline) + i * offset

        ax.plot(data_time[:current_frame + 1], shifted_trace, color=colors[i], lw=lw, zorder=-i)

        # mask the above trace where the underneath trace extends, highlights the largest signals
        if i < displayed_neurons - 1:
            prev_trace = f[i + 1, :current_frame + 1]
            prev_baseline = np.percentile(prev_trace, 8)
            prev_shifted = (prev_trace - prev_baseline) + (i + 1) * offset
            mask = shifted_trace > prev_shifted
            ax.fill_between(data_time[:current_frame + 1], shifted_trace, prev_shifted,
                            where=mask, color='black', zorder=-i - 1)

    all_shifted = [(f[i, :current_frame+1] - np.percentile(f[i, :current_frame+1], 10)) + i * offset
                   for i in range(displayed_neurons)]
    all_y = np.concatenate(all_shifted)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_range = x_upper - x_lower
    extra = 0.05 * x_range
    new_x_upper = x_upper + extra

    # time scalbar
    time_bar_length = 0.1 * (x_upper - x_lower)
    time_label = (f"{time_bar_length:.0f} s" if time_bar_length < 60 else
                  f"{time_bar_length / 60:.0f} min" if time_bar_length < 3600 else
                  f"{time_bar_length / 3600:.1f} hr")

    linekw = dict(color="white", linewidth=3)
    hsb = AnchoredHScaleBar(size=time_bar_length, label=time_label,
                            loc=4, frameon=False, pad=0.6, sep=4, linekw=linekw, ax=ax)
    hsb.set_bbox_to_anchor((new_x_upper - (x_upper - x_lower)*0.1, y_min - (y_max - y_min)*0.09),
                            transform=ax.transData)
    ax.add_artist(hsb)

    # bounds
    dff_bar_height = 0.1 * (y_max - y_min)
    num_traces_in_bar = 0.1 * displayed_neurons
    print(num_traces_in_bar)

    bottom_baseline = np.percentile(f[0, :current_frame+1], 8)
    bottom_trace_min = np.min(f[0, :current_frame+1] - bottom_baseline)
    y_bar = bottom_trace_min #- 0.05 * (y_max - y_min)

    rounded_dff = round(dff_bar_height / 5) * 5

    # Adjust scalebar label based on signal_units
    if signal_units == "dff":
        dff_label = f"{rounded_dff:.0f} % ΔF/F₀"
    else:
        dff_label = f"{rounded_dff:.0f} raw signal (a.u)"

    vsb = AnchoredVScaleBar(height=dff_bar_height, label=dff_label,
                            loc='lower left', frameon=False, pad=0, sep=4,
                            linekw=linekw, ax=ax, spacer_width=0)

    vsb.set_bbox_to_anchor((new_x_upper - (x_upper - x_lower)*0.05, y_bar), transform=ax.transData)
    ax.add_artist(vsb)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", color="white")

    plt.ylabel(f"Neuron Count: {displayed_neurons}", fontsize=8, fontweight="bold", labelpad=2)
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

    plt.close(fig)


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

