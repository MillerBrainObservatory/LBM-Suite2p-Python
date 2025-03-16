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
                 frameon=True, linekw={}, **kwargs):
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
    f, save_path="./stacked_traces.png", fps=17.0, start_neurons=20, window=120, offset=None, lw=0.1, cmap='tab10'
):
    """
    plot stacked fluorescence traces with automatic offset and scale bars.

    Parameters
    ----------
    f : ndarray
        2d array of fluorescence traces (n_neurons x n_timepoints).
    save_path : str, optional
        path to save the output plot (default is "./stacked_traces.png").
    fps : float, optional
        sampling rate in frames per second (default is 17.0).
    start_neurons : int, optional
        number of neurons to display (default is 20).
    window : float, optional
        time window (in seconds) to display (default is 120).
    offset : float or none, optional
        vertical offset between traces; if none, computed automatically.
    lw : float, optional
        line width for data points

    """
    if isinstance(f, dict):
        f = np.load(Path(f["ops_path"]).parent.joinpath("F.npy"))

    _, n_timepoints = f.shape

    data_time = np.arange(n_timepoints) / fps
    x_lower = 0
    x_upper = window
    current_frame = min(int(window * fps), n_timepoints - 1)
    displayed_neurons = start_neurons

    if offset is None:
        perc10 = np.percentile(f[:displayed_neurons, :current_frame+1], 10, axis=1)
        perc90 = np.percentile(f[:displayed_neurons, :current_frame+1], 90, axis=1)
        gap = np.median(perc90 - perc10) * 1.2
        offset = gap

    # randomize colors
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, displayed_neurons))
    perm = get_color_permutation(displayed_neurons)
    colors = colors[perm]
    # np.random.shuffle(colors)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    for i in range(displayed_neurons):
        trace = f[i, :current_frame+1]
        baseline = np.percentile(trace, 8)
        shifted_trace = (trace - baseline) + i * offset
        ax.plot(data_time[:current_frame+1], shifted_trace, color=colors[i], lw=lw)
    all_shifted = [(f[i, :current_frame+1] - np.percentile(f[i, :current_frame+1], 10)) + i * offset
                   for i in range(displayed_neurons)]

    all_y = np.concatenate(all_shifted)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_range = x_upper - x_lower

    # whitespace on right
    extra = 0.05 * x_range
    new_x_upper = x_upper + extra
    ax.set_xlim(x_lower, new_x_upper)
    ax.set_ylim(y_min, y_max)

    yticks = np.linspace(y_min, y_max, 10)
    ax.set_yticks(yticks)

    # show x-axis labels and hide y-axis labels
    ax.tick_params(axis='x', which='both', labelbottom=False, length=0)
    ax.tick_params(axis='y', which='both', labelleft=False, length=0)

    ax.set_axisbelow(True)
    # ax.grid(True, color='gray', linewidth=0.2, linestyle='--')

    for spine in ax.spines.values():
        spine.set_visible(False)

    linekw = dict(color="white", linewidth=3)
    time_bar_length = min(10, 0.1 * (x_upper - x_lower))

    # time scalebar
    hsb = AnchoredHScaleBar(size=time_bar_length,
                            label=f"{time_bar_length:.0f} s",
                            loc=4, frameon=False, pad=0.6, sep=4,
                            linekw=linekw, ax=ax)


    hsb.set_bbox_to_anchor((new_x_upper - (x_upper - x_lower)*0.1, y_min - (y_max - y_min)*0.09), transform=ax.transData)
    ax.add_artist(hsb)

    dff_bar_height = 0.05 * (y_max - y_min)
    y_bar = y_min + 0.05 * (y_max - y_min)

    # dff scalebar; tweak spacer_width to adjust gap between bar and text
    rounded_dff = round(dff_bar_height / 5) * 5
    vsb = AnchoredVScaleBar(height=dff_bar_height,
                            label=f"ΔF/F₀ = {rounded_dff:.0f}%",
                            loc='lower left', frameon=False, pad=0, sep=4,
                            linekw=linekw, ax=ax, spacer_width=0)
    vsb.set_bbox_to_anchor((new_x_upper - (x_upper - x_lower)*0.05, y_bar), transform=ax.transData)
    ax.add_artist(vsb)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, facecolor='black', bbox_inches='tight')
    plt.close(fig)


def plot_traces_anch(f, save_path="./stacked_traces.png", fps=17.0,
                     start_neurons=20, window=120, offset=None, final_state=True):
    """
    Plot stacked fluorescence traces with automatic offset and scale bars.

    Parameters
    ----------
    f : ndarray
        2d array of fluorescence traces (n_neurons x n_timepoints).
    save_path : str, optional
        path to save the output plot (default is "./stacked_traces.png").
    fps : float, optional
        sampling rate in frames per second (default is 17.0).
    start_neurons : int, optional
        number of neurons to display when final_state is False (default is 20).
    window : float, optional
        time window (in seconds) to display (default is 120).
    offset : float or None, optional
        vertical offset between traces; if None, computed automatically.
    final_state : bool, optional
        if True, display final view; if False, display initial view.

    Returns
    -------
    None

    Notes
    -----
    tweak 'window' to change time display, 'start_neurons' for neuron count, and
    'offset' to manually set vertical spacing.
    adjust scale bar parameters (pad, spacer_width, etc.) to change bar appearance.
    """
    n_neurons, n_timepoints = f.shape
    T_data = n_timepoints / fps
    data_time = np.arange(n_timepoints) / fps
    if final_state:
        x_lower = T_data - window
        x_upper = T_data
        current_frame = n_timepoints - 1
        displayed_neurons = n_neurons
    else:
        x_lower = 0
        x_upper = window
        current_frame = min(int(window * fps), n_timepoints - 1)
        displayed_neurons = start_neurons
    if offset is None:
        perc10 = np.percentile(f[:displayed_neurons, :current_frame+1], 10, axis=1)
        perc90 = np.percentile(f[:displayed_neurons, :current_frame+1], 90, axis=1)
        gap = np.median(perc90 - perc10) * 1.2
        offset = gap

    # randomize colors; tweak colormap
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, displayed_neurons))
    np.random.shuffle(colors)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    for i in range(displayed_neurons):
        trace = f[i, :current_frame+1]
        baseline = np.percentile(trace, 8)
        shifted_trace = (trace - baseline) + i * offset
        ax.plot(data_time[:current_frame+1], shifted_trace, color=colors[i], lw=1)

    # shift values in Y by an offset to raise  on the graph
    all_shifted = [(f[i, :current_frame+1] - np.percentile(f[i, :current_frame+1], 10)) + i * offset
                   for i in range(displayed_neurons)]
    all_y = np.concatenate(all_shifted)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_range = x_upper - x_lower

    # extra whitespace on right
    extra = 0.05 * x_range
    new_x_upper = x_upper + extra
    ax.set_xlim(x_lower, new_x_upper)
    ax.set_ylim(y_min, y_max)
    xticks = np.linspace(x_lower, new_x_upper, 10)
    yticks = np.linspace(y_min, y_max, 10)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # keep the axis but hide it for an optional grid background
    ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, length=0)
    ax.set_axisbelow(True)
    ax.grid(True, color='gray', linewidth=0.1, linestyle='--')
    for spine in ax.spines.values():
        spine.set_visible(False)

    linekw = dict(color="white", linewidth=3)
    time_bar_length = min(10, 0.1 * (x_upper - x_lower))

    # time scalebar
    hsb = AnchoredHScaleBar(size=time_bar_length,
                            label=f"{time_bar_length:.0f} s",
                            loc=4, frameon=False, pad=0.6, sep=4,
                            linekw=linekw, ax=ax)

    # (y_max-ymin)*N, change N to move scale further from graph in Y, same for (x_upper-x_lower)*N
    hsb.set_bbox_to_anchor((new_x_upper - (x_upper - x_lower)*0.1,
                            y_min - (y_max - y_min)*0.09), transform=ax.transData)
    ax.add_artist(hsb)

    dff_bar_height = 0.1 * (y_max - y_min)
    y_bar = y_min + 0.05 * (y_max - y_min)

    # dff scalebar; spacer_width controls gap between bar and text
    vsb = AnchoredVScaleBar(height=dff_bar_height,
                            label=f"{dff_bar_height:.1f} % dF/F",
                            loc='lower left', frameon=False, pad=0, sep=4,
                            linekw=linekw, ax=ax, spacer_width=0)
    vsb.set_bbox_to_anchor((new_x_upper - (x_upper - x_lower)*0.05, y_bar), transform=ax.transData)
    ax.add_artist(vsb)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, facecolor='black', bbox_inches='tight')
    plt.close(fig)
