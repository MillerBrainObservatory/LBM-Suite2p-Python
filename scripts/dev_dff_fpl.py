import mbo_utilities as mbo  # noqa
import lbm_suite2p_python as lsp
import numpy as np
from scipy.stats import skew
from pathlib import Path
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from skimage.segmentation import find_boundaries
import tifffile
import fastplotlib as fpl


def pick_unique_cells(peaks, stds, skews, noise, num_each=4):
    selected = set()
    cells = {"High Peak": [], "High Std": [], "High Skew": [], "Low Noise": []}

    def pick_top(metric, label, reverse=True):
        indices = np.argsort(metric)[::-1 if reverse else 1]
        picked = []
        for ix in indices:
            if ix not in selected:
                picked.append(ix)
                selected.add(ix)
                if len(picked) == num_each:
                    break
        cells[label] = picked

    pick_top(peaks, "High Peak")
    pick_top(stds, "High Std")
    pick_top(skews, "High Skew")
    pick_top(noise, "Low Noise", reverse=False)
    return cells


def dff_methods(
        F,
        indices,
        fps=17
):
    method_info = []
    dffs = {}
    f0s = {i: [] for i in indices}

    for i in indices:
        _trace = F[i, :]
        baselines = {
            ("Static", "median", None): np.full_like(_trace, np.median(_trace)),
            ("Moving", 8, 60): generic_filter(_trace, lambda x: np.percentile(x, 8), size=int(60 * fps),
                                              mode='nearest'),
            ("Moving", 8, 30): generic_filter(_trace, lambda x: np.percentile(x, 8), size=int(30 * fps),
                                              mode='nearest'),
            ("Moving", 40, 10): generic_filter(_trace, lambda x: np.percentile(x, 40), size=int(10 * fps),
                                               mode='nearest'),
            ("Moving", 40, 60): generic_filter(_trace, lambda x: np.percentile(x, 40), size=int(60 * fps),
                                               mode='nearest'),
        }

        dffs[i] = []
        f0s[i] = []
        for key, f0 in baselines.items():
            dff = (_trace - f0) / f0
            dffs[i].append(dff)
            f0s[i].append(f0)
            if i == indices[0]:
                method_info.append(key)  # only once

    return dffs, method_info, f0s

def plot_dff_event_counts(
        dffs,
        method_info, threshold=3, savename="dff_event_counts.png", cell_indices=None):
    counts = {label: [] for label in method_info}

    selected_keys = list(dffs.keys())
    if cell_indices is not None:
        selected_keys = [k for k in selected_keys if k in cell_indices]

    for key in selected_keys:
        _dff_list = dffs[key]
        for label, dff in zip(method_info, _dff_list):
            baseline_mask = dff < np.percentile(dff, 30)
            noise = np.nanstd(dff[baseline_mask])
            crossings = np.diff((dff > threshold * noise).astype(int)) == 1
            counts[label].append(np.sum(crossings))

    count_matrix = np.array([counts[label] for label in method_info])

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='black')
    im = ax.imshow(count_matrix, aspect='auto', cmap='viridis')

    num_cells = len(selected_keys)
    xticks = [0, num_cells // 3, 2 * num_cells // 3, num_cells - 1]
    xticklabels = [selected_keys[i] for i in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, color='white')
    ax.set_yticks(np.arange(len(method_info)))
    ax.set_yticklabels(
        [f"{m[0]} {m[1]}th ({m[2]}s)" if m[2] else f"{m[0]} {m[1]}th" for m in method_info],
        color='white'
    )

    ax.set_xlabel("Cell Index", color='white')
    ax.set_title(f"ΔF/F Events > {threshold}× Noise", color='white')

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.set_label("# suprathreshold events", color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    fig.tight_layout()
    fig.patch.set_alpha(0)

    if savename:
        plt.savefig(savename, bbox_inches='tight', facecolor='black', dpi=300)
        plt.close()
    else:
        plt.show()


def save_dff_traces_by_method(dffs, method_info, outpath):
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    for method_ix, method in enumerate(method_info):
        method_name = f"{method[0]}_{method[1]}th" + (f"_{method[2]}s" if method[2] else "")
        method_dir = outpath / method_name
        method_dir.mkdir(exist_ok=True)

        traces = []
        for cell_ix in sorted(dffs.keys()):
            traces.append(dffs[cell_ix][method_ix])

        traces = np.stack(traces, axis=0)  # shape: (ncells, nframes)
        np.save(method_dir / "dF_traces.npy", traces)

def plot_dff_comparison(
        trace,
        dffs,
        method_info,
        f0s,
        cell_index,
        fps=17,
        minimal=True,
        savename="dff_comparison2.png"
):
    time = np.arange(len(trace)) / fps
    dff_list = [d[:len(trace)] for d in dffs[cell_index]]
    f0_list = [f[:len(trace)] for f in f0s[cell_index]]

    dff_min = min(np.nanmin(d) for d in dff_list)
    dff_max = max(np.nanmax(d) for d in dff_list)

    fig, axs = plt.subplots(len(dff_list), 2, figsize=(36, 2.5 * len(dff_list)), sharex=True, facecolor='black')

    for i, (method, dff, f0) in enumerate(zip(method_info, dff_list, f0_list)):
        label = f"{method[0]} {method[1]}th" + (f" ({method[2]}s)" if method[2] is not None else "")

        # Raw + baseline
        axs[i, 0].plot(time, trace, color='white', lw=1.5)
        axs[i, 0].plot(time, f0, color='cyan', lw=2)

        # Compute threshold-based event locations
        baseline_mask = dff < np.percentile(dff, 30)
        noise = np.nanstd(dff[baseline_mask])
        event_mask = (dff > 3 * noise)
        event_crossings = np.where(np.diff(event_mask.astype(int)) == 1)[0]
        event_times = time[event_crossings]

        # Plot event dots
        axs[i, 1].scatter(event_times, dff[event_crossings], s=30, color='magenta', zorder=10)

        # dF/F + horizontal zero line
        axs[i, 1].plot(time, dff, color='lime', lw=2)
        axs[i, 1].axhline(0, color='white', linestyle='--', linewidth=1)
        axs[i, 1].set_ylim(dff_min, dff_max)

        if minimal:
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')
        else:
            axs[i, 0].set_title(label, color='white', fontsize=14)
            axs[i, 0].set_facecolor("black")
            axs[i, 0].tick_params(colors='white')

            axs[i, 1].set_title(label, color='white', fontsize=14)
            axs[i, 1].set_facecolor("black")
            axs[i, 1].tick_params(colors='white')

    if not minimal:
        axs[-1, 0].set_xlabel("Time (s)", color='white', fontsize=14)
        axs[-1, 1].set_xlabel("Time (s)", color='white', fontsize=14)

    fig.tight_layout()
    fig.patch.set_alpha(0)
    plt.savefig(savename, bbox_inches='tight', facecolor='black', dpi=300, transparent=True)
    plt.show()

def standardized_noise(dff_trace, fps):
    diff = np.abs(np.diff(dff_trace))
    return np.median(diff) / np.sqrt(fps)

def plot_full_trace_panel(f, fneu, spks, cell_index, fps=17, spike_thresh=0.2, savename=None):
    time = np.arange(f.shape[1]) / fps
    f_raw = f[cell_index]
    f_n = fneu[cell_index]
    spk = spks[cell_index]

    corrected = f_raw - 0.7 * f_n
    f0 = np.median(corrected)
    dff = (corrected - f0) / f0

    fig, axs = plt.subplots(3, 1, figsize=(28, 8), sharex=True, facecolor="black")

    axs[0].plot(time, f_raw, color='white', lw=1.5, label='F')
    axs[0].plot(time, f_n, color='orange', lw=1, label='Fneu')
    axs[0].legend(frameon=False, loc='upper right', fontsize=12)
    axs[0].set_title(f"Raw Fluorescence (Cell {cell_index})", color='white', fontsize=14)

    axs[1].plot(time, dff, color='lime', lw=1.5)
    axs[1].axhline(0, color='white', linestyle='--', lw=1)
    axs[1].set_title("ΔF/F (static baseline)", color='white', fontsize=14)

    axs[2].plot(time, spk, color='cyan', lw=1.5, label='Deconvolved')
    if spike_thresh is not None:
        spike_times = np.where(spk > spike_thresh)[0]
        axs[2].scatter(time[spike_times], spk[spike_times], color='magenta', s=20, zorder=10, label='> threshold')
    axs[2].legend(frameon=False, loc='upper right', fontsize=12)
    axs[2].set_title("OASIS Deconvolved Spikes", color='white', fontsize=14)

    for ax in axs:
        ax.set_facecolor("black")
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')

    axs[-1].set_xlabel("Time (s)", color='white')
    fig.tight_layout()
    fig.patch.set_alpha(0)

    if savename:
        plt.savefig(savename, bbox_inches='tight', facecolor='black', dpi=300, transparent=True)
    plt.show()

def plot_drift_correction(trace, fps=17, filter_duration_s=30, savename=None):
    from scipy.ndimage import median_filter
    from sklearn.linear_model import LinearRegression

    filt_size = int(filter_duration_s * fps)
    if filt_size % 2 == 0:
        filt_size += 1

    trend = median_filter(trace, size=filt_size)
    X = np.arange(len(trend)).reshape(-1, 1)
    lr = LinearRegression().fit(X, trend)
    ramp = lr.predict(X)

    corrected = trace - trend + np.median(trend) - (ramp - np.median(ramp))

    fig, ax = plt.subplots(figsize=(28, 4), facecolor="black")
    ax.plot(trace, label='Raw', color='white', lw=1.5)
    ax.plot(trend, label='Trend (median)', color='cyan', lw=1.5)
    ax.plot(ramp, label='Ramp (regression)', color='magenta', lw=1.5)
    ax.plot(corrected, label='Corrected', color='lime', lw=1)

    ax.set_facecolor("black")
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.set_xlabel("Time (frames)", color='white', fontsize=14)
    ax.set_ylabel("Fluorescence", color='white', fontsize=14)
    ax.set_title("Drift Removal", color='white', fontsize=16)
    ax.legend(frameon=False, fontsize=12, loc='best', facecolor='black', edgecolor='white')
    for text in ax.get_legend().get_texts():
        text.set_color("white")

    fig.tight_layout()
    fig.patch.set_alpha(0)

    if savename:
        plt.savefig(savename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    else:
        plt.show()

    return corrected, trend, ramp

def suite2p_roi_overlay(
        ops,
        stat,
        iscell,
        proj=None,
        plot_indices=None,
        savename=None,
        color_mode='random',  # options: 'random', 'uniform', 'colormap'
        red_border=False
):
    ops = lsp.load_ops(ops)
    img = ops[proj]
    if img.shape != (ops["Ly"], ops["Lx"]):
        img_full = np.zeros((ops["Ly"], ops["Lx"]), dtype=np.float32)
        img_full[ops["yrange"][0]:ops["yrange"][1], ops["xrange"][0]:ops["xrange"][1]] = img
        img = img_full

    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    norm_img = np.clip((img - p1) / (p99 - p1), 0, 1)

    H = np.zeros_like(norm_img)
    S = np.zeros_like(norm_img)
    mask_total = np.zeros_like(norm_img, dtype=bool)
    iscell = np.asarray(iscell).astype(bool)
    if plot_indices is not None:
        indices = plot_indices
    else:
        indices = np.flatnonzero(iscell)
    if plot_indices is not None:
        indices = [n for n in indices if n in plot_indices]

    for i, n in enumerate(indices):
        s = stat[n]
        ypix, xpix = s["ypix"], s["xpix"]
        mask_total[ypix, xpix] = True

        if color_mode == 'random':
            hue = np.random.rand()
        elif color_mode == 'uniform':
            hue = 0.6  # cyan
        elif color_mode == 'colormap':
            hue = (i / max(len(indices), 1)) % 1.0
        else:
            raise ValueError("color_mode must be 'random', 'uniform', or 'colormap'")

        H[ypix, xpix] = hue
        S[ypix, xpix] = 1

    rgb = hsv_to_rgb(np.stack([H, S, norm_img], axis=-1))

    if red_border and mask_total.any():
        borders = find_boundaries(mask_total, mode='outer')
        rgb[borders] = [1, 0, 0]  # red

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    if savename:
        plt.savefig(savename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    else:
        plt.show()


#%% Load Data

ops = r"D:\demo\test\suite2p\anatomical_cpsam\ops.npy"
all_ops = lsp.load_ops(ops)
max_proj = all_ops["max_proj"]
all_res = lsp.load_planar_results(ops)
iscell = all_res["iscell"]
fneu = all_res["Fneu"][iscell]
spks = all_res["spks"][iscell]
f = all_res["F"][iscell]

peaks = np.nanmax(f, axis=1)
stds = np.nanstd(f, axis=1)

# ΔF from neuropil-corrected F
corrected_f = f - 0.7 * fneu
skews = skew(f, axis=1)
skews_corrected = skew(corrected_f, axis=1)

noise = np.array([standardized_noise(trace, fps=17) for trace in corrected_f])

# cell_groups, noise = split_high_low_noise_groups(res[0], fps=17, num_each=4)
cell_groups = pick_unique_cells(peaks, stds, skews, noise, num_each=5)
# selected_indices = sorted(set(sum(cell_groups.values(), [])))
selected_indices = [130, 138, 61, 129]

res = dff_methods(f, indices=selected_indices, fps=17)
# res = dff_methods(f, indices=selected_indices, fps=17)

def get_dff_line(trace, fps=17, window_s=30, percentile=8, duration_s=None, savename=None, title=None):
    nt = len(trace)
    t = np.arange(nt) / fps

    if duration_s is not None:
        nframes = int(duration_s * fps)
        trace = trace[:nframes]
        t = t[:nframes]

    win = int(window_s * fps)
    if win % 2 == 0:
        win += 1

    f0 = generic_filter(trace, lambda x: np.percentile(x, percentile), size=win, mode='nearest')
    dff = (trace - f0) / f0 * 100  # in percent
    return dff, t

# fname = Path(r"D://demo//test//plane10_roi2.tif")
# data = tifffile.memmap(fname)
# mdata = mbo.get_metadata(fname)
## FASTPLOTLIB

dff, t = get_dff_line(f[61], fps=17, window_s=30, percentile=8, duration_s=60)

fig = fpl.Figure()

figure = fpl.Figure(
    names=["max_proj", "proj_histogram"],
    shape=(1,2),
)

# add image to the corresponding subplots
figure["max_proj"].add_image(max_proj)

# add histogram to the corresponding subplots
figure["proj_histogram"].add_line(dff)

for subplot in figure:
    print(subplot.name)
    # if subplot.name == "proj_histogram":
    subplot.camera.maintain_aspect = False

figure.show()
fpl.loop.run()
