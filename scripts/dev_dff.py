import mbo_utilities as mbo  # noqa
import lbm_suite2p_python as lsp
import numpy as np
from scipy.stats import skew
from pathlib import Path
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from skimage.segmentation import find_boundaries


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
        [f"{m[0]} {m[1]} ({m[2]}s)" if m[2] else f"{m[0]} {m[1]}" for m in method_info],
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
        print(method)
        if "median" in method:
            label = f"{method[0]} {method[1]}" + (f" ({method[2]}s)" if method[2] is not None else "")
        else:
            label = f"{method[0]} {method[1]}th percentile" + (f" ({method[2]}s)" if method[2] is not None else "")

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
            axs[i, 0].set_facecolor("black")
            axs[i, 0].tick_params(colors='white')

            axs[i, 1].set_facecolor("black")
            axs[i, 1].tick_params(colors='white')

    if not minimal:
        axs[-1, 0].set_xlabel("Time (s)", color='white', fontsize=14)
        axs[-1, 1].set_xlabel("Time (s)", color='white', fontsize=14)

    fig.tight_layout()
    fig.patch.set_alpha(0)
    plt.savefig(savename, bbox_inches='tight', facecolor='black', dpi=300, transparent=True)
    plt.show()

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

def plot_rastermap_clusters(f, model, show_individual=False, n_examples=3):
    labels = model.labels_
    n_clusters = np.max(labels) + 1
    time = np.arange(f.shape[1])

    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 2 * n_clusters), sharex=True)
    for k in range(n_clusters):
        cluster_traces = f[labels == k]
        mean_trace = np.mean(cluster_traces, axis=0)

        ax = axes[k] if n_clusters > 1 else axes
        ax.plot(time, mean_trace, color='black', label=f'Cluster {k} mean')

        if show_individual:
            for trace in cluster_traces[:n_examples]:
                ax.plot(time, trace, alpha=0.3)

        ax.set_ylabel(f'C{k}')
        ax.legend()

    axes[-1].set_xlabel('Time (frames)')
    plt.tight_layout()
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_rastermap_clusters(f, model, show_individual=False, n_examples=3):
        labels = model.labels_
        n_clusters = np.max(labels) + 1
        time = np.arange(f.shape[1])

        fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 2 * n_clusters), sharex=True)
        for k in range(n_clusters):
            cluster_traces = f[labels == k]
            mean_trace = np.mean(cluster_traces, axis=0)

            ax = axes[k] if n_clusters > 1 else axes
            ax.plot(time, mean_trace, color='black', label=f'Cluster {k} mean')

            if show_individual:
                for trace in cluster_traces[:n_examples]:
                    ax.plot(time, trace, alpha=0.3)

            ax.set_ylabel(f'C{k}')
            ax.legend()

        axes[-1].set_xlabel('Time (frames)')
        plt.tight_layout()
        plt.show()

    plt.show()

#%% Counting significant events in dF/F traces
ops = r"D:\demo\test\suite2p\anatomical_cpsam\ops.npy"
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
selected_indices = sorted(set(sum(cell_groups.values(), [])))

res = dff_methods(f, indices=selected_indices, fps=17)

for i in range(4):
    label = "High Std"
    cell_index = cell_groups[label][i]  # 4th cell in "High Mean" row
    trace = f[cell_index]

    fig, ax = plt.subplots(figsize=(28, 4), facecolor="black")
    ax.plot(trace, color='w', lw=1)
    ax.plot(spks[cell_index], color='cyan', lw=1, label='Static Baseline')
    print(min(trace))
    ax.set_facecolor("black")
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    plt.tight_layout()
    plt.savefig(
        f"example_trace_deonv{i}.png",
        bbox_inches='tight',
        facecolor='black',
        dpi=300,
        transparent=True
    )
    plt.show()

#%% Save traces
save_path = Path(r"D:\demo\test\suite2p\anatomical_cpsam\traces")
save_path.mkdir(exist_ok=True)
# save_dff_traces_by_method(res[0], res[1], save_path)

# all_cells_dff = dff_methods(f, np.arange(f.shape[0]), fps=17)
# plot_dff_event_counts(all_cells_dff[0], all_cells_dff[1], threshold=3)

root = Path(r"D:\demo\strategies")
root.mkdir(exist_ok=True)

for label in ["High Peak", "High Std", "High Skew", "Low Noise"]:
    label_text = label.lower().replace(" ", "_")
    fpath = root / label_text
    fpath.mkdir(exist_ok=True)

    for i, cell_index in enumerate(cell_groups[label]):
        trace = f[cell_index]

        panel_savename = fpath / f"panel_{label_text}_{i}.png"
        plot_full_trace_panel(
            f, fneu, spks, cell_index,
            fps=17, spike_thresh=0.2, savename=panel_savename
        )

        drift_savename = fpath / f"drift_{label_text}_{i}.png"
        corrected, trend, ramp = plot_drift_correction(
            trace, fps=17, savename=drift_savename
        )

        s2p_roi_savename = fpath / f"s2p_roi_overlay_{label_text}_{i}.png"
        suite2p_roi_overlay(
            ops,
            all_res["stat"],
            all_res["iscell"],
            plot_indices=[cell_index],
            proj="max_proj",
            color_mode='uniform',
            red_border=True,
            savename=s2p_roi_savename
        )

        dff_comp_savename = fpath / f"dff_comp_{label_text}_{i}.png"
        dff_comp_savename.parent.mkdir(exist_ok=True)
        plot_dff_comparison(
            trace,
            res[0], res[1], res[2], cell_index=cell_index, fps=17,
            minimal=False,
            savename=dff_comp_savename
        )


#%%
selected_indices = [130, 138, 61, 129]
res = dff_methods(f, indices=selected_indices, fps=17)

for i, cell_index in enumerate(selected_indices):
    label_text = f"cell_{cell_index}"
    fpath = root / label_text
    fpath.mkdir(exist_ok=True)

    trace = f[cell_index]

    panel_savename = fpath / f"panel_{label_text}.png"
    plot_full_trace_panel(f, fneu, spks, cell_index, fps=17, spike_thresh=0.2, savename=panel_savename)

    s2p_roi_savename = fpath / f"s2p_roi_overlay_{label_text}.png"
    suite2p_roi_overlay(
        ops,
        all_res["stat"],
        all_res["iscell"],
        plot_indices=[cell_index],
        proj="max_proj",
        color_mode='uniform',
        red_border=True,
        savename=s2p_roi_savename
    )

    dff_comp_savename = fpath / f"dff_comp_{label_text}.png"
    dff_comp_savename.parent.mkdir(exist_ok=True)
    plot_dff_comparison(
        trace,
        res[0], res[1], res[2], cell_index=cell_index, fps=17,
        minimal=False,
        savename=dff_comp_savename
    )


def plot_single_trace(trace, fps=17, savename=None, title=None):
    time = np.arange(len(trace)) / fps
    fig, ax = plt.subplots(figsize=(28, 4), facecolor="black")
    ax.plot(time, trace, color='white', lw=1.0)
    ax.set_xlabel("Time (s)", color='white', fontsize=14)
    if title:
        ax.set_title(title, color='white', fontsize=16)
    ax.set_facecolor("black")
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    fig.tight_layout()
    fig.patch.set_alpha(0)
    if savename:
        plt.savefig(savename, bbox_inches='tight', facecolor='black', dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()

plot_single_trace(f[61, :, ], fps=17, title="Cell 61 Trace")

def plot_single_dff_trace(trace, fps=17, window_s=30, percentile=8, duration_s=None, savename=None, title=None):
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
    dff = (trace - f0) / f0

    fig, ax = plt.subplots(figsize=(28, 4), facecolor="black")
    ax.plot(t, dff, color='lime', lw=1.5)
    ax.axhline(0, color='white', linestyle='--', lw=1)

    label = f"ΔF/F (win={window_s}s, {percentile}th percentile)"
    ax.set_title(title or label, color='white', fontsize=16)

    ax.set_xlabel("Time (s)", color='white', fontsize=14)
    ax.set_facecolor("black")
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    fig.tight_layout()
    fig.patch.set_alpha(0)

    if savename:
        plt.savefig(savename, bbox_inches='tight', facecolor='black', dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()

plot_single_dff_trace(f[61], fps=17, window_s=30, percentile=8, duration_s=60, title="Cell 61 ΔF/F")

def plot_dff(trace, fps=17, window_s=30, percentile=8, duration_s=None, savename=None, title=None):
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

    fig, ax1 = plt.subplots(figsize=(28, 4), facecolor="black")
    ax2 = ax1.twinx()

    ax1.plot(t, trace, color='white', lw=1.5)
    ax2.plot(t, dff, color='lime', lw=1.5)

    ax1.set_xlabel("Time (s)", color='white', fontsize=14)
    if title:
        ax1.set_title(title, color='white', fontsize=16)

    for ax in (ax1, ax2):
        ax.set_facecolor("black")
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

    ax1.spines['bottom'].set_color('white')
    fig.tight_layout()
    fig.patch.set_alpha(0)

    if savename:
        plt.savefig(savename, bbox_inches='tight', facecolor='black', dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()

root = Path(r"D:\demo\strategies")
for cell in selected_indices:
    import itertools
    win = [10, 30, 60, 120]
    percentiles = [8, 20, 40, 60]

    grid_search_dir = root / f"cell{cell}" / "grid_search"
    grid_search_dir.mkdir(parents=True, exist_ok=True)
    for w, p in itertools.product(win, percentiles):
        savename = grid_search_dir / f"raw_and_dff_win{w}_p{p}.png"
        plot_dff(
            f[cell], fps=17, window_s=w, percentile=p, duration_s=600,
            title=f"Cell 61 Raw and ΔF/F (win={w}s, p={p})",
            savename=savename
        )

def plot_noise_extremes(f, dff_noise, fps=17, savename=None):
    time = np.arange(f.shape[1]) / fps
    sorted_indices = np.argsort(dff_noise)
    low_noise = sorted_indices[:5]
    high_noise = sorted_indices[-5:][::-1]

    fig, axs = plt.subplots(5, 2, figsize=(18, 10), sharex=True, facecolor="black")

    for i in range(5):
        for col, idx in zip([0, 1], [low_noise[i], high_noise[i]]):
            trace = f[idx]
            dff = (trace - np.median(trace)) / np.median(trace)
            ax = axs[i, col]
            ax.plot(time, dff, color='lime', lw=1.5)
            ax.axhline(0, color='white', linestyle='--', lw=1)
            ax.set_ylabel(f"Cell {idx}", color='white')
            ax.set_facecolor("black")
            ax.tick_params(colors='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')

    axs[0, 0].set_title("Lowest Noise", color='white', fontsize=14)
    axs[0, 1].set_title("Highest Noise", color='white', fontsize=14)
    axs[-1, 0].set_xlabel("Time (s)", color='white')
    axs[-1, 1].set_xlabel("Time (s)", color='white')

    fig.tight_layout()
    fig.patch.set_alpha(0)

    if savename:
        plt.savefig(savename, bbox_inches='tight', facecolor='black', dpi=300, transparent=True)
    plt.show()

res = lsp.load_planar_results(ops)
f = res["F"][res["iscell"]]
dff = lsp.dff_rolling_percentile(f, window_size=300, percentile=20) * 100
noise = lsp.dff_shot_noise(dff, 17)

plot_noise_extremes(f, noise, fps=17, savename=r"D:\demo\strategies\noise_extremes.png")

time = np.arange(f.shape[1]) / 17
sorted_indices = np.argsort(noise)
low_noise = sorted_indices[:5]
high_noise = sorted_indices[-5:][::-1]

plot_single_dff_trace(f[low_noise[3]], fps=17, window_s=30, percentile=20, duration_s=600,)

cell_index = low_noise[3]
trace = f[cell_index]
res = dff_methods(f, indices=[cell_index], fps=17)
plot_dff_comparison(trace, *res, cell_index=cell_index, fps=17, minimal=True, savename=r"D:\demo\dff_baseline.png")

lsp.suite2p_roi_overlay(
    ops,
    res["stat"],
    res["iscell"], proj="max_proj", plot_indices=[130])


res = lsp.load_planar_results(ops)
f = res["F"][res["iscell"]]
dff = lsp.dff_rolling_percentile(f, window_size=300, percentile=20) * 100
noise = lsp.dff_shot_noise(dff, 17)

sorted_indices = np.argsort(noise)
low_noise = sorted_indices[:5]
high_noise = sorted_indices[-5:][::-1]

print("Low noise indices:", low_noise)
print("High noise indices:", high_noise)
print("Low noise values:", noise[low_noise])
print("High noise values:", noise[high_noise])
