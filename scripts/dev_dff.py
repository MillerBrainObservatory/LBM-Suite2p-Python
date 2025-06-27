import mbo_utilities as mbo
import lbm_suite2p_python as lsp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from lbm_suite2p_python.utils import dff_rolling_percentile
from lbm_suite2p_python.utils import _resize_masks_fit_crop
from suite2p.detection.stats import ROI

ops = r"D:\demo\test\suite2p\anatomical_cpsam\ops.npy"
all_res = lsp.load_planar_results(ops)
iscell = all_res["iscell"]
fneu = all_res["Fneu"][iscell]
spks = all_res["spks"][iscell]
f = all_res["F"][iscell]

def split_high_low_noise_groups(dffs_all, fps=17, num_each=4):
    dffs_static = [d[0] for d in dffs_all.values()]
    noise = np.array([np.median(np.abs(np.diff(d))) / np.sqrt(fps) for d in dffs_static])
    sorted_indices = np.argsort(noise)

    return {
        "Low Noise": sorted_indices[:num_each].tolist(),
        "High Noise": sorted_indices[-num_each:][::-1].tolist()
    }, noise

def pick_unique_cells(peaks, stds, num_each=4):
    selected = set()
    cells = {"High Peak": [], "High Std": []}

    def pick_top(metric, label):
        indices = np.argsort(metric)[::-1]
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

def plot_dff_event_counts(dffs, method_info, threshold=3, savename="dff_event_counts.png"):

    counts = {label: [] for label in method_info}

    for _dff_list in dffs.values():
        for label, dff in zip(method_info, _dff_list):
            baseline_mask = dff < np.percentile(dff, 30)
            noise = np.nanstd(dff[baseline_mask])
            crossings = np.diff((dff > threshold * noise).astype(int)) == 1
            counts[label].append(np.sum(crossings))

    count_matrix = np.array([counts[label] for label in method_info])

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='black')
    im = ax.imshow(count_matrix, aspect='auto', cmap='viridis')

    num_cells = len(dffs)
    xticks = [0, num_cells // 3, 2 * num_cells // 3, num_cells - 1]
    xticklabels = [list(dffs.keys())[i] for i in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, color='white')
    ax.set_yticks(np.arange(len(method_info)))
    ax.set_yticklabels([f"{m[0]} {m[1]}th ({m[2]}s)" if m[2] else f"{m[0]} {m[1]}th" for m in method_info], color='white')

    ax.set_xlabel("Cell Index", color='white')
    ax.set_title(f"ΔF/F Events > {threshold}× Noise", color='white')

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.set_label("# suprathreshold events", color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    fig.tight_layout()
    fig.patch.set_alpha(0)
    plt.savefig(savename, bbox_inches='tight', facecolor='black', dpi=300)
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

peaks = np.nanmax(f, axis=1)
stds = np.nanstd(f, axis=1)
# cell_groups, noise = split_high_low_noise_groups(res[0], fps=17, num_each=4)
cell_groups = pick_unique_cells(peaks, stds, num_each=4)
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
    plot_dff_comparison(
        trace,
        res[0], res[1], res[2], cell_index=cell_index, fps=17,
        savename=f"dff_comparison{i}.png"
    )

#%% Counting significant events in dF/F traces
def count_dff_events(dffs, method_info, threshold=3):
    counts = {label: [] for label in method_info}

    for _dff_list in dffs.values():
        for label, dff in zip(method_info, _dff_list):
            baseline_mask = dff < np.percentile(dff, 30)
            noise = np.nanstd(dff[baseline_mask])
            crossings = np.diff((dff > threshold * noise).astype(int)) == 1
            counts[label].append(int(np.sum(crossings)))  # ensure plain int
    return counts

event_counts = count_dff_events(res[0], res[1], threshold=3)

#%% Save traces
save_path = Path(r"D:\demo\test\suite2p\anatomical_cpsam\traces")
save_path.mkdir(exist_ok=True)
save_dff_traces_by_method(res[0], res[1], save_path)

plot_dff_event_counts(res[0], res[1], threshold=3)

all_cells_dff = dff_methods(f, np.arange(f.shape[0]), fps=17)
plot_dff_event_counts(all_cells_dff[0], all_cells_dff[1], threshold=3)

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

plot_full_trace_panel(f, fneu, spks, cell_index=cell_index, fps=17, spike_thresh=0.2, savename="cell0_panel.png")

for label in ["High Peak", "High Std"]:
    for i, cell_index in enumerate(cell_groups[label]):
        plot_full_trace_panel(f, fneu, spks, cell_index, fps=17, spike_thresh=0.2,
                              savename=f"panel_{label.replace(' ', '_')}_{i}.png")


def feather_mask(mask, edge_width=3, center_alpha=0.4, edge_alpha=0.85):
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import gaussian_filter
    dist_to_edge = distance_transform_edt(mask == 0)
    alpha = np.clip((edge_width - dist_to_edge) / edge_width, 0, 1)
    alpha = gaussian_filter(alpha, sigma=0.7)
    return center_alpha + alpha * (edge_alpha - center_alpha)


def plot_projection(
    ops,
    savepath=None,
    fig_label=None,
    vmin=None,
    vmax=None,
    add_scalebar=False,
    proj="meanImg",
    display_masks=False,
    edge_width=3,
    center_alpha=0.4,
    edge_alpha=0.85,
):
    from matplotlib import cm

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
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    vmin = np.nanpercentile(data, 2) if vmin is None else vmin
    vmax = np.nanpercentile(data, 98) if vmax is None else vmax
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)

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
        iscell = np.array(res["iscell"])
        rois = [ROI.from_stat_dict(s) for s in stat]
        accepted_rois = [r for r, c in zip(rois, iscell) if c]

        colors = cm.get_cmap("tab20")
        for i, roi in enumerate(accepted_rois):
            mask = _resize_masks_fit_crop(roi.to_array(ops["Ly"], ops["Lx"]), shape)
            alpha = feather_mask(mask > 0, edge_width=edge_width,
                                 center_alpha=center_alpha, edge_alpha=edge_alpha)
            color = np.array(colors(i / max(1, len(accepted_rois)))[:3]).reshape(1, 1, 3)

            overlay = np.ones((*shape, 3), dtype=np.float32) * color
            ax.imshow(overlay, alpha=alpha)

        ax.text(
            0.5,
            1.02,
            f"Accepted: {len(accepted_rois):03d}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            fontname="Courier New",
            color="lime",
            ha="center",
            va="bottom",
        )

    if add_scalebar and "dx" in ops:
        pixel_size = ops["dx"]
        scale_bar_length = 100 / pixel_size
        scalebar_x = shape[1] * 0.05
        scalebar_y = shape[0] * 0.90
        ax.add_patch(Rectangle((scalebar_x, scalebar_y), scale_bar_length, 5,
                               edgecolor="white", facecolor="white"))
        ax.text(scalebar_x + scale_bar_length / 2, scalebar_y - 10, "100 μm",
                color="white", fontsize=10, ha="center", fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, facecolor="black")
        plt.close(fig)
    else:
        plt.show()


output_ops = lsp.load_ops(ops)
plot_projection(
    output_ops,
    r"D:\demo\seg2.png",
    fig_label="test",
    display_masks=True,
    add_scalebar=True,
    proj="meanImg",
    edge_width = 3,
    center_alpha = 0.4,
    edge_alpha = 0.85
)
plt.savefig()



### MEDIAN FILTER AND REGRESSION
from scipy.ndimage import median_filter
from sklearn.linear_model import LinearRegression

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.linear_model import LinearRegression

def extract_drift_removal(trace, fps=17, filter_duration_s=30):
    from scipy.ndimage import median_filter
    from sklearn.linear_model import LinearRegression

    filt_size = int(filter_duration_s * fps)
    if filt_size % 2 == 0:
        filt_size += 1

    trend = median_filter(trace, size=filt_size)

    X = np.arange(len(trend)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, trend)
    ramp = lr.predict(X)

    corrected = trace - trend + np.median(trend) - (ramp - np.median(ramp))
    return corrected, trend, ramp

corrected, trend, ramp = extract_drift_removal(trace, fps=17)
plt.figure(figsize=(12, 6), facecolor="black")
plt.plot(trace, label='Original Trace', color='blue', lw=1)
plt.plot(trend, label='Median Filtered Trend', color='cyan', lw=1.5)
plt.plot(ramp, label='Linear Ramp', color='magenta', lw=1.5)
plt.plot(corrected, label='Corrected Trace', color='lime', lw=1)
plt.xlabel('Time (frames)', color='white')
plt.ylabel('Fluorescence Intensity', color='white')
plt.title('Drift Removal in Fluorescence Trace', color='white')
plt.legend(frameon=False, loc='upper right', fontsize=12)
plt.tick_params(colors='white')
plt.tight_layout()
# plt.savefig("drift_removal_example.png", dpi=300, facecolor="black")
plt.show()

def suite2p_roi_overlay(ops, stat, iscell, proj='meanImg'):
    img = ops[proj]
    if img.shape != (ops["Ly"], ops["Lx"]):
        img_full = np.zeros((ops["Ly"], ops["Lx"]), dtype=np.float32)
        img_full[ops["yrange"][0]:ops["yrange"][1], ops["xrange"][0]:ops["xrange"][1]] = img
        img = img_full

    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    norm_img = np.clip((img - p1) / (p99 - p1), 0, 1)

    H = np.zeros_like(norm_img)
    S = np.zeros_like(norm_img)
    for n, s in enumerate(stat):
        if iscell[n]:
            H[s["ypix"], s["xpix"]] = np.random.rand()
            S[s["ypix"], s["xpix"]] = 1

    rgb = hsv_to_rgb(np.stack([H, S, norm_img], axis=-1))

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# usage
suite2p_roi_overlay(lsp.load_ops(ops), all_res["stat"], all_res["iscell"], proj='max_proj')
