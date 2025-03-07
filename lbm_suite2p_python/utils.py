import numpy as np
from pathlib import Path

from matplotlib import patches

import matplotlib.pyplot as plt
import matplotlib as mpl

import suite2p
from scipy.ndimage import percentile_filter

mpl.rcParams.update({
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.subplot.wspace': .01,
    'figure.subplot.hspace': .01,
    'figure.figsize': (18, 13),
    'ytick.major.left': True,
})
jet = mpl.cm.get_cmap('jet')
jet.set_bad(color='k')


def load_ops(ops_input: str | Path | list[str | Path]):
    if isinstance(ops_input, (str, Path)):
        return np.load(ops_input, allow_pickle=True).item()
    elif isinstance(ops_input, dict):
        return ops_input


def resize_to_max_proj(mask, target_shape):
    """Centers a mask within the target shape, cropping if too large or padding if too small."""
    sy, sx = mask.shape
    ty, tx = target_shape

    # If mask is larger, crop it
    if sy > ty or sx > tx:
        start_y = (sy - ty) // 2
        start_x = (sx - tx) // 2
        return mask[start_y:start_y + ty, start_x:start_x + tx]

    # If mask is smaller, pad it
    resized_mask = np.zeros(target_shape, dtype=mask.dtype)
    start_y = (ty - sy) // 2
    start_x = (tx - sx) // 2
    resized_mask[start_y:start_y + sy, start_x:start_x + sx] = mask
    return resized_mask


def plot_registration(ops, savepath, fig_label=None):
    fig, axes = plt.subplots(1, 4, figsize=(12, 6), facecolor='black')

    for i, (ax, (key, title)) in enumerate(zip(axes, [
        ('refImg', "Reference Image"),
        ('max_proj', "Max Projection"),
        ('meanImg', "Mean Image"),
        ('meanImgE', "High-passed Filtered Mean Image")
    ])):
        ax.imshow(ops[key], cmap='gray')
        ax.set_title(title, fontweight='bold', color='white', fontsize=14, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0 and fig_label:
            ax.set_ylabel(fig_label, fontweight='bold', color='white', fontsize=14)

    plt.tight_layout()
    savepath = Path(savepath)
    if not savepath.parent.is_dir():
        savepath.mkdir(parents=True)
    plt.savefig(savepath, dpi=300, facecolor='black')
    plt.show()


def plot_segmentation(ops, savepath, fig_label=None, accepted_only=False, vmin=None, vmax=None, add_scalebar=False):
    savepath = Path(savepath)

    stats_file = Path(ops['save_path']).joinpath('stat.npy')
    iscell = np.load(Path(ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
    stats = np.load(stats_file, allow_pickle=True)

    im = suite2p.ROI.stats_dicts_to_3d_array(stats, Ly=ops['Ly'], Lx=ops['Lx'], label_id=True)
    im[im == 0] = np.nan

    accepted_cells = np.sum(iscell)
    rejected_cells = np.sum(~iscell)

    max_proj = ops['max_proj']
    shape = max_proj.shape

    # Resize masks correctly (using your function)
    cell_rois = resize_to_max_proj(np.nanmax(im[iscell], axis=0) if np.any(iscell) else np.zeros_like(im[0]), shape)

    green_overlay = np.zeros((*shape, 4), dtype=np.float32)
    green_overlay[..., 1] = 1
    green_overlay[..., 3] = (cell_rois > 0) * 1.0

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')

    if vmin is None:
        vmin = np.nanpercentile(max_proj, 2)
    if vmax is None:
        vmax = np.nanpercentile(max_proj, 98)

    ax.imshow(max_proj, cmap='gray', vmin=vmin, vmax=vmax)
    ax.imshow(green_overlay)

    if not accepted_only:
        non_cell_rois = resize_to_max_proj(np.nanmax(im[~iscell], axis=0) if np.any(~iscell) else np.zeros_like(im[0]),
                                           shape)

        magenta_overlay = np.zeros((*shape, 4), dtype=np.float32)
        magenta_overlay[..., 0] = 1
        magenta_overlay[..., 2] = 1
        magenta_overlay[..., 3] = (non_cell_rois > 0) * 0.5

        ax.imshow(magenta_overlay)

    title_text = f"Accepted: {accepted_cells:03d}   |   Rejected: {rejected_cells:03d}"
    ax.text(0.37, 1.02, f"Accepted: {accepted_cells:03d}", transform=ax.transAxes,
            fontsize=14, fontweight='bold', fontname="Courier New",
            color='lime', ha='right', va='bottom')

    ax.text(0.63, 1.02, f"Rejected: {rejected_cells:03d}", transform=ax.transAxes,
            fontsize=14, fontweight='bold', fontname="Courier New",
            color='magenta', ha='left', va='bottom')

    if fig_label:
        fig_label = fig_label.replace("_", " ").replace("-", " ").replace(".", " ")
        ax.set_ylabel(fig_label, color='white', fontweight='bold', fontsize=12)

    ax.set_xticks([])
    ax.set_yticks([])

    if add_scalebar and 'dx' in ops:
        pixel_size = ops['dx']
        scale_bar_length = 100 / pixel_size

        scalebar_x = shape[1] * 0.05
        scalebar_y = shape[0] * 0.90

        ax.add_patch(patches.Rectangle(
            (scalebar_x, scalebar_y), scale_bar_length, 5, edgecolor='white', facecolor='white'))
        ax.text(scalebar_x + scale_bar_length / 2, scalebar_y - 10, "100 μm",
                color='white', fontsize=10, ha='center', fontweight='bold')

    plt.tight_layout()
    savepath.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(savepath, dpi=300, facecolor='black')
    plt.show()


def plot_traces(ops, savepath, nframes=None, ntraces=5, show_best=False):
    fps = ops['fs']
    f_cells = np.load(Path(ops['save_path']).joinpath('F.npy'))
    f_neuropils = np.load(Path(ops['save_path']).joinpath('Fneu.npy'))
    spks = np.load(Path(ops['save_path']).joinpath('spks.npy'))

    total_rois, total_frames = f_cells.shape
    total_time = total_frames / fps
    max_time = 600 if total_time > 600 else total_time
    nframes = min(nframes if nframes is not None else int(max_time * fps), total_frames)

    if max_time < 1:
        time_factor, time_unit = 1000, "ms"
    elif max_time < 60:
        time_factor, time_unit = 1, "s"
    elif max_time < 3600:
        time_factor, time_unit = 1 / 60, "min"
    else:
        time_factor, time_unit = 1 / 3600, "hr"

    timepoints = np.linspace(0, max_time * time_factor, nframes)
    subsample_factor = max(1, nframes // 300)
    timepoints = timepoints[::subsample_factor]

    valid_rois = np.where(f_cells.mean(axis=1) > 0)[0]

    if show_best:
        z_F = (f_cells[valid_rois] - f_cells[valid_rois].mean(axis=1, keepdims=True)) / (
                    f_cells[valid_rois].std(axis=1, keepdims=True) + 1e-10)
        z_Fneu = (f_neuropils[valid_rois] - f_neuropils[valid_rois].mean(axis=1, keepdims=True)) / (
                    f_neuropils[valid_rois].std(axis=1, keepdims=True) + 1e-10)
        z_scores = np.abs(z_F.mean(axis=1) - z_Fneu.mean(axis=1))
        selected_rois = valid_rois[np.argsort(z_scores)[-ntraces:][::-1]]
    else:
        selected_rois = np.random.choice(valid_rois, ntraces, replace=False)

    fig, axes = plt.subplots(ntraces, 1, figsize=(12, 2 * ntraces), sharex=True)

    if show_best:
        fig.suptitle("Traces with Largest Difference (of Z Scores) from Neuropil", fontsize=24, fontweight='bold', fontname="Arial", y=0.95)
    else:
        fig.suptitle("Fluorescence and Deconvolved Traces for Randomly Selected ROIs", fontsize = 26, fontweight = 'bold', fontname = "Arial", y = 0.98)

    for i, roi in enumerate(selected_rois):
        ax = axes[i]
        f = f_cells[roi, :nframes][::subsample_factor]
        f_neu = f_neuropils[roi, :nframes][::subsample_factor]
        sp = spks[roi, :nframes][::subsample_factor]

        fmax = max(f.max(), f_neu.max())
        fmin = min(f.min(), f_neu.min())
        frange = fmax - fmin
        if sp.max() > 0:
            sp = (sp / sp.max()) * frange

        ax.plot(timepoints, f, label="Cell Fluorescence", linewidth=1.5)
        ax.plot(timepoints, f_neu, label="Neuropil Fluorescence", linewidth=1.5)
        ax.plot(timepoints, sp + fmin, label="Deconvolved", linewidth=1.5, linestyle='dashed')

        ax.set_yticks([fmin, fmax])
        ax.set_yticklabels([f"{fmin:.1f}", f"{fmax:.1f}"], fontsize=12, fontweight='bold', fontname="Arial",
                           rotation=45)

        ax.set_ylabel(f"ROI {roi}", rotation=90, labelpad=10, fontsize=14, fontweight='bold', fontname="Arial",
                      va="center", ha="right")

        if i == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), fontsize=14, ncol=3, frameon=False)

    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=16, fontweight='bold', fontname="Arial")
    axes[-1].set_xlim([0, timepoints[-1]])

    plt.subplots_adjust(top=0.88)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if not savepath.parent.is_dir():
        savepath.mkdir(parents=True)

    plt.savefig(savepath, dpi=300)
    plt.show()


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def compute_dff(f_trace, window_size=300, percentile=8):
    """
    Compute ΔF/F₀ using a rolling percentile baseline.

    Parameters:
    -----------
    f_trace : np.ndarray
        (N_neurons, N_frames) fluorescence traces.
    window_size : int
        Size of the rolling window (in frames).
    percentile : int
        Percentile to use for baseline F₀ estimation.

    Returns:
    --------
    dff : np.ndarray
        (N_neurons, N_frames) ΔF/F₀ traces.
    """
    f0 = np.array([
        percentile_filter(f, percentile, size=window_size, mode='nearest')
        for f in f_trace
    ])
    return (f_trace - f0) / (f0 + 1e-6)  # 1e-6 to avoid division by zero

