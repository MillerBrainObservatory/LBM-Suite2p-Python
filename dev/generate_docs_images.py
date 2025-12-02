"""
Generate documentation images for LBM-Suite2p-Python.

This script:
1. Takes raw ScanImage TIFFs as input
2. Assembles them with phase correction and FFT alignment
3. Saves to Zarr format (first 1500 frames)
4. Runs Suite2p processing via lsp.pipeline()
5. Generates comprehensive documentation images showcasing all postprocessing functions

Usage:
    python generate_docs_images.py --input /path/to/raw/tiffs --output /path/to/output

The script will create:
    output/
    ├── zarr/                    # Assembled Zarr files
    ├── suite2p/                 # Suite2p processing results
    └── docs_images/             # Generated documentation images
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Configure matplotlib for dark theme publication-quality figures
plt.style.use('dark_background')
plt.rcParams.update({
    'font.size': 11,
    'font.weight': 'bold',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.facecolor': 'black',
    'figure.facecolor': 'black',
    'axes.facecolor': 'black',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'legend.facecolor': 'black',
    'legend.edgecolor': 'white',
})


def assemble_raw_tiffs(input_path: Path, output_path: Path, num_frames: int = 1500):
    """
    Assemble raw ScanImage TIFFs with phase correction.

    Parameters
    ----------
    input_path : Path
        Path to directory containing raw ScanImage TIFFs
    output_path : Path
        Output directory for assembled Zarr files
    num_frames : int
        Number of frames to extract (default: 1500)

    Returns
    -------
    list[Path]
        List of paths to assembled Zarr files (one per plane)
    """
    import mbo_utilities as mbo

    print(f"\n{'='*60}")
    print("Step 1: Assembling raw ScanImage TIFFs")
    print(f"{'='*60}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Frames: {num_frames}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Load raw data with phase correction
    arr = mbo.imread(
        input_path,
        scan_phase=True,  # Apply scan phase correction
        use_fft=True,     # Use FFT for alignment
        register_z=True,  # Register across z-planes
    )

    print(f"  Loaded array shape: {arr.shape}")
    metadata = arr.metadata if hasattr(arr, 'metadata') else {}
    num_planes = arr.shape[1] if arr.ndim == 4 else 1

    # Extract first N frames
    if arr.shape[0] > num_frames:
        arr = arr[:num_frames]
        print(f"  Truncated to {num_frames} frames")

    # Save each plane as separate Zarr
    zarr_files = []
    for z in range(num_planes):
        plane_data = arr[:, z] if arr.ndim == 4 else arr
        zarr_path = output_path / f"plane{z+1:02d}.zarr"

        mbo.imwrite(zarr_path, plane_data, metadata=metadata)
        zarr_files.append(zarr_path)
        print(f"  Saved: {zarr_path.name}")

    return zarr_files


def run_suite2p_processing(zarr_files: list, output_path: Path, ops: dict = None):
    """
    Run Suite2p processing on assembled Zarr files.

    Parameters
    ----------
    zarr_files : list[Path]
        List of Zarr files to process
    output_path : Path
        Output directory for Suite2p results
    ops : dict, optional
        Custom Suite2p parameters

    Returns
    -------
    list[Path]
        List of paths to ops.npy files
    """
    import lbm_suite2p_python as lsp

    print(f"\n{'='*60}")
    print("Step 2: Running Suite2p processing")
    print(f"{'='*60}")

    if ops is None:
        ops = {
            "diameter": 6,
            "threshold_scaling": 1.0,
            "max_overlap": 0.75,
        }

    results = lsp.pipeline(
        input_data=zarr_files,
        save_path=output_path,
        ops=ops,
        keep_reg=True,
        keep_raw=False,
        force_reg=False,
        force_detect=False,
    )

    return results


def select_good_neurons(F, Fneu, stat, iscell, n_select=5):
    """
    Select neurons with good SNR and clear transients for visualization.

    Returns indices of selected neurons.
    """
    from scipy.stats import skew

    # Only consider accepted cells
    iscell_mask = iscell[:, 0].astype(bool)

    # Neuropil-corrected fluorescence
    F_corr = F - 0.7 * Fneu

    # Compute metrics for all neurons
    baseline = np.percentile(F_corr, 20, axis=1, keepdims=True)
    baseline = np.maximum(baseline, 1e-6)
    dff = (F_corr - baseline) / baseline

    # SNR
    signal = np.std(dff, axis=1)
    noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.6745
    snr = signal / (noise + 1e-6)

    # Skewness (positive = transients)
    skewness = skew(dff, axis=1)

    # Combined score: want high SNR, positive skewness, and accepted cells
    score = np.zeros(len(F))
    score[iscell_mask] = snr[iscell_mask] + 0.5 * np.clip(skewness[iscell_mask], 0, 10)

    # Get top neurons by score
    top_idx = np.argsort(score)[::-1][:n_select]

    return top_idx


def generate_dff_comparison_figure(F, Fneu, fs, iscell, stat, save_path):
    """
    Generate figure comparing different dF/F calculation methods.
    Shows a zoomed time window with clear transients.
    """
    from lbm_suite2p_python import dff_rolling_percentile, dff_median_filter

    print("  Generating: dff_methods_comparison.png")

    # Select good neurons for visualization
    selected_idx = select_good_neurons(F, Fneu, stat, iscell, n_select=5)
    n_neurons = len(selected_idx)
    n_frames = F.shape[1]

    # Neuropil-corrected fluorescence for selected neurons
    F_sel = F[selected_idx]
    Fneu_sel = Fneu[selected_idx]
    F_corr = F_sel - 0.7 * Fneu_sel

    # Calculate dF/F with different methods
    dff_rolling = dff_rolling_percentile(F_corr, window_size=300, percentile=20)
    dff_rolling_10 = dff_rolling_percentile(F_corr, window_size=300, percentile=10)
    dff_median = dff_median_filter(F_corr)

    # Select a time window with activity (20-60 seconds or frames 280-840 at 14Hz)
    # Adjust based on frame rate
    start_sec, end_sec = 20, 60
    start_frame = int(start_sec * fs)
    end_frame = min(int(end_sec * fs), n_frames)
    time = np.arange(end_frame - start_frame) / fs + start_sec

    fig, axes = plt.subplots(n_neurons, 4, figsize=(16, 2.5*n_neurons))
    if n_neurons == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_neurons):
        # Raw fluorescence
        axes[i, 0].plot(time, F_corr[i, start_frame:end_frame], color='cyan', lw=0.8)
        axes[i, 0].set_ylabel(f'Cell {selected_idx[i]}', fontweight='bold')
        if i == 0:
            axes[i, 0].set_title('Raw F (neuropil corrected)', fontweight='bold')

        # Rolling percentile 20
        axes[i, 1].plot(time, dff_rolling[i, start_frame:end_frame], color='lime', lw=0.8)
        if i == 0:
            axes[i, 1].set_title('dF/F: 20th percentile', fontweight='bold')

        # Rolling percentile 10
        axes[i, 2].plot(time, dff_rolling_10[i, start_frame:end_frame], color='yellow', lw=0.8)
        if i == 0:
            axes[i, 2].set_title('dF/F: 10th percentile', fontweight='bold')

        # Median filter
        axes[i, 3].plot(time, dff_median[i, start_frame:end_frame], color='magenta', lw=0.8)
        if i == 0:
            axes[i, 3].set_title('dF/F: Median filter', fontweight='bold')

    for ax in axes[-1]:
        ax.set_xlabel('Time (s)', fontweight='bold')

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path / "dff_methods_comparison.png", facecolor='black', edgecolor='none')
    plt.close()


def generate_dff_window_size_figure(F, Fneu, fs, iscell, stat, save_path):
    """
    Generate figure showing effect of window size on dF/F calculation.
    """
    from lbm_suite2p_python import dff_rolling_percentile

    print("  Generating: dff_window_size_effect.png")

    # Select best neuron for visualization
    selected_idx = select_good_neurons(F, Fneu, stat, iscell, n_select=1)
    best_neuron = selected_idx[0]

    F_corr = F - 0.7 * Fneu
    n_frames = F.shape[1]

    trace = F_corr[best_neuron:best_neuron+1]

    # Select a time window with activity
    start_sec, end_sec = 10, 80
    start_frame = int(start_sec * fs)
    end_frame = min(int(end_sec * fs), n_frames)
    time = np.arange(end_frame - start_frame) / fs + start_sec

    # Different window sizes
    window_sizes = [50, 150, 300, 600]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Raw trace
    axes[0].plot(time, trace[0, start_frame:end_frame], color='white', lw=1)
    axes[0].set_ylabel('Fluorescence (a.u.)', fontweight='bold')
    axes[0].set_title(f'Cell {best_neuron} - Raw Fluorescence (neuropil corrected)', fontweight='bold')

    # dF/F with different window sizes
    for ws, color in zip(window_sizes, colors):
        dff = dff_rolling_percentile(trace, window_size=ws, percentile=20)
        label = f'Window = {ws} frames ({ws/fs:.1f}s)'
        axes[1].plot(time, dff[0, start_frame:end_frame], color=color, lw=1, label=label)

    axes[1].set_xlabel('Time (s)', fontweight='bold')
    axes[1].set_ylabel('dF/F', fontweight='bold')
    axes[1].set_title('Effect of Window Size on dF/F Calculation', fontweight='bold')
    axes[1].legend(loc='upper right', framealpha=0.8)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path / "dff_window_size_effect.png", facecolor='black', edgecolor='none')
    plt.close()


def generate_shot_noise_figure(F, Fneu, fs, iscell, save_path):
    """
    Generate figure showing shot noise distribution and example traces.
    No quality classification bar chart.
    """
    from lbm_suite2p_python import dff_rolling_percentile, dff_shot_noise

    print("  Generating: shot_noise_analysis.png")

    # Calculate dF/F and shot noise
    # NOTE: dff_shot_noise expects dF/F in percent units (multiply by 100)
    F_corr = F - 0.7 * Fneu
    dff = dff_rolling_percentile(F_corr, window_size=300, percentile=20)
    dff_percent = dff * 100  # Convert to percent for shot noise calculation
    noise = dff_shot_noise(dff_percent, fs)

    # Separate accepted and rejected
    iscell_mask = iscell[:, 0].astype(bool)
    noise_acc = noise[iscell_mask]
    noise_rej = noise[~iscell_mask]

    n_frames = F.shape[1]

    # Select time window
    start_sec, end_sec = 10, 70
    start_frame = int(start_sec * fs)
    end_frame = min(int(end_sec * fs), n_frames)
    time = np.arange(end_frame - start_frame) / fs + start_sec

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.2])

    # Histogram of noise levels
    ax1 = fig.add_subplot(gs[0])
    bins = np.linspace(0, np.percentile(noise, 99), 50)
    ax1.hist(noise_acc, bins=bins, alpha=0.8, label=f'Accepted (n={len(noise_acc)})', color='#4ECDC4')
    ax1.hist(noise_rej, bins=bins, alpha=0.8, label=f'Rejected (n={len(noise_rej)})', color='#FF6B6B')
    ax1.axvline(0.5, color='lime', linestyle='--', lw=2, label='Excellent (<0.5)')
    ax1.axvline(1.0, color='yellow', linestyle='--', lw=2, label='Good (<1.0)')
    ax1.set_xlabel('Shot Noise (%/√Hz)', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Shot Noise Distribution', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Example traces: low vs high noise
    # Find lowest and highest noise neurons (among accepted only for cleaner display)
    noise_acc_idx = np.where(iscell_mask)[0]
    noise_acc_values = noise[iscell_mask]
    sorted_by_noise = np.argsort(noise_acc_values)

    low_noise_idx = noise_acc_idx[sorted_by_noise[:3]]
    high_noise_idx = noise_acc_idx[sorted_by_noise[-3:]]

    ax2 = fig.add_subplot(gs[1])

    # Plot low noise traces at TOP (best quality)
    for i, idx in enumerate(low_noise_idx):
        offset = (5 - i) * 1.5  # Top of plot
        ax2.plot(time, dff[idx, start_frame:end_frame] + offset, color='#4ECDC4', lw=0.8)
        ax2.text(time[-1] + 0.5, offset, f'ν={noise[idx]:.2f}', fontsize=10, color='#4ECDC4', fontweight='bold')

    # Plot high noise traces at BOTTOM
    for i, idx in enumerate(high_noise_idx):
        offset = (2 - i) * 1.5  # Bottom of plot
        ax2.plot(time, dff[idx, start_frame:end_frame] + offset, color='#FF6B6B', lw=0.8)
        ax2.text(time[-1] + 0.5, offset, f'ν={noise[idx]:.2f}', fontsize=10, color='#FF6B6B', fontweight='bold')

    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('dF/F (stacked)', fontweight='bold')
    ax2.set_title('Low Noise (cyan, top) vs High Noise (red, bottom)', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path / "shot_noise_analysis.png", facecolor='black', edgecolor='none')
    plt.close()


def generate_quality_score_figure(F, Fneu, stat, fs, iscell, save_path):
    """
    Generate figure showing quality score components and trace comparison.
    Simplified: histograms + scatter + traces. No bar charts.
    """
    from lbm_suite2p_python import compute_trace_quality_score, dff_rolling_percentile

    print("  Generating: quality_score_breakdown.png")

    # Only use accepted cells for cleaner visualization
    iscell_mask = iscell[:, 0].astype(bool)
    F_acc = F[iscell_mask]
    Fneu_acc = Fneu[iscell_mask]
    stat_acc = stat[iscell_mask]

    result = compute_trace_quality_score(F_acc, Fneu_acc, stat_acc, fs)
    sort_idx = result['sort_idx']

    # Calculate dF/F
    F_corr = F_acc - 0.7 * Fneu_acc
    dff = dff_rolling_percentile(F_corr, window_size=300, percentile=20)

    # Recalculate shot noise in proper units
    from lbm_suite2p_python import dff_shot_noise
    shot_noise_proper = dff_shot_noise(dff * 100, fs)

    n_frames = F_acc.shape[1]

    # Select time window
    start_sec, end_sec = 15, 75
    start_frame = int(start_sec * fs)
    end_frame = min(int(end_sec * fs), n_frames)
    time = np.arange(end_frame - start_frame) / fs + start_sec

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2])

    # SNR distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(result['snr'], bins=50, color='#45B7D1', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax1.axvline(np.median(result['snr']), color='lime', linestyle='--', lw=2,
                label=f"Median: {np.median(result['snr']):.2f}")
    ax1.set_xlabel('SNR', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Signal-to-Noise Ratio', fontweight='bold')
    ax1.legend(framealpha=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Skewness distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(result['skewness'], bins=50, color='#96CEB4', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2.axvline(0, color='white', linestyle='-', lw=1, alpha=0.5)
    ax2.axvline(np.median(result['skewness']), color='lime', linestyle='--', lw=2,
                label=f"Median: {np.median(result['skewness']):.2f}")
    ax2.set_xlabel('Skewness', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('Skewness (positive = transients)', fontweight='bold')
    ax2.legend(framealpha=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Shot noise distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(shot_noise_proper, bins=50, color='#FF6B6B', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax3.axvline(np.median(shot_noise_proper), color='lime', linestyle='--', lw=2,
                label=f"Median: {np.median(shot_noise_proper):.2f}")
    ax3.set_xlabel('Shot Noise (%/√Hz)', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Shot Noise (lower = better)', fontweight='bold')
    ax3.legend(framealpha=0.8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Top vs Bottom traces
    ax4 = fig.add_subplot(gs[1, :])

    # Top 5 quality traces at TOP of plot
    for i, idx in enumerate(sort_idx[:5]):
        offset = (9 - i) * 1.2  # Top positions
        ax4.plot(time, dff[idx, start_frame:end_frame] + offset, color='#4ECDC4', lw=0.6)
        ax4.text(time[0] - 1, offset, f'#{i+1}', fontsize=10, color='#4ECDC4', fontweight='bold', ha='right')

    # Bottom 5 quality traces at BOTTOM of plot
    for i, idx in enumerate(sort_idx[-5:]):
        offset = (4 - i) * 1.2  # Bottom positions
        ax4.plot(time, dff[idx, start_frame:end_frame] + offset, color='#FF6B6B', lw=0.6)
        ax4.text(time[0] - 1, offset, f'#{len(sort_idx)-4+i}', fontsize=10, color='#FF6B6B', fontweight='bold', ha='right')

    ax4.set_xlabel('Time (s)', fontweight='bold')
    ax4.set_ylabel('dF/F (stacked)', fontweight='bold')
    ax4.set_title('Top 5 Quality (cyan, top) vs Bottom 5 Quality (red, bottom)', fontweight='bold')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path / "quality_score_breakdown.png", facecolor='black', edgecolor='none')
    plt.close()


def generate_trace_sorting_figure(F, Fneu, stat, fs, iscell, save_path):
    """
    Generate figure showing traces before and after quality sorting.
    """
    from lbm_suite2p_python import sort_traces_by_quality, dff_rolling_percentile

    print("  Generating: trace_sorting_comparison.png")

    # Only use accepted cells
    iscell_mask = iscell[:, 0].astype(bool)
    F_acc = F[iscell_mask]
    Fneu_acc = Fneu[iscell_mask]
    stat_acc = stat[iscell_mask]

    F_sorted, sort_idx, quality = sort_traces_by_quality(F_acc, Fneu_acc, stat_acc, fs)

    # Calculate dF/F for both
    F_corr = F_acc - 0.7 * Fneu_acc
    dff_orig = dff_rolling_percentile(F_corr, window_size=300, percentile=20)
    dff_sorted = dff_orig[sort_idx]

    n_show = min(20, F_acc.shape[0])
    n_frames = F_acc.shape[1]

    # Select time window
    start_sec, end_sec = 10, 80
    start_frame = int(start_sec * fs)
    end_frame = min(int(end_sec * fs), n_frames)
    time = np.arange(end_frame - start_frame) / fs + start_sec

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Original order
    for i in range(n_show):
        offset = (n_show - 1 - i) * 1.5  # Top to bottom
        axes[0].plot(time, dff_orig[i, start_frame:end_frame] + offset, color='white', lw=0.4, alpha=0.8)

    axes[0].set_xlabel('Time (s)', fontweight='bold')
    axes[0].set_ylabel('Neuron (original order)', fontweight='bold')
    axes[0].set_title(f'Original Order (first {n_show} neurons)', fontweight='bold')
    axes[0].set_yticks([])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['left'].set_visible(False)

    # Sorted by quality (best at top)
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, n_show))
    for i in range(n_show):
        offset = (n_show - 1 - i) * 1.5  # Top to bottom
        axes[1].plot(time, dff_sorted[i, start_frame:end_frame] + offset, color=colors[i], lw=0.4, alpha=0.9)

    axes[1].set_xlabel('Time (s)', fontweight='bold')
    axes[1].set_ylabel('Neuron (sorted by quality)', fontweight='bold')
    axes[1].set_title('Sorted by Quality Score (best at top)', fontweight='bold')
    axes[1].set_yticks([])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path / "trace_sorting_comparison.png", facecolor='black', edgecolor='none')
    plt.close()


def generate_normalize_traces_figure(F, Fneu, iscell, stat, save_path):
    """
    Generate figure showing trace normalization methods.
    """
    from lbm_suite2p_python.postprocessing import normalize_traces

    print("  Generating: trace_normalization.png")

    # Select good neurons
    selected_idx = select_good_neurons(F, Fneu, stat, iscell, n_select=10)
    n_show = len(selected_idx)

    F_sel = F[selected_idx]
    Fneu_sel = Fneu[selected_idx]
    F_corr = F_sel - 0.7 * Fneu_sel

    # Different normalization modes
    F_per_neuron = normalize_traces(F_corr, mode="per_neuron")
    F_percentile = normalize_traces(F_corr, mode="percentile")

    n_frames = F_corr.shape[1]

    # Select time window
    start_frame = int(10 * 14)  # Assuming ~14 Hz
    end_frame = min(int(80 * 14), n_frames)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Raw traces
    for i in range(n_show):
        offset = (n_show - 1 - i)
        axes[0].plot(F_corr[i, start_frame:end_frame] / np.max(np.abs(F_corr[i, start_frame:end_frame])) + offset,
                     color='white', lw=0.4, alpha=0.8)
    axes[0].set_title('Raw F (scaled per neuron)', fontweight='bold')
    axes[0].set_xlabel('Frame', fontweight='bold')
    axes[0].set_ylabel('Neuron', fontweight='bold')
    axes[0].set_yticks([])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['left'].set_visible(False)

    # Per-neuron normalization
    for i in range(n_show):
        offset = (n_show - 1 - i)
        axes[1].plot(F_per_neuron[i, start_frame:end_frame] + offset, color='#45B7D1', lw=0.4, alpha=0.8)
    axes[1].set_title('normalize_traces(mode="per_neuron")\nMin-max scaling', fontweight='bold')
    axes[1].set_xlabel('Frame', fontweight='bold')
    axes[1].set_yticks([])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(False)

    # Percentile normalization
    for i in range(n_show):
        offset = (n_show - 1 - i)
        axes[2].plot(F_percentile[i, start_frame:end_frame] + offset, color='#96CEB4', lw=0.4, alpha=0.8)
    axes[2].set_title('normalize_traces(mode="percentile")\n1st-99th percentile scaling', fontweight='bold')
    axes[2].set_xlabel('Frame', fontweight='bold')
    axes[2].set_yticks([])
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path / "trace_normalization.png", facecolor='black', edgecolor='none')
    plt.close()


def generate_event_exceptionality_figure(F, Fneu, fs, iscell, stat, save_path):
    """
    Generate figure showing event exceptionality detection.
    Just histogram + example traces (no log-probability panel).
    """
    from lbm_suite2p_python.postprocessing import compute_event_exceptionality
    from lbm_suite2p_python import dff_rolling_percentile

    print("  Generating: event_exceptionality.png")

    # Only use accepted cells
    iscell_mask = iscell[:, 0].astype(bool)
    F_acc = F[iscell_mask]
    Fneu_acc = Fneu[iscell_mask]

    # Use spike traces for event detection
    F_corr = F_acc - 0.7 * Fneu_acc
    dff = dff_rolling_percentile(F_corr, window_size=300, percentile=20)

    # Compute exceptionality
    fitness, erfc, sd_r, md = compute_event_exceptionality(dff, N=5, robust_std=False)

    n_frames = F_acc.shape[1]

    # Select time window
    start_sec, end_sec = 10, 80
    start_frame = int(start_sec * fs)
    end_frame = min(int(end_sec * fs), n_frames)
    time = np.arange(end_frame - start_frame) / fs + start_sec

    # Find neurons with most exceptional events (lowest fitness)
    best_neurons = np.argsort(fitness)[:5]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Fitness score distribution
    axes[0].hist(fitness, bins=50, color='#45B7D1', alpha=0.8, edgecolor='white', linewidth=0.5)
    axes[0].axvline(np.percentile(fitness, 10), color='lime', linestyle='--', lw=2,
                    label=f'10th percentile: {np.percentile(fitness, 10):.2f}')
    axes[0].set_xlabel('Fitness Score (lower = more exceptional)', fontweight='bold')
    axes[0].set_ylabel('Count', fontweight='bold')
    axes[0].set_title('Event Exceptionality Distribution', fontweight='bold')
    axes[0].legend(framealpha=0.8)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Example traces with exceptional events (best at top)
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, 5))
    for i, idx in enumerate(best_neurons):
        offset = (4 - i) * 2  # Top to bottom
        axes[1].plot(time, dff[idx, start_frame:end_frame] + offset, color=colors[i], lw=0.8)
        axes[1].text(time[-1] + 0.5, offset, f'fitness={fitness[idx]:.2f}', fontsize=10,
                     color=colors[i], fontweight='bold')

    axes[1].set_xlabel('Time (s)', fontweight='bold')
    axes[1].set_ylabel('dF/F (stacked)', fontweight='bold')
    axes[1].set_title('Top 5 Most Exceptional Neurons (lowest fitness, best at top)', fontweight='bold')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path / "event_exceptionality.png", facecolor='black', edgecolor='none')
    plt.close()


def generate_all_docs_images(suite2p_path: Path, output_path: Path):
    """
    Generate all documentation images from Suite2p results.

    Parameters
    ----------
    suite2p_path : Path
        Path to Suite2p output directory
    output_path : Path
        Path to save documentation images
    """
    import lbm_suite2p_python as lsp

    print(f"\n{'='*60}")
    print("Step 3: Generating documentation images")
    print(f"{'='*60}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Find ops files
    ops_files = list(suite2p_path.glob("**/ops.npy"))
    if not ops_files:
        print("  ERROR: No ops.npy files found!")
        return

    print(f"  Found {len(ops_files)} processed planes")

    # Load first plane for single-plane examples
    ops_file = ops_files[0]
    print(f"  Loading: {ops_file}")

    ops = lsp.load_ops(ops_file)
    results = lsp.load_planar_results(ops_file)

    F = results['F']
    Fneu = results['Fneu']
    stat = results['stat']
    iscell = results['iscell']
    fs = ops.get('fs', 30.0)

    print(f"  Loaded {F.shape[0]} ROIs, {F.shape[1]} frames")
    print(f"  Frame rate: {fs} Hz")
    print(f"  Accepted: {iscell[:, 0].sum()}, Rejected: {(~iscell[:, 0].astype(bool)).sum()}")

    # Generate individual figures (removed filter_functions_figure)
    print("\nGenerating documentation figures...")

    generate_dff_comparison_figure(F, Fneu, fs, iscell, stat, output_path)
    generate_dff_window_size_figure(F, Fneu, fs, iscell, stat, output_path)
    generate_shot_noise_figure(F, Fneu, fs, iscell, output_path)
    generate_quality_score_figure(F, Fneu, stat, fs, iscell, output_path)
    generate_trace_sorting_figure(F, Fneu, stat, fs, iscell, output_path)
    generate_normalize_traces_figure(F, Fneu, iscell, stat, output_path)
    generate_event_exceptionality_figure(F, Fneu, fs, iscell, stat, output_path)

    print(f"\n{'='*60}")
    print("Documentation images generated successfully!")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate documentation images for LBM-Suite2p-Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Path to raw ScanImage TIFFs (or already-assembled Zarr files)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--frames", "-n",
        type=int,
        default=1500,
        help="Number of frames to extract (default: 1500)"
    )
    parser.add_argument(
        "--skip-assembly",
        action="store_true",
        help="Skip assembly step (use existing Zarr files in output/zarr)"
    )
    parser.add_argument(
        "--skip-suite2p",
        action="store_true",
        help="Skip Suite2p processing (use existing results in output/suite2p)"
    )
    parser.add_argument(
        "--suite2p-path",
        type=Path,
        default=None,
        help="Path to existing Suite2p results (if --skip-suite2p)"
    )

    args = parser.parse_args()

    # Setup paths
    output_base = args.output
    zarr_path = output_base / "zarr"
    suite2p_path = args.suite2p_path or output_base / "suite2p"
    docs_path = output_base / "docs_images"

    output_base.mkdir(parents=True, exist_ok=True)

    # Step 1: Assembly
    if not args.skip_assembly and args.input:
        zarr_files = assemble_raw_tiffs(args.input, zarr_path, args.frames)
    else:
        # Find existing Zarr files
        zarr_files = list(zarr_path.glob("*.zarr"))
        if not zarr_files:
            print(f"No Zarr files found in {zarr_path}")
            if not args.skip_suite2p:
                return
        else:
            print(f"Using existing Zarr files: {len(zarr_files)} found")

    # Step 2: Suite2p processing
    if not args.skip_suite2p:
        run_suite2p_processing(zarr_files, suite2p_path)

    # Step 3: Generate documentation images
    generate_all_docs_images(suite2p_path, docs_path)

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")
    print(f"\nGenerated images are in: {docs_path}")
    print("\nTo use in documentation, copy images to docs/_images/ and reference in .md files")


if __name__ == "__main__":
    main()
