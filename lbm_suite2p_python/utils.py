from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import suite2p

def plot_registration(ops, savepath):
    plt.subplot(1, 4, 1)
    plt.imshow(ops['refImg'], cmap='gray', )
    plt.title("Reference Image for Registration")

    plt.subplot(1, 4, 2)
    plt.imshow(ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection")

    plt.subplot(1, 4, 3)
    plt.imshow(ops['meanImg'], cmap='gray')
    plt.title("Mean registered image")

    plt.subplot(1, 4, 4)
    plt.imshow(ops['meanImgE'], cmap='gray')
    plt.title("High-pass filtered Mean registered image")
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    print(f'Saved to {savepath}')

def plot_segmentation(ops, savepath):

    stats_file = Path(ops['save_path']).joinpath('stat.npy')
    iscell = np.load(Path(ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
    stats = np.load(stats_file, allow_pickle=True)

    im = suite2p.ROI.stats_dicts_to_3d_array(stats, Ly=ops['Ly'], Lx=ops['Lx'], label_id=True)
    im[im == 0] = np.nan

    plt.subplot(1, 4, 1)
    plt.imshow(ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection")

    plt.subplot(1, 4, 2)
    plt.imshow(np.nanmax(im, axis=0), cmap='jet')
    plt.title("All ROIs Found")

    plt.subplot(1, 4, 3)
    plt.imshow(np.nanmax(im[~iscell], axis=0, ), cmap='jet')
    plt.title("All Non-Cell ROIs")

    plt.subplot(1, 4, 4)
    plt.imshow(np.nanmax(im[iscell], axis=0), cmap='jet')
    plt.title("All Cell ROIs")
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')

def plot_traces(ops, savepath):

    f_cells = np.load(Path(ops['save_path']).joinpath('F.npy'))
    f_neuropils = np.load(Path(ops['save_path']).joinpath('Fneu.npy'))
    spks = np.load(Path(ops['save_path']).joinpath('spks.npy'))

    plt.figure(figsize=[20, 20])
    plt.suptitle("Fluorescence and Deconvolved Traces for Different ROIs", y=0.92)

    num_rois = min(20, len(f_cells))
    rois = np.random.choice(len(f_cells), num_rois, replace=False)

    T = f_cells.shape[1]
    subsample_factor = max(1, T // 300)
    timepoints = np.arange(0, T, subsample_factor)

    for i, roi in enumerate(rois):
        plt.subplot(num_rois, 1, i + 1)
        f = f_cells[roi][::subsample_factor]
        f_neu = f_neuropils[roi][::subsample_factor]
        sp = spks[roi][::subsample_factor]

        fmax = np.maximum(f.max(), f_neu.max())
        fmin = np.minimum(f.min(), f_neu.min())
        frange = fmax - fmin
        sp /= sp.max()
        sp *= frange

        plt.plot(timepoints, f, label="Cell Fluorescence")
        plt.plot(timepoints, f_neu, label="Neuropil Fluorescence")
        plt.plot(timepoints, sp + fmin, label="Deconvolved")

        plt.xticks(np.linspace(0, T, 10))
        plt.ylabel(f"ROI {roi}", rotation=0)
        plt.xlabel("Frame")

        if i == 0:
            plt.legend(bbox_to_anchor=(0.93, 2))

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')

def make_subdir_from_list(files: list):
    """Put each file in a list of filepaths into its own subdirectory"""
    for file in files:
        fpath = file.parent / file.stem
        plane_name = fpath.stem.rpartition('_')[:-2][0]
        plane_path = file.parent / plane_name
        plane_path.mkdir(exist_ok=True)
        new_fname = plane_path / file.name
        file.rename(new_fname)