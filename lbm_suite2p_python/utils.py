import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import tifffile
import numpy as np

import suite2p

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


def get_zplane_stats(ops_files: list[str | Path], overwrite: bool=True):
    plane_stats = {}
    for i, file in enumerate(ops_files):
        output_ops = np.load(file, allow_pickle=True).item()
        iscell = np.load(Path(output_ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
        num_accepted = len(iscell)
        num_rejected = len(~iscell)
        plane_stats[i + 1] = (num_accepted, num_rejected, file)

    common_path = os.path.commonpath(ops_files)
    plane_save = os.path.join(common_path, "plane_stats.npy")
    plane_stats = np.array(
        list(plane_stats.items()),
        dtype=[("plane", "i4"), ("accepted_rejected", "2i4")],
    )

    if not Path(plane_save).is_file():
        np.save(plane_save, plane_stats)
    elif Path(plane_save).is_file() and overwrite:
        np.save(plane_save, plane_stats)
    else:
        print(f"File {plane_save} already exists. Skipping.")
    return plane_stats

def post_process(ops, overwrite=True):
    """Plot registration, segmentation and traces to the ops path."""
    print(f"Post processing started.")
    filenames = {
        "registration.png": plot_registration,
        "segmentation.png": plot_segmentation,
        "traces.png": plot_traces
    }

    for fname, plot_func in filenames.items():
        path = Path(ops["save_path"]) / fname
        if overwrite or not path.exists():
            plot_func(ops, str(path))

def plot_registration(ops, savepath):

    plt.figure()
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
    plt.savefig(savepath, dpi=300)
    print(f'Saved to {savepath}')

def plot_segmentation(ops, savepath):

    stats_file = Path(ops['save_path']).joinpath('stat.npy')
    iscell = np.load(Path(ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
    stats = np.load(stats_file, allow_pickle=True)

    im = suite2p.ROI.stats_dicts_to_3d_array(stats, Ly=ops['Ly'], Lx=ops['Lx'], label_id=True)
    im[im == 0] = np.nan

    plt.figure()
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
    plt.savefig(savepath, dpi=300)

def plot_traces(ops, savepath):

    f_cells = np.load(Path(ops['save_path']).joinpath('F.npy'))
    f_neuropils = np.load(Path(ops['save_path']).joinpath('Fneu.npy'))
    spks = np.load(Path(ops['save_path']).joinpath('spks.npy'))

    plt.figure()
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
    plt.savefig(savepath, dpi=300)

def combine_tiffs(files):
    """
    Combine tiff files into a single tiff file.

    Input Tyx * N_files gives TNyx
    """
    # Load the first file to get the shape
    first_file = files[0]
    first_tiff = tifffile.imread(first_file)
    num_files = len(files)
    num_frames, height, width = first_tiff.shape

    # Create the new tiff file
    new_tiff = np.zeros((num_frames * num_files, height, width), dtype=first_tiff.dtype)

    # Load the tiffs
    for i, f in enumerate(files):
        tiff = tifffile.imread(f)
        new_tiff[i * num_frames:(i + 1) * num_frames] = tiff

    return new_tiff

def make_subdir_from_list(files: list):
    """Put each file in a list of filepaths into its own subdirectory"""
    for file in files:
        fpath = file.parent / file.stem
        plane_name = fpath.stem.rpartition('_')[:-2][0]
        plane_path = file.parent / plane_name
        plane_path.mkdir(exist_ok=True)
        new_fname = plane_path / file.name
        file.rename(new_fname)