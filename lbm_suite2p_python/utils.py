from pathlib import Path
import matplotlib.pyplot as plt

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


def make_subdir_from_list(files: list):
    """Put each file in a list of filepaths into its own subdirectory"""
    for file in files:
        fpath = file.parent / file.stem
        plane_name = fpath.stem.rpartition('_')[:-2][0]
        plane_path = file.parent / plane_name
        plane_path.mkdir(exist_ok=True)
        new_fname = plane_path / file.name
        file.rename(new_fname)