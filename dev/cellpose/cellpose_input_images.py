"""Render the images suite2p 1.1.0 feeds into Cellpose, for inspection.

suite2p 1.1.0 picks the Cellpose detection image with detection.cellpose_settings.img,
one of "max_proj / meanImg" (default), "meanImg", "max_proj". The old enhanced-mean
option (anatomical_only=3) is gone; spatial sharpening is now the separate
cellpose_settings.highpass_spatial knob, applied on top of whichever image is chosen.

For each of the three images this reproduces the exact array select_rois hands to
Cellpose (suite2p/detection/anatomical.py) at several highpass_spatial values, plus the
legacy meanImgE (registration/GUI only -- never fed to Cellpose) for reference. PNGs are
written to a sibling dir of the raw data.

Source images are built as in detect.detection_wrapper:
    meanImg  = binned.mean(0)                         # before temporal high-pass
    max_proj = temporal_high_pass_filter(binned).max(0)
suite2p computes these on the registered movie; this runs on the loaded frames, so
absolute sharpness differs -- the option logic and transforms are identical.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mbo_utilities as mbo

from suite2p.detection.detect import bin_movie
from suite2p.detection.utils import temporal_high_pass_filter
from suite2p.registration import highpass_mean_image  # legacy 'enhanced mean' (meanImgE)
from scipy.ndimage import gaussian_filter

try:
    from cellpose.transforms import normalize99
except Exception:
    def normalize99(img):  # same 1/99-percentile normalization
        x = img.astype(np.float32)
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        return (x - lo) / max(hi - lo, 1e-12)

OPTIONS = ["meanImg", "max_proj", "max_proj / meanImg"]
SLUG = {"meanImg": "meanImg", "max_proj": "max_proj",
        "max_proj / meanImg": "max_proj_over_meanImg"}
# old anatomical_only int each img option corresponds to
OLD_NUM = {"meanImg": 2, "max_proj": 4, "max_proj / meanImg": 1}


def cellpose_input(mean_img, max_proj, img, highpass_spatial=0.0, diameter=12.0):
    """The exact image select_rois hands to Cellpose for a given img option."""
    if img == "max_proj / meanImg":
        out = np.log(np.maximum(1e-3, max_proj / np.maximum(1e-3, mean_img)))
    elif img == "meanImg":
        out = mean_img.copy()
    elif img == "max_proj":
        out = max_proj.copy()
    else:
        raise ValueError(f"suite2p 1.1.0 accepts only {OPTIONS} -- got {img!r}")
    if highpass_spatial:
        out = np.clip(normalize99(out), 0, 1)
        out -= gaussian_filter(out, diameter * highpass_spatial)
        out -= gaussian_filter(out, diameter * highpass_spatial)
    return out


def save_gray(path, im):
    lo, hi = np.percentile(im, 1), np.percentile(im, 99)
    plt.imsave(path, im, cmap="gray", vmin=lo, vmax=float(max(hi, lo + 1e-6)))


def panel(ax, im, title):
    lo, hi = np.percentile(im, 1), np.percentile(im, 99)
    ax.imshow(im, cmap="gray", vmin=lo, vmax=float(max(hi, lo + 1e-6)))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=r"E:/demo/mk355/raw", help="raw data dir/file")
    p.add_argument("--plane", type=int, default=7, help="0-based z-plane")
    p.add_argument("--nframes", type=int, default=1000, help="frames to load; -1 = all")
    p.add_argument("--spatial-hp", type=float, nargs="+", default=[0, 0.5, 1, 3],
                   help="highpass_spatial values to render")
    p.add_argument("--tau", type=float, default=1.0, help="indicator timescale (s)")
    p.add_argument("--highpass-time", type=int, default=100, help="temporal high-pass width")
    p.add_argument("--diameter", type=float, default=12.0, help="cell diameter (px)")
    p.add_argument("--out", default=None,
                   help="output dir (default: sibling 'cellpose_input_images' of raw)")
    args = p.parse_args()

    raw = Path(args.raw)
    out_dir = Path(args.out) if args.out else raw.parent / "cellpose_input_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = mbo.imread(str(raw))
    fs = float(arr.metadata["fs"])
    n = arr.shape[0] if args.nframes < 0 else min(args.nframes, arr.shape[0])
    mov = np.asarray(arr[:n, 0, args.plane]).astype(np.float32)  # (T, Ly, Lx)
    T, Ly, Lx = mov.shape
    print(f"plane {args.plane}: {mov.shape}  fs={fs} Hz")

    # reproduce detection_wrapper's image construction
    nbins = 5000
    bin_size = int(max(1, T // nbins, round(args.tau * fs)))
    binned = bin_movie(mov, bin_size, yrange=[0, Ly], xrange=[0, Lx], nbins=nbins)
    mean_img = binned.mean(axis=0)
    max_proj = temporal_high_pass_filter(mov=binned.copy(), width=args.highpass_time).max(axis=0)
    print(f"bin_size={bin_size}  binned={binned.shape}")

    hps = args.spatial_hp
    written = []

    # per-image, per-spatial_hp PNGs
    for opt in OPTIONS:
        for hp in hps:
            im = cellpose_input(mean_img, max_proj, opt, hp, args.diameter)
            fp = out_dir / f"{SLUG[opt]}_hp{hp:g}.png"
            save_gray(fp, im)
            written.append(fp)

    # legacy enhanced mean (not a Cellpose option) for reference
    mE = highpass_mean_image(mean_img.astype("float32"), aspect=1.0)
    fp = out_dir / "meanImgE_reference.png"
    save_gray(fp, mE)
    written.append(fp)

    # combined grid: rows = image option, cols = spatial_hp
    fig, axs = plt.subplots(len(OPTIONS), len(hps),
                            figsize=(3.6 * len(hps), 3.9 * len(OPTIONS)), squeeze=False)
    for r, opt in enumerate(OPTIONS):
        for c, hp in enumerate(hps):
            im = cellpose_input(mean_img, max_proj, opt, hp, args.diameter)
            tag = "  [DEFAULT]" if opt == "max_proj / meanImg" else ""
            panel(axs[r, c], im, f"img='{opt}'{tag}\n(old={OLD_NUM[opt]})  spatial_hp={hp:g}")
    fig.suptitle(f"Cellpose detection images -- {raw}  plane {args.plane}", fontsize=12)
    fig.tight_layout()
    grid_fp = out_dir / "cellpose_inputs_grid.png"
    fig.savefig(grid_fp, dpi=130)
    plt.close(fig)
    written.append(grid_fp)

    print(f"wrote {len(written)} files to {out_dir}")
    for fp in written:
        print(" ", fp.name)


if __name__ == "__main__":
    main()
