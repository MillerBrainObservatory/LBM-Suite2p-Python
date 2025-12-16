#!/usr/bin/env python
"""
generate documentation images by running lsp.pipeline() on sample data.

usage:
    # run full pipeline with demo data defaults
    uv run python scripts/generate_images.py

    # run full pipeline with custom paths
    uv run python scripts/generate_images.py --input D:/data/raw --output D:/data/results

    # generate projection comparison from existing results
    uv run python scripts/generate_images.py --ops D:/demo/results/plane01_stitched/ops.npy --projections-only

    # skip pipeline, just organize existing images
    uv run python scripts/generate_images.py --skip-pipeline

demo defaults:
    input:  D:/demo/raw
    output: D:/demo/results
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def normalize99(img):
    """normalize to 0-1 using 1st/99th percentiles."""
    p1, p99 = np.percentile(img, [1, 99])
    return np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)


def apply_hp_filter(img, diameter, spatial_hp_cp):
    """apply high-pass filter (suite2p preprocessing)."""
    img_norm = normalize99(img)
    if spatial_hp_cp > 0:
        sigma = diameter * spatial_hp_cp
        img_hp = img_norm - gaussian_filter(img_norm, sigma)
    else:
        img_hp = img_norm
    return img_hp


def plot_comparison(images, titles, save_path, suptitle=None, ncols=3):
    """create comparison grid."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (img, title) in enumerate(zip(images, titles)):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        if img is not None:
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=11, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
            ax.set_title(title, fontsize=11)
        ax.axis("off")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved: {save_path.name}")


def generate_projection_images(ops_path: Path, output_dir: Path, diameter: int = 4):
    """
    generate projection comparison images from an ops.npy file.

    creates:
        01_raw_projections.png - suite2p output images
        02_anatomical_modes.png - cellpose input modes
        03_spatial_hp_filter.png - hp filter effect
        04_cellpose_final_input.png - final cellpose inputs
        05_hp_filter_zoom.png - zoomed hp detail
    """
    import lbm_suite2p_python as lsp

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"generating projection images from {ops_path}")
    ops = lsp.load_ops(ops_path)

    # extract images
    mean_img = ops.get("meanImg")
    mean_img_e = ops.get("meanImgE")
    max_proj = ops.get("max_proj")
    ref_img = ops.get("refImg")
    vcorr = ops.get("Vcorr")

    print(f"  meanImg: {mean_img.shape if mean_img is not None else None}")
    print(f"  max_proj: {max_proj.shape if max_proj is not None else None}")

    # 1. raw projections
    raw_imgs = [
        normalize99(mean_img) if mean_img is not None else None,
        normalize99(mean_img_e) if mean_img_e is not None else None,
        normalize99(max_proj) if max_proj is not None else None,
        normalize99(ref_img) if ref_img is not None else None,
        normalize99(vcorr) if vcorr is not None else None,
    ]
    raw_titles = [
        "meanImg\n(mean of registered)",
        "meanImgE\n(enhanced mean)",
        "max_proj\n(max projection)",
        "refImg\n(registration reference)",
        "Vcorr\n(activity correlation)",
    ]
    plot_comparison(raw_imgs, raw_titles, output_dir / "01_raw_projections.png",
                    "Suite2p Output Images")

    # handle cropped images
    yrange = ops.get("yrange", [0, mean_img.shape[0] if mean_img is not None else 0])
    xrange = ops.get("xrange", [0, mean_img.shape[1] if mean_img is not None else 0])

    if mean_img is not None:
        mean_img_crop = mean_img[yrange[0]:yrange[1], xrange[0]:xrange[1]]
    else:
        mean_img_crop = None

    if mean_img_e is not None:
        mean_img_e_crop = mean_img_e[yrange[0]:yrange[1], xrange[0]:xrange[1]]
    else:
        mean_img_e_crop = None

    # 2. anatomical_only modes
    mode1 = np.log(np.maximum(1e-3, max_proj / np.maximum(1e-3, mean_img_crop))) if (max_proj is not None and mean_img_crop is not None) else None
    mode2 = mean_img_crop
    mode3 = mean_img_e_crop if mean_img_e_crop is not None else mean_img_crop
    mode4 = max_proj

    anat_imgs = [
        normalize99(mode1) if mode1 is not None else None,
        normalize99(mode2) if mode2 is not None else None,
        normalize99(mode3) if mode3 is not None else None,
        normalize99(mode4) if mode4 is not None else None,
    ]
    anat_titles = [
        "anatomical_only=1\nlog(max/mean)",
        "anatomical_only=2\nmean image",
        "anatomical_only=3\nenhanced mean",
        "anatomical_only=4\nmax projection",
    ]
    plot_comparison(anat_imgs, anat_titles, output_dir / "02_anatomical_modes.png",
                    "Cellpose Input: anatomical_only Parameter")

    # 3. spatial_hp_cp effect
    base = max_proj if max_proj is not None else mean_img
    if base is not None:
        hp_imgs = [
            normalize99(base),
            apply_hp_filter(base, diameter, 0.5),
            apply_hp_filter(base, diameter, 1.0),
            apply_hp_filter(base, diameter, 2.0),
            apply_hp_filter(base, diameter, 3.0),
            apply_hp_filter(base, diameter, 5.0),
        ]
        hp_titles = [
            "No filter\n(normalized only)",
            "spatial_hp_cp=0.5\n(subtle)",
            "spatial_hp_cp=1.0\n(mild)",
            "spatial_hp_cp=2.0\n(moderate)",
            "spatial_hp_cp=3.0\n(LBM default)",
            "spatial_hp_cp=5.0\n(strong)",
        ]
        plot_comparison(hp_imgs, hp_titles, output_dir / "03_spatial_hp_filter.png",
                        f"Spatial High-Pass Filter Effect (diameter={diameter})")

    # 4. final cellpose input comparison
    final_imgs = []
    final_titles = []

    if max_proj is not None:
        final_imgs.append(normalize99(max_proj))
        final_titles.append("Suite2p default\n(max projection)")

    if mode1 is not None:
        final_imgs.append(apply_hp_filter(mode1, diameter, 3.0))
        final_titles.append("Mode 1 + hp=3\nlog(max/mean)")

    if mode3 is not None:
        final_imgs.append(apply_hp_filter(mode3, diameter, 3.0))
        final_titles.append("Mode 3 + hp=3\nenhanced mean")

    if mode4 is not None:
        final_imgs.append(apply_hp_filter(mode4, diameter, 3.0))
        final_titles.append("LBM default\nmax + hp=3")

    if mode4 is not None:
        final_imgs.append(normalize99(mode4))
        final_titles.append("Mode 4, no hp\nmax projection")

    plot_comparison(final_imgs, final_titles, output_dir / "04_cellpose_final_input.png",
                    "What Cellpose Receives: Configuration Comparison")

    # 5. zoomed hp filter detail
    if base is not None:
        h, w = base.shape
        cy, cx = h // 2, w // 2
        sz = min(h, w) // 3
        crop = lambda img: img[cy - sz:cy + sz, cx - sz:cx + sz]

        zoom_imgs = [
            crop(normalize99(base)),
            crop(apply_hp_filter(base, diameter, 1.0)),
            crop(apply_hp_filter(base, diameter, 3.0)),
        ]
        zoom_titles = [
            "No filter (zoom)",
            "hp=1.0 (zoom)",
            "hp=3.0 (zoom)",
        ]
        plot_comparison(zoom_imgs, zoom_titles, output_dir / "05_hp_filter_zoom.png",
                        "High-Pass Filter Detail (Center Crop)")

    print(f"  projection images saved to {output_dir}")


def get_default_ops():
    """default ops for generating documentation images."""
    return {
        "diameter": 4,
        "anatomical_only": 4,
        "accept_all_cells": True,
        "spatial_hp_cp": 3,
        "denoise": 1,
        "two_step_registration": 1,
    }


def organize_images(results_path: Path, docs_images_path: Path):
    """
    copy generated images from results to docs/_images/ subdirectories.

    planar outputs -> /outputs
    volume outputs -> /volume
    """
    results_path = Path(results_path)
    docs_images_path = Path(docs_images_path)

    planar_dir = docs_images_path / "outputs"
    volume_dir = docs_images_path / "volume"
    planar_dir.mkdir(parents=True, exist_ok=True)
    volume_dir.mkdir(parents=True, exist_ok=True)

    volume_patterns = [
        "all_planes_masks.png",
        "orthoslices.png",
        "roi_map_3d*.png",
        "volume_*.png",
        "mean_volume_signal.png",
        "rastermap.png",
        "volume_trace_analysis.png",
        "volume_diagnostics.png",
    ]

    planar_filenames = [
        "01_correlation.png",
        "01_correlation_segmentation.png",
        "02_max_projection.png",
        "02_max_projection_segmentation.png",
        "03_mean.png",
        "03_mean_segmentation.png",
        "04_mean_enhanced.png",
        "04_mean_enhanced_segmentation.png",
        "05_quality_diagnostics.png",
        "07a_traces_raw_20.png",
        "07b_traces_raw_50.png",
        "07c_traces_raw_100.png",
        "08a_traces_dff_20.png",
        "08b_traces_dff_50.png",
        "08c_traces_dff_100.png",
        "09_traces_rejected.png",
        "10_shot_noise_accepted.png",
        "11_shot_noise_rejected.png",
    ]

    # backwards-compatible aliases (copy new names to old names for docs)
    planar_aliases = {
        "07_traces_raw.png": "07a_traces_raw_20.png",
        "08_traces_dff.png": "08a_traces_dff_20.png",
    }

    copied_files = {"planar": [], "volume": []}

    # find plane directories
    plane_dirs = sorted(results_path.glob("**/plane*_stitched")) or sorted(results_path.glob("**/plane*"))

    if plane_dirs:
        plane_dir = plane_dirs[0]
        print(f"  using {plane_dir.name} for planar images")

        for filename in planar_filenames:
            src = plane_dir / filename
            if src.exists():
                dst = planar_dir / filename
                shutil.copy2(src, dst)
                copied_files["planar"].append(filename)
                print(f"    {filename} -> outputs/")

        # create backwards-compatible aliases from already-copied files
        for alias_name, source_name in planar_aliases.items():
            src = planar_dir / source_name  # already copied to planar_dir
            if src.exists():
                dst = planar_dir / alias_name
                shutil.copy2(src, dst)
                copied_files["planar"].append(alias_name)
                print(f"    {alias_name} (alias for {source_name})")

    # volume images
    suite2p_dirs = list(results_path.glob("**/suite2p")) or [results_path]

    for suite2p_dir in suite2p_dirs:
        for pattern in volume_patterns:
            for src in suite2p_dir.glob(pattern):
                if src.is_file():
                    dst = volume_dir / src.name
                    shutil.copy2(src, dst)
                    copied_files["volume"].append(src.name)
                    print(f"    {src.name} -> volume/")

    print(f"  organized {len(copied_files['planar'])} planar + {len(copied_files['volume'])} volume images")
    return copied_files


def run_pipeline(input_path: str, output_path: str, ops: dict = None):
    """run lsp.pipeline() and return ops files."""
    from lbm_suite2p_python import pipeline

    if ops is None:
        ops = get_default_ops()

    print(f"running pipeline on {input_path}")
    print(f"saving results to {output_path}")
    print(f"ops: {ops}")

    ops_files = pipeline(
        input_data=input_path,
        save_path=output_path,
        ops=ops,
        keep_raw=False,
        keep_reg=True,
    )

    return ops_files


DEMO_INPUT = "D:/demo/raw"
DEMO_OUTPUT = "D:/demo/results"


def main():
    parser = argparse.ArgumentParser(
        description="generate documentation images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # run with demo defaults (D:/demo/raw -> D:/demo/results)
  uv run python scripts/generate_images.py

  # run full pipeline with custom paths
  uv run python scripts/generate_images.py -i D:/data/raw -o D:/data/results

  # generate projection images only (from existing ops)
  uv run python scripts/generate_images.py --ops D:/demo/results/plane01_stitched/ops.npy --projections-only

  # organize existing images (skip pipeline)
  uv run python scripts/generate_images.py --skip-pipeline
"""
    )
    parser.add_argument("--input", "-i", type=str, default=DEMO_INPUT, help=f"input movie path (default: {DEMO_INPUT})")
    parser.add_argument("--output", "-o", type=str, default=DEMO_OUTPUT, help=f"output results path (default: {DEMO_OUTPUT})")
    parser.add_argument("--ops", type=str, help="existing ops.npy for projection images")
    parser.add_argument("--docs-images", type=str, default=None, help="docs/_images/ path")
    parser.add_argument("--skip-pipeline", action="store_true", help="skip pipeline, organize only")
    parser.add_argument("--projections-only", action="store_true", help="generate projection images only")
    parser.add_argument("--diameter", "-d", type=int, default=4, help="cell diameter (default: 4)")
    parser.add_argument("--anatomical-only", type=int, default=4, choices=[0, 1, 2, 3, 4])

    args = parser.parse_args()

    # determine docs/_images/ path
    script_dir = Path(__file__).parent
    if args.docs_images:
        docs_images_path = Path(args.docs_images)
    else:
        docs_images_path = script_dir.parent / "docs" / "_images"
    docs_images_path.mkdir(parents=True, exist_ok=True)

    # projection images only mode
    if args.projections_only:
        if not args.ops:
            # try default location
            default_ops = Path(DEMO_OUTPUT) / "plane01_stitched" / "ops.npy"
            if default_ops.exists():
                args.ops = str(default_ops)
            else:
                parser.error("--projections-only requires --ops path (or default at D:/demo/results/plane01_stitched/ops.npy)")
        ops_path = Path(args.ops)
        projections_dir = docs_images_path / "projections"
        generate_projection_images(ops_path, projections_dir, args.diameter)
        return

    output_path = Path(args.output)

    # run pipeline
    if not args.skip_pipeline:
        input_path = Path(args.input)
        if not input_path.exists():
            parser.error(f"input path does not exist: {input_path}")

        ops = get_default_ops()
        ops["diameter"] = args.diameter
        ops["anatomical_only"] = args.anatomical_only

        ops_files = run_pipeline(str(input_path), str(output_path), ops)
        print(f"\npipeline complete: {len(ops_files)} ops files")

        # generate projection images from first plane
        if ops_files:
            projections_dir = docs_images_path / "projections"
            generate_projection_images(ops_files[0], projections_dir, args.diameter)

    # organize images
    print("\norganizing images...")
    organize_images(output_path, docs_images_path)

    # if we have ops files and skipped pipeline, still try to generate projections
    if args.skip_pipeline:
        plane_dirs = sorted(output_path.glob("**/plane*_stitched")) or sorted(output_path.glob("**/plane*"))
        if plane_dirs:
            ops_file = plane_dirs[0] / "ops.npy"
            if ops_file.exists():
                projections_dir = docs_images_path / "projections"
                generate_projection_images(ops_file, projections_dir, args.diameter)

    print("\ndone!")


if __name__ == "__main__":
    main()
