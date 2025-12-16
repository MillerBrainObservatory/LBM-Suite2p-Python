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


def plot_grid_with_zoom(images, titles, save_path, zoom_frac=0.3):
    """plot images in top row, zoomed regions in bottom row."""
    from matplotlib.patches import Rectangle

    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10), facecolor="black")

    for col, (img, title) in enumerate(zip(images, titles)):
        img_norm = normalize99(img)
        h, w = img.shape
        zoom_h, zoom_w = int(h * zoom_frac), int(w * zoom_frac)
        # bottom-left region
        y0, x0 = int(h * 0.55), int(w * 0.15)

        # top row: full image with box
        ax_full = axes[0, col]
        ax_full.set_facecolor("black")
        ax_full.imshow(img_norm, cmap="gray")
        ax_full.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
        ax_full.axis("off")
        rect = Rectangle((x0, y0), zoom_w, zoom_h, linewidth=2, edgecolor="cyan", facecolor="none")
        ax_full.add_patch(rect)

        # bottom row: zoomed region
        ax_zoom = axes[1, col]
        ax_zoom.set_facecolor("black")
        ax_zoom.imshow(img_norm[y0:y0+zoom_h, x0:x0+zoom_w], cmap="gray")
        ax_zoom.axis("off")
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor("cyan")
            spine.set_linewidth(2)
            spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close()
    print(f"  saved: {save_path.name}")


def generate_projection_images(ops_path: Path, output_dir: Path, diameter: int = 4):
    """
    generate projection images from an ops.npy file.

    creates:
        01_anatomical_modes.png - meanImg, meanImgE, max_proj with zoom
        02_spatial_hp_filter.png - spatial_hp_cp values 0, 0.5, 1, 3, 7, 15
    """
    import lbm_suite2p_python as lsp

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"generating projection images from {ops_path}")
    ops = lsp.load_ops(ops_path)

    # extract images
    mean_img = ops.get("meanImg")
    mean_img_e = ops.get("meanImgE")
    max_proj = ops.get("max_proj")

    print(f"  meanImg: {mean_img.shape if mean_img is not None else None}")
    print(f"  max_proj: {max_proj.shape if max_proj is not None else None}")

    # handle cropped images (max_proj uses yrange/xrange, others don't)
    yrange = ops.get("yrange", [0, mean_img.shape[0] if mean_img is not None else 0])
    xrange = ops.get("xrange", [0, mean_img.shape[1] if mean_img is not None else 0])

    # crop full-size images to match max_proj
    if mean_img is not None and max_proj is not None and mean_img.shape != max_proj.shape:
        mean_img = mean_img[yrange[0]:yrange[1], xrange[0]:xrange[1]]
    if mean_img_e is not None and max_proj is not None and mean_img_e.shape != max_proj.shape:
        mean_img_e = mean_img_e[yrange[0]:yrange[1], xrange[0]:xrange[1]]

    # 1. anatomical modes: meanImg, meanImgE, max_proj
    images = []
    titles = []
    if mean_img is not None:
        images.append(mean_img)
        titles.append("meanImg")
    if mean_img_e is not None:
        images.append(mean_img_e)
        titles.append("meanImgE")
    if max_proj is not None:
        images.append(max_proj)
        titles.append("max_proj")

    if images:
        plot_grid_with_zoom(images, titles, output_dir / "01_anatomical_modes.png")

    # 2. spatial_hp_cp comparison
    base = max_proj if max_proj is not None else mean_img
    if base is not None:
        hp_values = [0, 0.5, 1, 3, 7, 15]
        hp_images = []
        hp_titles = []
        for hp in hp_values:
            if hp == 0:
                hp_images.append(base)
            else:
                hp_images.append(apply_hp_filter(base, diameter, hp))
            hp_titles.append(f"spatial_hp_cp={hp}")

        plot_grid_with_zoom(hp_images, hp_titles, output_dir / "02_spatial_hp_filter.png")

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
