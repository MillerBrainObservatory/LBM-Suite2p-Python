"""
generate documentation images by running lsp.pipeline() on sample data.

usage:
    python scripts/generate_images.py --input D:/data/raw --output D:/data/results

after pipeline completes, images are organized into docs/_images/ subdirectories.
"""

import argparse
import shutil
from pathlib import Path

from lbm_suite2p_python import pipeline


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
    move generated images from results to docs/_images/ subdirectories.

    planar outputs go to /outputs
    volume outputs go to /volume
    """
    results_path = Path(results_path)
    docs_images_path = Path(docs_images_path)

    # create target directories
    planar_dir = docs_images_path / "outputs"
    volume_dir = docs_images_path / "volume"
    planar_dir.mkdir(parents=True, exist_ok=True)
    volume_dir.mkdir(parents=True, exist_ok=True)

    # planar image patterns (from individual plane directories)
    planar_patterns = [
        "**/plane*_stitched/*.png",
        "**/plane*/*.png",
    ]

    # volume image patterns (from suite2p root)
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

    # planar filenames to look for
    planar_filenames = [
        "01_reference.png",
        "02_max_projection.png",
        "03_mean_image.png",
        "04_mean_enhanced.png",
        "01_reference_segmentation.png",
        "02_max_projection_segmentation.png",
        "03_mean_image_segmentation.png",
        "04_mean_enhanced_segmentation.png",
        "05_quality_diagnostics.png",
        "06_masks.png",
        "07_traces_raw.png",
        "08_traces_dff.png",
        "09_trace_analysis.png",
        "10_rastermap.png",
        "11_noise_distribution.png",
        "12_plane_diagnostics.png",
        "13_regional_zoom.png",
        "noise_comp.png",
    ]

    copied_files = {"planar": [], "volume": []}

    # find and copy planar images (take from plane00 or first available plane)
    plane_dirs = sorted(results_path.glob("**/plane*_stitched")) or sorted(results_path.glob("**/plane*"))

    if plane_dirs:
        # use first plane as representative
        plane_dir = plane_dirs[0]
        print(f"using {plane_dir.name} for planar images")

        for filename in planar_filenames:
            src = plane_dir / filename
            if src.exists():
                dst = planar_dir / filename
                shutil.copy2(src, dst)
                copied_files["planar"].append(filename)
                print(f"  copied {filename} -> outputs/")

    # find and copy volume images from results root
    suite2p_dirs = list(results_path.glob("**/suite2p")) or [results_path]

    for suite2p_dir in suite2p_dirs:
        for pattern in volume_patterns:
            for src in suite2p_dir.glob(pattern):
                if src.is_file():
                    dst = volume_dir / src.name
                    shutil.copy2(src, dst)
                    copied_files["volume"].append(src.name)
                    print(f"  copied {src.name} -> volume/")

    # summary
    print(f"\norganized {len(copied_files['planar'])} planar images to {planar_dir}")
    print(f"organized {len(copied_files['volume'])} volume images to {volume_dir}")

    return copied_files


def run_pipeline(input_path: str, output_path: str, ops: dict = None):
    """run lsp.pipeline() and return ops files."""
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


def main():
    parser = argparse.ArgumentParser(
        description="generate documentation images from sample data"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="path to input movie data (tiff/zarr)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="path to save pipeline results"
    )
    parser.add_argument(
        "--docs-images",
        type=str,
        default=None,
        help="path to docs/_images/ (default: auto-detect from script location)"
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="skip pipeline, only organize existing images"
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=4,
        help="cell diameter in pixels (default: 4)"
    )
    parser.add_argument(
        "--anatomical-only",
        type=int,
        default=4,
        choices=[0, 1, 2, 3, 4],
        help="anatomical detection mode (default: 4)"
    )

    args = parser.parse_args()

    # determine docs/_images/ path
    if args.docs_images:
        docs_images_path = Path(args.docs_images)
    else:
        # auto-detect from script location
        script_dir = Path(__file__).parent
        docs_images_path = script_dir.parent / "docs" / "_images"

    if not docs_images_path.exists():
        docs_images_path.mkdir(parents=True, exist_ok=True)

    # build ops
    ops = get_default_ops()
    ops["diameter"] = args.diameter
    ops["anatomical_only"] = args.anatomical_only

    # run pipeline
    if not args.skip_pipeline:
        ops_files = run_pipeline(args.input, args.output, ops)
        print(f"\npipeline complete, generated {len(ops_files)} ops files")

    # organize images
    print("\norganizing images...")
    organize_images(args.output, docs_images_path)

    print("\ndone!")


if __name__ == "__main__":
    main()
