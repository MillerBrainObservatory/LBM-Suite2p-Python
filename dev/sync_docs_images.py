#!/usr/bin/env python
"""
Sync documentation images from a processed volumetric results folder.

Copies planar and volumetric output images to docs/_images for documentation.

Usage:
    uv run dev/sync_docs_images.py                    # Opens folder picker
    uv run dev/sync_docs_images.py /path/to/results   # Use specified path
"""

from pathlib import Path
import shutil
import sys


# Planar output files (found in plane subdirectories)
PLANAR_FILES = [
    "01_correlation.png",
    "01_correlation_segmentation.png",
    "02_max_projection.png",
    "02_max_projection_segmentation.png",
    "03_mean.png",
    "03_mean_segmentation.png",
    "04_mean_enhanced.png",
    "04_mean_enhanced_segmentation.png",
    "05_quality_diagnostics.png",
    "06_registration.png",
    "07_traces_raw.png",
    "08_traces_dff.png",
    "09_traces_noise.png",
    "10_noise_accepted.png",
    "11_noise_rejected.png",
    "12_rastermap.png",
]

# Volumetric output files (found at root level)
VOLUME_FILES = [
    "all_planes_masks.png",
    "orthoslices.png",
    "roi_map_3d_snr.png",
    "rastermap.png",
    "volume_trace_analysis.png",
    "volume_quality_metrics.png",
    "mean_volume_signal.png",
]


def find_plane_dir(results_path: Path) -> Path | None:
    """Find first plane directory with output files."""
    # Look for directories that look like plane folders
    for pattern in ["plane*", "*plane*", "plane_*"]:
        for d in sorted(results_path.glob(pattern)):
            if d.is_dir() and (d / "ops.npy").exists():
                return d

    # Fallback: find any directory with numbered png files
    for d in results_path.iterdir():
        if d.is_dir() and list(d.glob("0?_*.png")):
            return d

    return None


def sync_files(source_dir: Path, files: list, dest_dir: Path, label: str) -> tuple:
    """Copy files from source to destination."""
    copied = 0
    missing = []

    for filename in files:
        src = source_dir / filename
        dest = dest_dir / filename

        if src.exists():
            shutil.copy2(src, dest)
            copied += 1
            print(f"  {filename}")
        else:
            missing.append(filename)

    return copied, missing


def main():
    # Determine results path
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
    else:
        # Use mbo_utilities folder picker
        try:
            import mbo_utilities as mbo
            results_path = mbo.select_folder("Select volumetric results folder")
            if results_path is None:
                print("No folder selected.")
                return
            results_path = Path(results_path)
        except ImportError:
            print("mbo_utilities not installed. Please provide path as argument:")
            print("  uv run dev/sync_docs_images.py /path/to/results")
            return

    if not results_path.exists():
        print(f"ERROR: Path not found: {results_path}")
        return

    # Find docs/_images
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    docs_images = repo_root / "docs" / "_images"
    docs_images.mkdir(parents=True, exist_ok=True)

    print("Syncing documentation images")
    print("=" * 60)
    print(f"Source: {results_path}")
    print(f"Destination: {docs_images}")
    print()

    # Find a plane directory for planar files
    plane_dir = find_plane_dir(results_path)

    if plane_dir:
        print(f"Planar outputs (from {plane_dir.name}):")
        print("-" * 40)
        p_copied, p_missing = sync_files(plane_dir, PLANAR_FILES, docs_images, "planar")
        if p_missing:
            print(f"  Missing: {len(p_missing)} files")
    else:
        print("No plane directory found with output files.")
        p_copied, p_missing = 0, PLANAR_FILES

    # Sync volumetric files from root
    print()
    print("Volumetric outputs (from root):")
    print("-" * 40)
    v_copied, v_missing = sync_files(results_path, VOLUME_FILES, docs_images, "volume")
    if v_missing:
        print(f"  Missing: {len(v_missing)} files")

    # Regenerate GIF
    print()
    print("Regenerating segmentation GIF...")
    print("-" * 40)
    try:
        # Import from same directory
        sys.path.insert(0, str(script_dir))
        from make_summary_gif import make_summary_gif
        make_summary_gif(
            images_dir=docs_images,
            output_path=docs_images / "segmentation_summary.gif",
            duration=1500,
        )
    except Exception as e:
        print(f"  Warning: Could not regenerate GIF: {e}")

    # Summary
    print()
    print("=" * 60)
    total_copied = p_copied + v_copied
    total_missing = len(p_missing) + len(v_missing)
    print(f"Copied: {total_copied} files")
    if total_missing:
        print(f"Missing: {total_missing} files")
    print("Done!")


if __name__ == "__main__":
    main()
