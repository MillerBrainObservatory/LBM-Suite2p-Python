"""
Generate a GIF showing summary images alongside their segmentation overlays.

Creates a side-by-side animation cycling through:
- Correlation image
- Max projection
- Mean image
- Enhanced mean image

Each paired with its corresponding segmentation mask.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def make_summary_gif(
    images_dir: Path,
    output_path: Path,
    duration: int = 1000,  # ms per frame
    loop: int = 0,  # 0 = infinite loop
):
    """
    Create a GIF cycling through summary images with segmentation overlays.

    Parameters
    ----------
    images_dir : Path
        Directory containing the numbered image files
    output_path : Path
        Where to save the output GIF
    duration : int
        Milliseconds to display each frame
    loop : int
        Number of loops (0 = infinite)
    """
    # Define image pairs: (base_image, segmentation_image, label)
    image_pairs = [
        ("01_correlation.png", "01_correlation_segmentation.png", "Correlation Image"),
        ("02_max_projection.png", "02_max_projection_segmentation.png", "Max Projection Image"),
        ("03_mean.png", "03_mean_segmentation.png", "Mean Image"),
        ("04_mean_enhanced.png", "04_mean_enhanced_segmentation.png", "Enhanced Mean Image"),
    ]

    frames = []

    for base_name, seg_name, label in image_pairs:
        base_path = images_dir / base_name
        seg_path = images_dir / seg_name

        if not base_path.exists() or not seg_path.exists():
            print(f"Skipping {label}: missing files")
            continue

        # Load images
        base_img = Image.open(base_path).convert("RGB")
        seg_img = Image.open(seg_path).convert("RGB")

        # Resize to same height if needed
        if base_img.height != seg_img.height:
            target_height = min(base_img.height, seg_img.height)
            base_img = base_img.resize(
                (int(base_img.width * target_height / base_img.height), target_height),
                Image.Resampling.LANCZOS
            )
            seg_img = seg_img.resize(
                (int(seg_img.width * target_height / seg_img.height), target_height),
                Image.Resampling.LANCZOS
            )

        # Create side-by-side frame with minimal padding, no titles
        padding = 10
        total_width = base_img.width + seg_img.width + padding * 3
        total_height = base_img.height + padding * 2

        frame = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))

        # Paste images side by side
        frame.paste(base_img, (padding, padding))
        frame.paste(seg_img, (base_img.width + padding * 2, padding))

        frames.append(frame)
        print(f"Added frame: {label}")

    if not frames:
        print("No frames created!")
        return

    # Resize all frames to same size (use first frame as reference)
    target_size = frames[0].size
    frames = [f.resize(target_size, Image.Resampling.LANCZOS) if f.size != target_size else f for f in frames]

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
    print(f"\nSaved GIF to: {output_path}")
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {duration}ms per frame")
    print(f"  Size: {target_size}")


def main():
    # Paths
    repo_root = Path(__file__).parent.parent
    images_dir = repo_root / "docs" / "_images"
    output_path = repo_root / "docs" / "_images" / "segmentation_summary.gif"

    print("Creating segmentation summary GIF")
    print("=" * 50)
    print(f"Images directory: {images_dir}")
    print(f"Output: {output_path}")
    print()

    make_summary_gif(
        images_dir=images_dir,
        output_path=output_path,
        duration=1500,  # 1.5 seconds per frame
    )


if __name__ == "__main__":
    main()
