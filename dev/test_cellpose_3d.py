"""
Test script for Cellpose 3D segmentation on volumetric calcium imaging data.

This script demonstrates how to use Cellpose-SAM for 3D cell detection on
TZYX volumetric data from LBM microscopy.

References:
- Cellpose 3D docs: https://cellpose.readthedocs.io/en/latest/do3d.html
- Cellpose API: https://cellpose.readthedocs.io/en/latest/api.html
- Cellpose-SAM paper: https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1

Usage:
    python dev/test_cellpose_3d.py
"""

from pathlib import Path
import numpy as np


def load_volume_for_cellpose(
    data_path: str | Path,
    projection: str = "mean",
    normalize: bool = True,
) -> np.ndarray:
    """
    Load a TZYX volume and create a ZYX projection suitable for Cellpose 3D.

    Cellpose 3D expects input of shape (Z, Y, X) or (Z, C, Y, X).
    For calcium imaging, we typically project across time to get a static
    volume representing cell locations.

    Parameters
    ----------
    data_path : str or Path
        Path to the zarr directory containing TZYX data.
    projection : str, default "mean"
        How to project across time: "mean", "max", or "std".
    normalize : bool, default True
        Normalize each plane to [0, 1] range.

    Returns
    -------
    np.ndarray
        ZYX array suitable for Cellpose 3D segmentation.
    """
    import mbo_utilities as mbo

    print(f"Loading volume from: {data_path}")
    arr = mbo.imread(data_path)
    print(f"  Original shape (TZYX): {arr.shape}")

    # Project across time dimension
    # Lazy arrays don't have .mean() method, use np.mean() instead
    print(f"  Computing {projection} projection across time...")
    print(f"  (Loading full volume into memory...)")
    if projection == "mean":
        vol = np.mean(arr[:], axis=0)
    elif projection == "max":
        vol = np.max(arr[:], axis=0)
    elif projection == "std":
        vol = np.std(arr[:], axis=0)
    else:
        raise ValueError(f"Unknown projection: {projection}")

    print(f"  Projected shape (ZYX): {vol.shape}")

    # Normalize per-plane for better cellpose performance
    if normalize:
        for z in range(vol.shape[0]):
            plane = vol[z]
            pmin, pmax = np.percentile(plane, [1, 99])
            vol[z] = np.clip((plane - pmin) / (pmax - pmin + 1e-6), 0, 1)

    return vol.astype(np.float32)


def run_cellpose_3d(
    volume: np.ndarray,
    diameter: float = None,
    anisotropy: float = None,
    do_3D: bool = True,
    stitch_threshold: float = 0.0,
    flow3D_smooth: int = 0,
    min_size: int = 15,
    model_type: str = "cpsam",
    gpu: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Run Cellpose 3D segmentation on a ZYX volume.

    Parameters
    ----------
    volume : np.ndarray
        ZYX array to segment.
    diameter : float, optional
        Expected cell diameter in pixels. Cellpose-SAM is diameter-invariant,
        so this is optional. If None, auto-estimated.
    anisotropy : float, optional
        Z/XY sampling ratio. E.g., if Z spacing is 16um and XY is 2um,
        set anisotropy=8. If None, assumes isotropic.
    do_3D : bool, default True
        Use 2.5D segmentation (flows computed on YX, ZY, ZX slices then averaged).
        If False and stitch_threshold>0, uses 2D+stitching instead.
    stitch_threshold : float, default 0.0
        IoU threshold for stitching 2D masks across Z. Only used if do_3D=False.
    flow3D_smooth : int, default 0
        Gaussian smoothing stddev for 3D flows. Helps with fragmented cells.
    min_size : int, default 15
        Minimum mask size in pixels. Used instead of flow_threshold for 3D.
    model_type : str, default "cpsam"
        Cellpose model to use. "cpsam" is the new Cellpose-SAM model.
    gpu : bool, default True
        Use GPU acceleration.

    Returns
    -------
    masks : np.ndarray
        ZYX label array where 0=background, 1,2,...=cell labels.
    info : dict
        Dictionary with flows, styles, diameters, and parameters used.
    """
    from cellpose import models

    print(f"\nInitializing Cellpose model: {model_type}")
    print(f"  GPU: {gpu}")
    model = models.CellposeModel(
        gpu=gpu,
        pretrained_model=model_type,
    )

    print(f"\nRunning 3D segmentation...")
    print(f"  Volume shape: {volume.shape}")
    print(f"  do_3D: {do_3D}")
    print(f"  anisotropy: {anisotropy}")
    print(f"  diameter: {diameter}")
    print(f"  min_size: {min_size}")
    print(f"  flow3D_smooth: {flow3D_smooth}")
    print(f"  stitch_threshold: {stitch_threshold}")

    # Run segmentation
    masks, flows, styles, diams = model.eval(
        volume,
        diameter=diameter,
        do_3D=do_3D,
        anisotropy=anisotropy,
        stitch_threshold=stitch_threshold,
        flow3D_smooth=flow3D_smooth,
        min_size=min_size,
        z_axis=0,  # Z is first dimension
        channel_axis=None,  # No channel dimension (grayscale)
    )

    n_cells = masks.max()
    print(f"\nSegmentation complete!")
    print(f"  Found {n_cells} cells")
    print(f"  Masks shape: {masks.shape}")

    info = {
        "flows": flows,
        "styles": styles,
        "diams": diams,
        "params": {
            "diameter": diameter,
            "anisotropy": anisotropy,
            "do_3D": do_3D,
            "stitch_threshold": stitch_threshold,
            "flow3D_smooth": flow3D_smooth,
            "min_size": min_size,
            "model_type": model_type,
        },
    }

    return masks, info


def visualize_3d_masks(
    volume: np.ndarray,
    masks: np.ndarray,
    save_path: str | Path = None,
    n_slices: int = 5,
):
    """
    Visualize 3D masks overlaid on volume slices.

    Parameters
    ----------
    volume : np.ndarray
        ZYX volume used for segmentation.
    masks : np.ndarray
        ZYX label array from Cellpose.
    save_path : str or Path, optional
        Path to save figure. If None, displays interactively.
    n_slices : int, default 5
        Number of Z slices to display.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    n_z = volume.shape[0]
    slice_indices = np.linspace(0, n_z - 1, n_slices, dtype=int)

    fig, axes = plt.subplots(2, n_slices, figsize=(4 * n_slices, 8))

    # Create random colormap for masks
    n_cells = masks.max()
    np.random.seed(42)
    colors = np.random.rand(n_cells + 1, 4)
    colors[0] = [0, 0, 0, 0]  # Background transparent
    colors[1:, 3] = 0.5  # Semi-transparent masks
    cmap = ListedColormap(colors)

    for i, z in enumerate(slice_indices):
        # Top row: volume only
        axes[0, i].imshow(volume[z], cmap="gray")
        axes[0, i].set_title(f"Z={z}")
        axes[0, i].axis("off")

        # Bottom row: volume + masks overlay
        axes[1, i].imshow(volume[z], cmap="gray")
        axes[1, i].imshow(masks[z], cmap=cmap, vmin=0, vmax=n_cells)
        n_cells_slice = len(np.unique(masks[z])) - 1  # exclude background
        axes[1, i].set_title(f"Z={z} ({n_cells_slice} cells)")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Volume", fontsize=12)
    axes[1, 0].set_ylabel("+ Masks", fontsize=12)

    plt.suptitle(f"Cellpose 3D Segmentation: {n_cells} total cells", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
        plt.close()
    else:
        plt.show()


def extract_3d_stats(masks: np.ndarray, volume: np.ndarray) -> dict:
    """
    Extract statistics for each 3D cell.

    Parameters
    ----------
    masks : np.ndarray
        ZYX label array from Cellpose.
    volume : np.ndarray
        ZYX volume used for segmentation.

    Returns
    -------
    dict
        Per-cell statistics including centroid, volume, mean intensity, etc.
    """
    from scipy import ndimage

    n_cells = masks.max()
    stats = {
        "cell_id": [],
        "centroid_z": [],
        "centroid_y": [],
        "centroid_x": [],
        "volume_pixels": [],
        "mean_intensity": [],
        "max_intensity": [],
        "z_span": [],
    }

    for cell_id in range(1, n_cells + 1):
        cell_mask = masks == cell_id
        coords = np.where(cell_mask)

        if len(coords[0]) == 0:
            continue

        stats["cell_id"].append(cell_id)
        stats["centroid_z"].append(coords[0].mean())
        stats["centroid_y"].append(coords[1].mean())
        stats["centroid_x"].append(coords[2].mean())
        stats["volume_pixels"].append(cell_mask.sum())
        stats["mean_intensity"].append(volume[cell_mask].mean())
        stats["max_intensity"].append(volume[cell_mask].max())
        stats["z_span"].append(coords[0].max() - coords[0].min() + 1)

    return stats


def main():
    """Main test function."""
    import pandas as pd

    # Configuration
    data_path = Path(r"D:\example_extraction\zarr")
    output_path = Path(r"D:\example_extraction\cellpose_3d_test")
    output_path.mkdir(exist_ok=True, parents=True)

    # LBM microscopy parameters
    # Typical LBM: 2um XY resolution, ~15-16um Z spacing
    xy_resolution = 2.0  # um/pixel
    z_spacing = 16.0  # um between planes
    anisotropy = z_spacing / xy_resolution  # = 8.0

    # Expected cell diameter in XY pixels (soma ~10-15um -> 5-8 pixels at 2um/px)
    diameter = 6  # pixels (optional for cpsam)

    print("=" * 60)
    print("Cellpose 3D Segmentation Test")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print(f"XY resolution: {xy_resolution} um/pixel")
    print(f"Z spacing: {z_spacing} um")
    print(f"Anisotropy (Z/XY): {anisotropy}")

    # Check for cached projection or masks
    projection_file = output_path / "volume_projection.npy"
    masks_file = output_path / "masks_3d.npy"

    # Load or compute volume projection
    if projection_file.exists():
        print(f"\nLoading cached projection from: {projection_file}")
        volume = np.load(projection_file)
        print(f"  Volume shape (ZYX): {volume.shape}")
    else:
        volume = load_volume_for_cellpose(data_path, projection="mean")
        np.save(projection_file, volume)
        print(f"  Saved projection to: {projection_file}")

    # Load cached masks or run segmentation
    if masks_file.exists():
        print(f"\nLoading cached masks from: {masks_file}")
        masks = np.load(masks_file)
        info = np.load(output_path / "cellpose_info.npy", allow_pickle=True).item()
        print(f"  Masks shape: {masks.shape}, {masks.max()} cells")
    else:
        # Run Cellpose 3D
        masks, info = run_cellpose_3d(
            volume,
            diameter=diameter,
            anisotropy=anisotropy,
            do_3D=True,
            flow3D_smooth=1,  # Light smoothing for calcium imaging
            min_size=15,
            model_type="cpsam",
            gpu=True,
        )

        # Save results
        print("\n" + "-" * 40)
        print("Saving results...")
        print("-" * 40)
        np.save(masks_file, masks)
        np.save(output_path / "cellpose_info.npy", info, allow_pickle=True)

    # Extract and save stats
    print("\n" + "-" * 40)
    print("Extracting cell statistics...")
    print("-" * 40)
    stats = extract_3d_stats(masks, volume)
    df = pd.DataFrame(stats)
    df.to_csv(output_path / "cell_stats_3d.csv", index=False)
    print(f"  Saved {len(df)} cells to cell_stats_3d.csv")

    # Visualize
    print("\n" + "-" * 40)
    print("Generating visualization...")
    print("-" * 40)
    visualize_3d_masks(
        volume, masks,
        save_path=output_path / "cellpose_3d_visualization.png"
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total cells detected: {masks.max()}")
    print(f"Cells per plane: {[len(np.unique(masks[z])) - 1 for z in range(masks.shape[0])]}")
    if len(df) > 0:
        print(f"Mean cell volume: {df['volume_pixels'].mean():.1f} pixels")
        print(f"Mean Z span: {df['z_span'].mean():.1f} planes")
    print(f"\nResults saved to: {output_path}")

    return masks, info, stats


def compare_2d_vs_3d():
    """
    Compare 2D per-plane vs 3D volumetric segmentation.

    This helps understand when to use do_3D=True vs stitch_threshold.
    """
    from cellpose import models

    data_path = Path(r"D:\example_extraction\zarr")
    output_path = Path(r"D:\example_extraction\cellpose_comparison")
    output_path.mkdir(exist_ok=True, parents=True)

    # Check for cached comparison
    comparison_file = output_path / "segmentation_comparison.npz"
    if comparison_file.exists():
        print(f"Loading cached comparison from: {comparison_file}")
        data = np.load(comparison_file)
        return data["masks_3d"], data["masks_stitch"], data["masks_2d"]

    # Load volume (use cached if available)
    projection_file = Path(r"D:\example_extraction\cellpose_3d_test\volume_projection.npy")
    if projection_file.exists():
        print(f"Loading cached projection from: {projection_file}")
        volume = np.load(projection_file)
    else:
        volume = load_volume_for_cellpose(data_path, projection="mean")

    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    # Method 1: Full 3D (do_3D=True)
    print("\n" + "-" * 40)
    print("Method 1: Full 3D (do_3D=True)")
    print("-" * 40)
    masks_3d, _, _, _ = model.eval(
        volume,
        do_3D=True,
        anisotropy=8.0,
        min_size=15,
        z_axis=0,
    )
    print(f"  Cells found: {masks_3d.max()}")

    # Method 2: 2D + Stitching
    print("\n" + "-" * 40)
    print("Method 2: 2D + Stitching (stitch_threshold=0.5)")
    print("-" * 40)
    masks_stitch, _, _, _ = model.eval(
        volume,
        do_3D=False,
        stitch_threshold=0.5,
        min_size=15,
        z_axis=0,
    )
    print(f"  Cells found: {masks_stitch.max()}")

    # Method 3: Per-plane 2D (no stitching)
    print("\n" + "-" * 40)
    print("Method 3: Per-plane 2D (no stitching)")
    print("-" * 40)
    masks_2d = np.zeros_like(volume, dtype=np.int32)
    total_cells = 0
    for z in range(volume.shape[0]):
        m, _, _, _ = model.eval(volume[z], min_size=15)
        # Offset labels to be unique across planes
        m[m > 0] += total_cells
        masks_2d[z] = m
        total_cells = masks_2d.max()
        print(f"  Plane {z}: {len(np.unique(m)) - 1} cells")
    print(f"  Total cells: {total_cells}")

    # Save comparison
    np.savez(
        comparison_file,
        masks_3d=masks_3d,
        masks_stitch=masks_stitch,
        masks_2d=masks_2d,
        volume=volume,
    )

    print(f"\nComparison saved to: {comparison_file}")

    return masks_3d, masks_stitch, masks_2d


if __name__ == "__main__":
    masks, info, stats = main()
