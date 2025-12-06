"""
Cellpose segmentation demo on LBM data.

This script demonstrates:
1. Converting raw ScanImage TIFFs to Zarr
2. 2D segmentation on a max projection of plane 1
3. 3D segmentation on a max projection Z-stack

Results and timing logs are saved to a cellpose/ subfolder.
Re-running will skip steps that have already completed.
"""

import json
import time
from pathlib import Path

import numpy as np
import tifffile

from mbo_utilities import imread, imwrite
from cellpose import models, core, io

io.logger_setup()


def run_cellpose_demo(
    input_path: str | Path = r"D:\raw_scanimage_tiffs",
    plane_idx: int = 1,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    force: bool = False,
):
    """
    Run Cellpose 2D and 3D segmentation demo.

    Parameters
    ----------
    input_path : str or Path
        Path to raw ScanImage TIFFs directory.
    plane_idx : int
        Which z-plane to use for 2D segmentation (0-indexed).
    flow_threshold : float
        Maximum allowed error of flows for each mask.
    cellprob_threshold : float
        Probability threshold for cell detection.
    force : bool
        If True, re-run all steps even if outputs exist.
    """
    input_path = Path(input_path)
    output_dir = input_path / "cellpose"
    output_dir.mkdir(exist_ok=True)

    # Load existing timing log if present
    timing_path = output_dir / "timing_log.json"
    if timing_path.exists() and not force:
        with open(timing_path) as f:
            timing_log = json.load(f)
        print(f"Loaded existing timing log from {timing_path}")
    else:
        timing_log = {}

    # Check GPU
    use_gpu = core.use_gpu()
    print(f"GPU available: {use_gpu}")
    timing_log["gpu_available"] = use_gpu

    # Initialize model (only if we need to run segmentation)
    model = None

    def get_model():
        nonlocal model
        if model is None:
            print("Loading Cellpose model...")
            t0 = time.perf_counter()
            model = models.CellposeModel(gpu=use_gpu)
            timing_log["model_load_seconds"] = time.perf_counter() - t0
            print(f"Model loaded in {timing_log['model_load_seconds']:.2f}s")
        return model

    # Step 1: Convert raw TIFFs to Zarr
    zarr_path = output_dir / "volume"
    if zarr_path.exists() and not force:
        print(f"\nStep 1: Zarr already exists at {zarr_path}, skipping conversion...")
    else:
        print(f"\nStep 1: Converting raw TIFFs to Zarr...")
        t0 = time.perf_counter()
        data = imread(input_path)
        print(f"Raw data shape: {data.shape}")
        imwrite(data, zarr_path, ext=".zarr", overwrite=True)
        timing_log["convert_to_zarr_seconds"] = time.perf_counter() - t0
        print(f"Saved to {zarr_path} in {timing_log['convert_to_zarr_seconds']:.2f}s")

    # Step 2: Read back from Zarr
    print(f"\nStep 2: Reading Zarr...")
    t0 = time.perf_counter()
    volume = imread(zarr_path)
    timing_log["read_zarr_seconds"] = time.perf_counter() - t0
    print(f"Volume shape: {volume.shape}, read in {timing_log['read_zarr_seconds']:.2f}s")

    # Ensure we have TZYX or ZYX shape
    if volume.ndim == 4:  # TZYX
        n_frames, n_planes, ly, lx = volume.shape
    elif volume.ndim == 3:  # ZYX or TYX
        # Assume ZYX for now
        n_planes, ly, lx = volume.shape
        n_frames = 1
        volume = volume[np.newaxis, ...]
    else:
        raise ValueError(f"Unexpected volume shape: {volume.shape}")

    print(f"Volume: {n_frames} frames, {n_planes} planes, {ly}x{lx} pixels")

    # Adjust plane_idx if needed
    if plane_idx >= n_planes:
        print(f"Warning: plane_idx {plane_idx} >= n_planes {n_planes}, using plane 0")
        plane_idx = 0

    # Step 3: 2D segmentation on max projection of specified plane
    masks_2d_path = output_dir / f"masks_2d_plane{plane_idx}.tif"
    max_proj_2d_path = output_dir / f"max_proj_plane{plane_idx}.tif"

    if masks_2d_path.exists() and not force:
        print(f"\nStep 3: 2D masks already exist at {masks_2d_path}, skipping...")
        # Load existing results for summary
        masks_2d = tifffile.imread(masks_2d_path)
        timing_log["cellpose_2d_n_cells"] = int(masks_2d.max())
    else:
        print(f"\nStep 3: 2D segmentation on plane {plane_idx} max projection...")

        # Max projection over time for one plane
        t0 = time.perf_counter()
        plane_data = volume[:, plane_idx, :, :]  # (T, Y, X)
        max_proj_2d = np.max(plane_data, axis=0)  # (Y, X)
        timing_log["max_proj_2d_seconds"] = time.perf_counter() - t0
        print(f"2D max projection shape: {max_proj_2d.shape}")

        # Save max projection
        tifffile.imwrite(max_proj_2d_path, max_proj_2d)

        # Run Cellpose 2D
        print("Running Cellpose 2D...")
        t0 = time.perf_counter()
        masks_2d, flows_2d, styles_2d = get_model().eval(
            max_proj_2d,
            batch_size=32,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        timing_log["cellpose_2d_seconds"] = time.perf_counter() - t0
        timing_log["cellpose_2d_n_cells"] = int(masks_2d.max())
        print(f"2D segmentation: {timing_log['cellpose_2d_n_cells']} cells in {timing_log['cellpose_2d_seconds']:.2f}s")

        # Save 2D results
        tifffile.imwrite(masks_2d_path, masks_2d.astype(np.uint16))
        print(f"Saved 2D masks to {masks_2d_path}")

    # Step 4: 3D segmentation on max projection Z-stack
    masks_3d_path = output_dir / "masks_3d_zstack.tif"
    max_proj_3d_path = output_dir / "max_proj_zstack.tif"

    if n_planes < 2:
        print(f"\nStep 4: Skipping 3D segmentation (only {n_planes} plane)")
        timing_log["cellpose_3d_skipped"] = "single plane data"
    elif masks_3d_path.exists() and not force:
        print(f"\nStep 4: 3D masks already exist at {masks_3d_path}, skipping...")
        masks_3d = tifffile.imread(masks_3d_path)
        timing_log["cellpose_3d_n_cells"] = int(masks_3d.max())
    else:
        print("\nStep 4: 3D segmentation on max projection Z-stack...")

        # Max projection over time for all planes -> (Z, Y, X)
        t0 = time.perf_counter()
        max_proj_3d = np.max(volume, axis=0)  # (Z, Y, X)
        timing_log["max_proj_3d_seconds"] = time.perf_counter() - t0
        print(f"3D max projection shape: {max_proj_3d.shape}")

        # Save 3D max projection
        tifffile.imwrite(max_proj_3d_path, max_proj_3d)

        # Run Cellpose 3D
        # z_axis=0 tells cellpose that the first axis is Z
        print("Running Cellpose 3D...")
        t0 = time.perf_counter()
        masks_3d, flows_3d, styles_3d = get_model().eval(
            max_proj_3d,
            batch_size=32,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            do_3D=True,
            z_axis=0,
        )
        timing_log["cellpose_3d_seconds"] = time.perf_counter() - t0
        timing_log["cellpose_3d_n_cells"] = int(masks_3d.max())
        print(f"3D segmentation: {timing_log['cellpose_3d_n_cells']} cells in {timing_log['cellpose_3d_seconds']:.2f}s")

        # Save 3D results
        tifffile.imwrite(masks_3d_path, masks_3d.astype(np.uint16))
        print(f"Saved 3D masks to {masks_3d_path}")

    # Save timing log
    timing_log["input_path"] = str(input_path)
    timing_log["volume_shape"] = list(volume.shape)
    timing_log["plane_idx"] = plane_idx

    with open(timing_path, "w") as f:
        json.dump(timing_log, f, indent=2)
    print(f"\nTiming log saved to {timing_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"GPU used: {use_gpu}")
    print(f"Volume shape: {volume.shape}")
    if "cellpose_2d_n_cells" in timing_log:
        time_str = f" in {timing_log.get('cellpose_2d_seconds', 'N/A'):.2f}s" if "cellpose_2d_seconds" in timing_log else " (cached)"
        print(f"2D segmentation (plane {plane_idx}): {timing_log['cellpose_2d_n_cells']} cells{time_str}")
    if "cellpose_3d_n_cells" in timing_log:
        time_str = f" in {timing_log.get('cellpose_3d_seconds', 'N/A'):.2f}s" if "cellpose_3d_seconds" in timing_log else " (cached)"
        print(f"3D segmentation (z-stack): {timing_log['cellpose_3d_n_cells']} cells{time_str}")
    elif "cellpose_3d_skipped" in timing_log:
        print(f"3D segmentation: skipped ({timing_log['cellpose_3d_skipped']})")
    print(f"Results saved to: {output_dir}")

    return timing_log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Cellpose segmentation demo")
    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        default=r"D:\raw_scanimage_tiffs",
        help="Path to raw ScanImage TIFFs directory",
    )
    parser.add_argument(
        "--plane",
        type=int,
        default=1,
        help="Z-plane index for 2D segmentation (0-indexed)",
    )
    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4,
        help="Flow threshold for Cellpose",
    )
    parser.add_argument(
        "--cellprob-threshold",
        type=float,
        default=0.0,
        help="Cell probability threshold for Cellpose",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run all steps even if outputs exist",
    )

    args = parser.parse_args()

    run_cellpose_demo(
        input_path=args.input_path,
        plane_idx=args.plane,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        force=args.force,
    )
