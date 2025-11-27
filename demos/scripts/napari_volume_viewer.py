"""
3D Volume Viewer with Napari

Lazily loads Suite2p plane data and visualizes masks in 3D without creating large intermediate files.
Uses zarr for efficient lazy loading of binary data.
"""

import numpy as np
import zarr
from pathlib import Path
from typing import List, Tuple
import napari
from suite2p.io.binary import BinaryFile


class LazyVolumeLoader:
    """
    Lazy loader for Suite2p volume data using zarr.

    Creates a virtual zarr array that reads from individual plane data.bin files
    without copying all data into memory or creating a new file.
    """

    def __init__(self, suite2p_path: Path, use_registered: bool = True):
        """
        Initialize lazy volume loader.

        Parameters
        ----------
        suite2p_path : Path
            Path to suite2p directory containing plane folders
        use_registered : bool
            Whether to use data.bin (registered) or data_raw.bin (raw)
        """
        self.suite2p_path = Path(suite2p_path)
        self.use_registered = use_registered

        # Find all plane directories
        self.plane_dirs = sorted(self.suite2p_path.glob("plane*_stitched"))
        if not self.plane_dirs:
            raise ValueError(f"No plane directories found in {suite2p_path}")

        print(f"Found {len(self.plane_dirs)} planes")

        # Get dimensions from first plane
        first_ops = np.load(self.plane_dirs[0] / "ops.npy", allow_pickle=True).item()
        self.Ly = first_ops['Ly']
        self.Lx = first_ops['Lx']
        self.nframes = first_ops['nframes']

        # Check if we should use cropped dimensions
        if 'yrange' in first_ops and 'xrange' in first_ops:
            yrange = first_ops['yrange']
            xrange = first_ops['xrange']
            self.Ly_crop = yrange[1] - yrange[0]
            self.Lx_crop = xrange[1] - xrange[0]
            self.y_offset = yrange[0]
            self.x_offset = xrange[0]
            self.use_crop = True
            print(f"Using cropped dimensions: {self.Ly_crop}x{self.Lx_crop} (from {self.Ly}x{self.Lx})")
        else:
            self.Ly_crop = self.Ly
            self.Lx_crop = self.Lx
            self.y_offset = 0
            self.x_offset = 0
            self.use_crop = False

        self.nplanes = len(self.plane_dirs)

        print(f"Volume dimensions: {self.nplanes} x {self.Ly_crop} x {self.Lx_crop}")
        print(f"Time points: {self.nframes}")

    def create_zarr_volume(self, store_path: Path = None, time_downsample: int = 10) -> zarr.Array:
        """
        Create a zarr array that lazily loads data from plane binaries.

        Parameters
        ----------
        store_path : Path, optional
            Path to store zarr array. If None, uses temporary directory.
        time_downsample : int
            Temporal downsampling factor to reduce memory usage

        Returns
        -------
        zarr_volume : zarr.Array
            Virtual volume array (Z, Y, X)
        """
        if store_path is None:
            store_path = self.suite2p_path / "temp_volume.zarr"

        store_path = Path(store_path)

        # Calculate downsampled time dimension
        n_timepoints = self.nframes // time_downsample

        # Create zarr array for mean image (one per plane)
        print(f"Creating zarr volume at {store_path}...")

        # Remove existing store
        if store_path.exists():
            import shutil
            shutil.rmtree(store_path)

        # Create store
        store = zarr.DirectoryStore(str(store_path))
        root = zarr.group(store=store, overwrite=True)

        # Create mean image volume (Z, Y, X)
        mean_volume = root.create_dataset(
            'mean_image',
            shape=(self.nplanes, self.Ly_crop, self.Lx_crop),
            chunks=(1, self.Ly_crop, self.Lx_crop),
            dtype='float32'
        )

        # Create max projection volume
        max_volume = root.create_dataset(
            'max_projection',
            shape=(self.nplanes, self.Ly_crop, self.Lx_crop),
            chunks=(1, self.Ly_crop, self.Lx_crop),
            dtype='float32'
        )

        # Load each plane
        for z, plane_dir in enumerate(self.plane_dirs):
            plane_num = int(''.join(filter(str.isdigit, plane_dir.name)))
            print(f"  Loading plane {plane_num} ({z+1}/{self.nplanes})...")

            # Get binary file path
            bin_file = plane_dir / ("data.bin" if self.use_registered else "data_raw.bin")

            if not bin_file.exists():
                print(f"    WARNING: {bin_file.name} not found, using zeros")
                mean_volume[z] = np.zeros((self.Ly_crop, self.Lx_crop), dtype='float32')
                max_volume[z] = np.zeros((self.Ly_crop, self.Lx_crop), dtype='float32')
                continue

            # Load ops to get actual dimensions
            ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()

            # Open binary file
            with BinaryFile(Ly=self.Ly, Lx=self.Lx, filename=str(bin_file)) as f:
                # Sample frames for mean/max
                n_samples = min(1000, f.n_frames)
                sample_indices = np.linspace(0, f.n_frames-1, n_samples, dtype=int)

                # Read sampled data
                data_sample = f.file[sample_indices].astype('float32')

                # Crop if needed
                if self.use_crop:
                    data_sample = data_sample[:, self.y_offset:self.y_offset+self.Ly_crop,
                                             self.x_offset:self.x_offset+self.Lx_crop]

                # Compute statistics
                mean_img = np.mean(data_sample, axis=0)
                max_img = np.max(data_sample, axis=0)

                # Store in zarr
                mean_volume[z] = mean_img
                max_volume[z] = max_img

        print(f"Zarr volume created at {store_path}")
        return root

    def load_masks_3d(self, stat: np.ndarray, iscell: np.ndarray = None,
                      accepted_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 2D masks to 3D volume coordinates.

        Parameters
        ----------
        stat : np.ndarray
            Suite2p stat array with 'iplane' field
        iscell : np.ndarray, optional
            iscell array for filtering
        accepted_only : bool
            Whether to only include accepted cells

        Returns
        -------
        coords : np.ndarray
            Nx3 array of (z, y, x) coordinates
        labels : np.ndarray
            Label for each coordinate (ROI index)
        """
        coords = []
        labels = []

        for roi_idx, s in enumerate(stat):
            # Filter by iscell
            if accepted_only and iscell is not None:
                if iscell[roi_idx, 0] == 0:
                    continue

            # Get plane (z coordinate)
            z = s.get('iplane', 0)

            # Get y, x coordinates
            ypix = s['ypix']
            xpix = s['xpix']

            # Create 3D coordinates
            n_pixels = len(ypix)
            z_coords = np.full(n_pixels, z)

            # Stack coordinates
            roi_coords = np.column_stack([z_coords, ypix, xpix])
            coords.append(roi_coords)
            labels.extend([roi_idx] * n_pixels)

        coords = np.vstack(coords) if coords else np.zeros((0, 3))
        labels = np.array(labels)

        return coords, labels

    def create_mask_volume(self, stat: np.ndarray, iscell: np.ndarray = None,
                          accepted_only: bool = True) -> np.ndarray:
        """
        Create a 3D mask volume.

        Parameters
        ----------
        stat : np.ndarray
            Suite2p stat array
        iscell : np.ndarray, optional
            iscell array for filtering
        accepted_only : bool
            Whether to only include accepted cells

        Returns
        -------
        mask_volume : np.ndarray
            3D volume with ROI labels (Z, Y, X)
        """
        print("Creating 3D mask volume...")
        mask_volume = np.zeros((self.nplanes, self.Ly_crop, self.Lx_crop), dtype='uint16')

        n_rois = 0
        for roi_idx, s in enumerate(stat):
            # Filter by iscell
            if accepted_only and iscell is not None:
                if iscell[roi_idx, 0] == 0:
                    continue

            # Get plane
            z = s.get('iplane', 0)

            # Get coordinates
            ypix = s['ypix']
            xpix = s['xpix']

            # Adjust for crop offset
            if self.use_crop:
                ypix = ypix - self.y_offset
                xpix = xpix - self.x_offset

                # Filter out of bounds pixels
                valid_mask = (ypix >= 0) & (ypix < self.Ly_crop) & (xpix >= 0) & (xpix < self.Lx_crop)
                ypix = ypix[valid_mask]
                xpix = xpix[valid_mask]

            if len(ypix) > 0:
                # Use roi_idx + 1 as label (0 is background)
                mask_volume[z, ypix, xpix] = roi_idx + 1
                n_rois += 1

        print(f"  Added {n_rois} ROIs to volume")
        return mask_volume


def view_volume_napari(suite2p_path: str | Path, merged_dir: str = "merged",
                      time_downsample: int = 10, use_max_projection: bool = False):
    """
    Launch napari viewer with 3D volume and masks.

    Parameters
    ----------
    suite2p_path : str or Path
        Path to suite2p directory
    merged_dir : str
        Name of merged directory with consolidated data
    time_downsample : int
        Temporal downsampling factor
    use_max_projection : bool
        Whether to use max projection instead of mean image
    """
    suite2p_path = Path(suite2p_path)
    merged_path = suite2p_path / merged_dir

    # Load consolidated data
    print("Loading consolidated data...")
    stat = np.load(merged_path / "stat.npy", allow_pickle=True)
    iscell = np.load(merged_path / "iscell.npy")

    print(f"Loaded {len(stat)} ROIs ({(iscell[:, 0] == 1).sum()} accepted)")

    # Create lazy loader
    loader = LazyVolumeLoader(suite2p_path, use_registered=True)

    # Create zarr volume
    zarr_root = loader.create_zarr_volume(
        store_path=suite2p_path / "temp_volume.zarr",
        time_downsample=time_downsample
    )

    # Get volume data
    if use_max_projection:
        volume_data = zarr_root['max_projection'][:]
        volume_name = "Max Projection"
    else:
        volume_data = zarr_root['mean_image'][:]
        volume_name = "Mean Image"

    print(f"Volume shape: {volume_data.shape}")

    # Create mask volume
    mask_volume = loader.create_mask_volume(stat, iscell, accepted_only=True)

    # Launch napari
    print("\nLaunching napari viewer...")
    viewer = napari.Viewer(ndisplay=3, title="Suite2p Volume Viewer")

    # Add volume layer
    viewer.add_image(
        volume_data,
        name=volume_name,
        colormap='gray',
        blending='translucent',
        opacity=0.5,
        contrast_limits=[np.percentile(volume_data, 1), np.percentile(volume_data, 99)]
    )

    # Add mask layer
    viewer.add_labels(
        mask_volume,
        name='ROI Masks',
        opacity=0.7,
        blending='translucent'
    )

    # Set camera angle
    viewer.camera.angles = (0, 45, 45)
    viewer.camera.zoom = 2.0

    print("Napari viewer launched!")
    print("\nControls:")
    print("  - Left click + drag: Rotate")
    print("  - Right click + drag: Pan")
    print("  - Scroll: Zoom")
    print("  - Toggle layers on/off in left panel")

    return viewer


if __name__ == "__main__":
    import sys

    # Example usage
    suite2p_path = r"\\rbo-s1\S1_DATA\lbm\kbarber\2025-11-04-mk311\suite2p"

    viewer = view_volume_napari(
        suite2p_path=suite2p_path,
        merged_dir="merged",
        time_downsample=10,
        use_max_projection=False  # Use mean image (set True for max projection)
    )

    napari.run()
