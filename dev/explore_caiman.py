#!/usr/bin/env python
"""Explore CAIMAN .mat file structure using h5py."""

import h5py
import numpy as np

def explore_hdf5_structure(file_path, max_depth=10):
    """Recursively explore HDF5 file structure."""

    def print_structure(name, obj, indent=0):
        """Print information about HDF5 objects."""
        prefix = "  " * indent

        if isinstance(obj, h5py.Dataset):
            # dataset
            dtype_str = str(obj.dtype)
            shape_str = str(obj.shape)
            print(f"{prefix}Dataset: {name}")
            print(f"{prefix}  Shape: {shape_str}")
            print(f"{prefix}  Dtype: {dtype_str}")

            # show some values for small datasets
            if obj.size < 10 and obj.size > 0:
                try:
                    data = obj[()]
                    print(f"{prefix}  Value: {data}")
                except:
                    pass

        elif isinstance(obj, h5py.Group):
            # group
            print(f"{prefix}Group: {name}")
            print(f"{prefix}  Keys: {list(obj.keys())}")

    print(f"Exploring: {file_path}\n")
    print("=" * 80)

    with h5py.File(file_path, 'r') as f:
        print(f"Top-level keys: {list(f.keys())}\n")
        print("=" * 80)
        print("\nDetailed structure:\n")

        # recursively visit all objects
        f.visititems(print_structure)

        print("\n" + "=" * 80)
        print("\nSummary of main variables:\n")

        # summarize main variables
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                print(f"{key}:")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                print(f"  Size: {obj.size} elements")

                # check if it's a reference array or actual data
                if h5py.check_string_dtype(obj.dtype):
                    print(f"  Type: String/Reference array")
                elif obj.dtype == np.dtype('O'):
                    print(f"  Type: Object reference")
                else:
                    print(f"  Type: Numeric array")

                print()

if __name__ == "__main__":
    file_path = "//rbo-s1/S1_DATA/lbm/jdemas/bi_hemisphere/output/caiman_output_plane_1.mat"
    explore_hdf5_structure(file_path)
