# Processing a Single Plane

You can process a single imaging plane using the {func}`run_plane` function, which wraps `suite2p` registration, segmentation, and result plotting in a single call.

```{eval-rst}

.. autofunction:: lbm_suite2p_python.run_plane

```

## Usage

``` {code} python3
import mbo_utilities as mbo
import lbm_suite2p_python as lsp
import suite2p

```

# 1. Find input TIFF files
input_files = mbo.get_files(assembled_path, str_contains='tif', max_depth=3)
metadata = mbo.get_metadata(input_files[0])

# 2. Get and configure Suite2p ops
ops = suite2p.default_ops()
mbo_ops = mbo.params_from_metadata(metadata, ops)

# 3. Run a single z-plane through Suite2p
output_ops = lsp.run_plane(
    ops=mbo_ops,
    input_file_path=input_files[0],
    save_path="output_directory"
)

### Parameters

ops: Dict of Suite2p parameters (can be created with suite2p.default_ops()).

input_file_path: Path to the TIFF file for the z-plane.

save_path: Directory to write results.

save_folder: Optional subfolder name; defaults to the TIFF stem.

replot: If True, regenerate plots even if they exist.

dryrun: If True, only show what would be done (no actual processing).

### Output

Returns a Suite2p ops dictionary with processing results.

Writes ops.npy, stat.npy, iscell.npy, plots, and animations to disk.

Can generate the following visualizations:

   Registration overlay

   Segmentation masks

   Mean/max projection images

   Fluorescence trace plots

   Animated trace video

### Notes

If results already exist for a file, it will skip processing unless replot=True.

Automatically applies metadata from TIFF headers using mbo.get_metadata() and params_from_metadata().
