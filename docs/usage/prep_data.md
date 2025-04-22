# Data preparation

Suite2p accepts multiple input formats, but `.tiff` is the only thoroughly tested format.

See suite2p's {doc}`inputs documentation <suite2p:inputs>` for more details on the use of other input formats such as `h5` and `.zarr`.

## mbo_utilities

To prepare your raw ScanImage TIFFs for Suite2p, we will use [`mbo_utilities`](https://millerbrainobservatory.github.io/mbo_utilities/). The assembly.py These utilities read, deinterleave, and assemble the raw data into per-plane TIFFs.

Refer to the full guide: {doc}`mbo_utilities:index`

Concise usage:

```python
import mbo_utilities as mbo
scan = mbo.read_scan("/path/to/raw/*.tiff")
mbo.save_as(scan, "/path/to/assembled", ext=".tiff")
```

- Each Z-plane is saved as a separate TIFF.
- TIFFs are compatible with Suite2p.
- For details, see {doc}`mbo_utilities:assembly`

## Create `ops` from metadata

You can process a single imaging plane using the {func}`run_plane` function, which wraps `suite2p` registration, segmentation, and result plotting in a single call.

```{eval-rst}
.. autosummary:: lbm_suite2p_python.run_lsp
```

## Usage

```python
import mbo_utilities as mbo
import lbm_suite2p_python as lsp
import suite2p

# Find input TIFF files
input_files = mbo.get_files(assembled_path, str_contains='tif', max_depth=3)
metadata = mbo.get_metadata(input_files[0])

# Get and configure Suite2p ops
ops = suite2p.default_ops()
mbo_ops = mbo.params_from_metadata(metadata, ops)

# Run a single z-plane through Suite2p
output_ops = lsp.run_plane(
    ops=mbo_ops,
    input_file_path=input_files[0],
    save_path="output_directory"
)
```

