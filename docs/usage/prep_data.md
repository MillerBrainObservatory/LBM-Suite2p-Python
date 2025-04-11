# Data preparation


Suite2p accepts multiple input formats, but `.tiff` is the only format that has been tested thouroughly.

See suite2p's {doc}`inputs documentation <suite2p:inputs>` for more details on the use of other input formats such as `h5` and `.zarr`.

## mbo_utilities

{func}`mbo_utilities.save_as`
{func}`mbo_utilities.read_scan`
{doc}`mbo_utilities.index`


## Create `ops` from metadata

You can process a single imaging plane using the {func}`run_plane` function, which wraps `suite2p` registration, segmentation, and result plotting in a single call.

```{eval-rst}

.. autosummary:: lbm_suite2p_python.run_lsp

```

## Usage

``` {code} python3
import mbo_utilities as mbo
import lbm_suite2p_python as lsp
import suite2p

```

*Find input TIFF files*
input_files = mbo.get_files(assembled_path, str_contains='tif', max_depth=3)
metadata = mbo.get_metadata(input_files[0])

*Get and configure Suite2p ops*
ops = suite2p.default_ops()
mbo_ops = mbo.params_from_metadata(metadata, ops)

*Run a single z-plane through Suite2p*
output_ops = lsp.run_plane(
    ops=mbo_ops,
    input_file_path=input_files[0],
    save_path="output_directory"
)
