---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: .lsp
  language: python
  name: python3
---

# LBM-Suite2p Quickstart

``` {note}
Example dataset collected by kevin barber with Dr. Alipasha Vaziri @rockefeller university.

Animal: mk301
Date:  2025-03-01
Virus: jGCaMP8s
Framerate: 17hz
FOV: 900um x 900um
Resolution: 2um x 2um x 16um
```

```{code-cell} ipython3
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import suite2p
import mbo_utilities as mbo
import fastplotlib as fpl
from copy import deepcopy
import lbm_suite2p_python as lsp
```

See the [assembly documentation](https://millerbrainobservatory.github.io/mbo_utilities/assembly.html) for a guide on extracting data before input into suite2p.

`````` {admonition} TL;DR
``` {code} 
scan = mbo.read_scan(r"D:\W2_DATA\kbarber\2025_03_01\mk301\green\*")
mbo.save_as(scan, "/path/to/save")
```
``````

Suite2p is primarily a 2D pipeline - we will run each z-plane sequentially and combine results at the end.

```{code-cell} ipython3
animal_path = Path(r"D:\W2_DATA\kbarber\2025_03_01\mk301")  # (optional) the parent directory for this session
assembled_path = animal_path.joinpath("assembled")          # where our assembled tiffs live
```

## Input tifs

The tifs we use as input are planar timeseries `[Txy]`. Raw ScanImage tiffs **will not work here**, as they are not in the correct frame order. 

```{code-cell} ipython3
input_files = mbo.get_files(assembled_path, str_contains='tif', max_depth=3)
input_files[:3] # show just the first 3 files
```

## Default metadata

{func}`mbo_utilities.get_metadata()` will retrieve the ScanImage metadata (frame rate, pixel resolution, image dimensions).

We then feed this metadata into {func}`mbo_utilities.params_from_metadata` to autofill the suite2p parameters that rely on these metadata.

```{code-cell} ipython3
metadata = mbo.get_metadata(input_files[0])
ops = suite2p.default_ops()
ops = mbo.params_from_metadata(metadata, ops)

# we filled in pixel resolution and frame rate
ops["dx"], ops["dy"], ops["fs"]
```

## Process single z-plane

```{code-cell} ipython3
save_path = Path("./results")
save_path.mkdir(exist_ok=True)
print(f"Saving suite2p results to:  {save_path.resolve()}")
```

```{code-cell} ipython3
input_file = Path(input_files[7]) # pick a zplane in the middle of the cavity for example
input_file
```

For this demo, we use the the default ops provided by {func}`suite2p.default_ops()`.

```{code-cell} ipython3
ops = lsp.run_plane(
    ops=ops,
    input_tiff=input_file,
    save_path=save_path,
    save_folder = str(input_file.stem),  # strip the path and extension from this filename
    replot=True
)
```

## Planar Outputs

Standard suite2p files:

- ops.npy
- stat.npy
- spks.npy
- iscell.npy
- F.npy
- Fneu.npy

Plots are created by default::

- summary images (max-projection, mean-image)
- Accepted/rejected masks drawn on summary image
- 20 randomly selected DF/F traces

```{code-cell} ipython3
list(save_path.rglob("*"))
```

### Run full volume

To run the entire volume, {func}`lbm_suite2p_python.run_volume()` takes the same inputs as its planar varient, except give a list of input files rather than a single input tiff file.

```{code-cell} ipython3
output_ops = lsp.run_volume(ops, input_files, save_path)
```

```{code-cell} ipython3
ops_files = mbo.get_files(save_path.parent, 'ops', 8)
ops_files
```

```{code-cell} ipython3
stat_files = mbo.get_files(save_path.parent, 'stat.npy', max_depth=5)
stat_files[:3]
```

```{code-cell} ipython3
output_ops = lsp.run_volume(ops, input_files, save_path=save_path, replot=True)
```
