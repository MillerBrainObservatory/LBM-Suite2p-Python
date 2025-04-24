---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
(grid_search)=
# Grid-Search

These are examples of how to use {ref}`lbm_suite2p_python.run_grid_search`.

The term {ref}`grid search`, also called a parameter sweeps, are often used in machine learning.

We run a grid search on a single z-plane.

## Setup

The workflow is similar to that used in {ref}`quickstart`.

```{code-cell} ipython3
from pathlib import Path
import suite2p
import mbo_utilities as mbo
import lbm_suite2p_python as lsp

input_tiff = r"D:/W2_DATA/kbarber/2025_03_01/mk301/assembled/plane_07_mk301.tiff"
metadata = mbo.get_metadata(input_tiff)
base_ops = mbo.params_from_metadata(metadata, suite2p.default_ops())
```

## ROI Detection

When tuning segmentation parameters, the easiest knobs to turn are `threshold_scaling` and `max_overlap`.

Lower `threshold_scaling` → more candidate ROIs.
Higher `max_overlap` → more overlapping ROIs are kept.

But their effects aren’t linear or always intuitive, so it’s often best to **grid search** them.

```{admonition} Example dataset
:name: example_dataset
:class: dropdown

| Attribute           | Value           | Description                    |
|--------------------|------------------|--------------------------------|
| Animal             | mk301            | Sample ID                      |
| Date               | 2025-03-01       | Imaging date                   |
| Plane              | plane_07         | Z-plane used in this demo      |
| FOV                | 448 × 448 px     | Field of view per plane        |
| Frame rate         | 17 Hz            |                                |
| ROIs (default)     | 324 accepted     |                                |
```


Override a few ops to use Cellpose (anatomical) detection:

```{code-cell} ipython3
base_ops["anatomical_only"] = 3
base_ops["diameter"] = 6
base_ops["flow_threshold"] = 0
base_ops["cellprob_threshold"] = -6
base_ops["max_overlap"] = 1.0
```

## Run the Grid Search

```{code-cell} ipython3
search_dict = {
    "max_overlap": [0.6, 0.75, 1.0],
    "threshold_scaling": [0.75, 1.0, 1.25]
}

save_path = Path("./grid_search")
save_path.mkdir(exist_ok=True)

lsp.run_grid_search(
    base_ops,
    search_dict,
    input_file=input_tiff,
    save_root=save_path.joinpath("spatial")
)
```

Each parameter combination will be saved to a subdirectory like:

```text
./grid_search/spatial/
├── max0.60_thr0.75/
├── max0.75_thr1.00/
├── max1.00_thr1.25/
...
```

## Visualizing the Outputs

You can loop through the results using the saved `ops.npy` files, then use:

```{code-cell} ipython3
ops = lsp.load_ops("./grid_search/spatial/max0.75_thr1.00/plane0/ops.npy")
print("Accepted ROIs:", ops['iscell'].sum())
```

Use `fpl.ImageWidget`, Suite2p’s `BinaryFile`, or just `tifffile` to preview motion-corrected frames and masks.

```{tip}
Some values (like `spatial_hp_cp`, `tau`, or `cellprob_threshold`) interact in non-obvious ways.
Grid searching more than 2 parameters is possible, but interpretation becomes harder.
```

This grid search setup is extensible. Just edit `search_dict` to sweep any combination of Suite2p ops parameters.

---

Let me know if you want a helper to auto-plot results across runs (e.g., #ROIs vs threshold).


