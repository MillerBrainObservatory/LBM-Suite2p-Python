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

These are examples of how to use {func}`lbm_suite2p_python.run_grid_search`.

The term grid search (also called a parameter sweep) is common in machine learning, and refers to evaluating a set of parameters over a structured search space.

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

### Run the Grid Search

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

### Visualizing the Outputs

You can loop through the results using the saved `ops.npy` files, then use:

```{code-cell} ipython3
ops = lsp.load_ops("./grid_search/spatial/max0.75_thr1.00/plane0/ops.npy")
print("Accepted ROIs:", ops['iscell'].sum())
```

Use `fpl.ImageWidget`, Suite2p’s `BinaryFile`, or just `tifffile` to preview motion-corrected frames and masks.

```{tip}
Some values (like `spatial_hp_cp`, `tau`, or `high_pass`) can interact in non-obvious ways.

Grid searching more than 2 parameters is really the only way to evaluate these interactions, though this can take up a lot of memory and disk space. We encourage making sure `ops['keep_data_raw']=False` and `ops['reg_tif'] = False` (they are by default).
```

This grid search setup is extensible.
Just edit `search_dict` to sweep any combination of Suite2p ops parameters.

## Registration Grid Search

To evaluate what registration parameters you should use, you can try both enabling two-step registration and lowering the block-size for rigid registration.

Here, we recommend `ops["delete_bin"] = True` (not default), but `ops['reg_tif'] = True` to allow comparisons after registration using a `tiff` rather than a binary file which takes an extra step to load and is not as easily memory mapped.


```{code-cell} ipython3
base_ops["roidetect"] = False

search_dict = {
    "two_step_registration": [False, True],
    "block_size": [[128, 128], [64, 64]]
}

save_path = Path("./grid_search")
save_path.mkdir(exist_ok=True)

lsp.run_grid_search(
    base_ops,
    search_dict,
    input_file=input_tiff,
    save_root=save_path.joinpath("registration")
)

```

Now, we can use [fastplotlib](https://www.fastplotlib.org/user_guide/guide.html#imagewidget) to display the raw and registered movies:

`input_tiff` is the filepath to our data pre-registration. We can simply memory map it with [tifffile](https://github.com/cgohlke/tifffile/blob/master/tifffile/tifffile.py) so only frames that are being shown will be loaded.

For the binary file, we can use {func}`mbo_utilities.get_files`:

```{code-cell} ipython3
movie_paths = mbo.get_files(save_path.joinpath("registration"), 'tif')
```
### Loading a registered `tif` 

If you set `ops['reg_tiff']=True`, you will have an additional `reg_tif` folder next to your `ops.npy` suite2p outputs.

### Loading a `data.bin` or `raw_data.bin` file

``` {code} python3

nframes = metadata["num_frames"]
bin_size = int(max(1, nframes // ops["nbinned"], np.round(ops["tau"] * ops["fs"])))

ops = lsp.load_ops(r"./grid_search/registration/two0/plane0/ops.npy")
bin_path = r"./grid_search/registration/two0/plane0/data.bin"

with suite2p.io.BinaryFile(filename=bin_path, Ly=ops["Ly"], Lx=ops["Lx"]) as f:
    registered_data = f.bin_movie(
        bin_size=bin_size,
        bad_frames=ops.get("badframes"),
        y_range=ops["yrange"],
        x_range=ops["xrange"],
    )

```

You can now use `registered_data` in the widget. 


