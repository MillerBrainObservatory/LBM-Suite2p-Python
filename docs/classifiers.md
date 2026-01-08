(classifiers)=
# Suite2p ROI Classifiers

Suite2p uses a logistic regression classifier to label detected ROIs as cells vs non-cells. This happens after detection and extraction, using morphological features computed from each ROI's spatial footprint.

## Pipeline Position

```
Detection → Extraction → Classification → Spike Deconvolution
                              ↑
                        classifier.npy
```

The classifier runs on already-detected ROIs. It does not affect which ROIs are found, only how they're labeled in `iscell.npy`.

## Selection Priority

Suite2p checks for classifiers in this order:

1. **Custom classifier** - `ops["classifier_path"]` if set
2. **Builtin classifier** - if `ops["use_builtin_classifier"] = True`
3. **User default** - `~/.suite2p/classifiers/classifier_user.npy` if it exists
4. **Builtin fallback** - if no user default exists

You'll see one of these messages during processing:

```
NOTE: applying classifier D:\path\to\custom.npy     # custom
NOTE: Applying builtin classifier at ...            # builtin
NOTE: applying default C:\Users\...classifier_user.npy  # user default
```

## Using a Custom Classifier

Add `classifier_path` to your ops:

```python
import lbm_suite2p_python as lsp

ops = {
    "diameter": 3,
    "anatomical_only": 3,
    "classifier_path": r"D:\classifiers\my_trained_classifier.npy",
}

results = lsp.pipeline(
    input_data="D:/data/raw",
    ops=ops,
)
```

## What the Classifier Uses

The classifier uses {term}`logistic regression` to predict cell probability from three morphological features stored in `stat`:

| Feature | Description | Cells tend to have... |
|---------|-------------|----------------------|
| `skew` | {term}`skewness` of fluorescence trace | positive skew (sparse transients) |
| `compact` | spatial compactness (1 = circular), see {term}`compact` | higher values (round soma) |
| `npix_norm` | normalized pixel count, see {term}`npix_norm` | values near 1.0 (expected size) |

These features are computed during extraction. The classifier then:

1. Bins each feature into 100 quantiles
2. Converts to (log) probability ratios
3. Fits a logistic regression on those ratios to predict cell vs non-cell
4. Outputs probability 0-1, thresholded at 0.5

````{dropdown} Visualize classifier features for your data
:icon: code

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mbo_utilities import select_files
import lbm_suite2p_python as lsp

# select stat.npy file
files = select_files(
    title="Select stat.npy",
    filters=["Stat files", "*.npy"],
)
if not files:
    raise ValueError("No file selected")

stat_path = files[0]
plane_dir = stat_path.parent

# load data
stat = np.load(stat_path, allow_pickle=True)
ops = lsp.load_ops(plane_dir / "ops.npy")
iscell = np.load(plane_dir / "iscell.npy")

mean_img = ops.get("meanImg", ops.get("meanImgE"))
Ly, Lx = mean_img.shape

# get 3 accepted and 3 rejected cells sorted by classifier probability
probs = iscell[:, 1]
accepted_idx = np.where(iscell[:, 0] == 1)[0]
rejected_idx = np.where(iscell[:, 0] == 0)[0]

# pick 3 from each, spread across probability range
n_examples = 3
acc_sorted = accepted_idx[np.argsort(probs[accepted_idx])]
rej_sorted = rejected_idx[np.argsort(probs[rejected_idx])[::-1]]
examples_acc = acc_sorted[np.linspace(0, len(acc_sorted)-1, n_examples, dtype=int)]
examples_rej = rej_sorted[np.linspace(0, len(rej_sorted)-1, n_examples, dtype=int)]

# plot
fig, axes = plt.subplots(2, n_examples, figsize=(14, 10))

for row, (label, indices, color) in enumerate([
    ("Accepted", examples_acc, "lime"),
    ("Rejected", examples_rej, "red"),
]):
    for col, idx in enumerate(indices):
        ax = axes[row, col]
        s = stat[idx]

        # extract roi region with padding
        ypix, xpix = s["ypix"], s["xpix"]
        pad = 20
        y0, y1 = max(0, ypix.min() - pad), min(Ly, ypix.max() + pad + 1)
        x0, x1 = max(0, xpix.min() - pad), min(Lx, xpix.max() + pad + 1)

        roi_img = mean_img[y0:y1, x0:x1]
        ax.imshow(roi_img, cmap="gray", vmin=np.percentile(roi_img, 1),
                  vmax=np.percentile(roi_img, 99))

        # overlay mask
        local_y, local_x = ypix - y0, xpix - x0
        ax.scatter(local_x, local_y, c=color, s=8, alpha=0.7, linewidths=0)

        # show features as text overlay
        prob = probs[idx]
        skew = s.get("skew", np.nan)
        compact = s.get("compact", np.nan)
        npix_norm = s.get("npix_norm", np.nan)

        info = f"{label} p={prob:.2f}\nskew={skew:.2f}\ncompact={compact:.2f}\nnpix_norm={npix_norm:.2f}"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=11,
                verticalalignment="top", color="white", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        ax.axis("off")

plt.tight_layout()
plt.show()
```
````

## Training a New Classifier

Use the Suite2p GUI to train classifiers:

1. Run processing on representative data
2. Open results in Suite2p GUI: `suite2p.gui.run(statfile="path/to/stat.npy")`
3. Manually curate ROIs (mark cells/non-cells)
4. File → Save classifier → saves to `~/.suite2p/classifiers/classifier_user.npy`
5. Copy/rename for project-specific use

The classifier file (`.npy`) contains:

- `stats`: feature statistics from training data
- `iscell`: labels from training data
- `keys`: which features were used

## Bypassing Classification

If you want all detected ROIs regardless of classification:

```python
# accept all ROIs after classification runs
results = lsp.pipeline(
    input_data="D:/data/raw",
    accept_all_cells=True,
)
```

This doesn't disable the classifier - it runs normally, but all ROIs get marked as accepted afterward. The original suite2p classification is preserved in `iscell_suite2p.npy`.

## Common Issues

**"applying default" when you expected custom:**

Your `classifier_path` is empty or the file doesn't exist. Verify the path:

```python
from pathlib import Path
print(Path(ops["classifier_path"]).exists())  # should be True
```

**classifier trained on different data:**

Classifiers work best when trained on similar data (same indicator, magnification, cell types). A classifier trained on GCaMP6s cortical data may not transfer well to GCaMP8f cerebellar data.

**too many cells rejected:**

The builtin classifier is conservative. Options:

1. Train a custom classifier on your data
2. Use `accept_all_cells=True` and apply your own filters
3. Lower the threshold by editing `iscell.npy` probabilities

```{seealso}
- {doc}`Processing Flow <processing_flow>` - where classification fits in the pipeline
- {doc}`Postprocessing <postprocessing>` - ROI filtering functions
- [Suite2p Classifier Docs](https://suite2p.readthedocs.io/en/latest/gui.html#classifier) - GUI training instructions
```
