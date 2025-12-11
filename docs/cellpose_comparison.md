# Suite2p vs Cellpose

Suite2p is a pipeline in which cellpose sits as software responsible for segmentation, just one of the steps involved in calcium imaging.

Cellpose itself is useful in non-calcium imaging paradigms and is much more actively developed. However, we may wish to incorperate some pre/post processing routines from suite2p to help sharpen our 2D image which is segmented.

Eventually, we would like to consolidate these workflows.

## Overview

| Feature | Suite2p (`anatomical_only`) | `lsp.cellpose()` |
|---------|----------------------------|------------------|
| Preprocessing | Binning, denoising, HP filtering | Simple projection |
| Image options | 4 computed images | Direct projection |
| Parameter control | Suite2p ops dict | Cellpose native |
| Output format | Suite2p (stat.npy) | Both Suite2p + Cellpose GUI |
| Speed | Slower (more steps) | Faster |

## Suite2p Cellpose Pipeline

When you use `lsp.pipeline()` with `anatomical_only > 0`, Suite2p processes your data through several steps before running Cellpose:

![Suite2p Cellpose Pipeline](_images/cellpose_suite2p_pipeline.png)

### Step 1: Temporal Binning

Suite2p bins frames together to reduce noise and computation time.

```python
# controlled by ops["nbinned"] (default: 1000)
bin_size = max(1, nframes // nbinned, tau * fs)
```

**Effect:** A 10,000 frame movie becomes ~1,000 binned frames (averaging every 10 frames).

![Temporal Binning](_images/cellpose_temporal_binning.png)

### Step 2: PCA Denoising (optional)

If `denoise=1` (default), Suite2p applies PCA-based spatial denoising.

```python
# controlled by ops["denoise"] (default: 1)
mov = pca_denoise(mov, block_size=[32, 32], n_comps_frac=0.5)
```

**Effect:** Reduces high-frequency spatial noise while preserving cell structures.

![PCA Denoising](_images/cellpose_pca_denoise.png)

### Step 3: Temporal High-Pass Filtering

If `high_pass > 0`, Suite2p applies a temporal high-pass filter to remove slow baseline drift.

```python
# controlled by ops["high_pass"] (default: 100)
mov = temporal_high_pass_filter(mov, width=high_pass)
```

**Effect:** Removes slow fluctuations, emphasizing transient activity.

| `high_pass` value | Effect |
|-------------------|--------|
| 0 | No filtering (keep raw dynamics) |
| 10-50 | Mild filtering (preserves some baseline) |
| 100 (default) | Standard filtering |
| 500+ | Aggressive filtering (only fast transients) |

![Temporal High-Pass Filter](_images/cellpose_temporal_hp.png)

### Step 4: Image Computation

Suite2p computes different images for Cellpose based on `anatomical_only` mode:

| Mode | Image | Best for |
|------|-------|----------|
| 1 | `log(max_proj / mean_img)` | Activity contrast |
| 2 | `mean_img` | Stable anatomy |
| 3 | `meanImgE` (enhanced) | Enhanced contrast |
| 4 | `max_proj` | Direct max projection |

![Anatomical Modes](_images/cellpose_anatomical_modes.png)

**Mode 1** (default) emphasizes pixels that are brighter in max vs mean - highlighting active cells.

**Mode 4** is closest to what `lsp.cellpose()` does by default.

### Step 5: Spatial High-Pass (optional)

If `spatial_hp_cp > 0`, Suite2p subtracts a Gaussian-smoothed version of the image.

```python
# controlled by ops["spatial_hp_cp"] (default: 0)
img = normalize99(img)
img -= gaussian_filter(img, diameter * spatial_hp_cp)
```

**Effect:** Removes large-scale intensity gradients, useful for uneven illumination.

![Spatial High-Pass](_images/cellpose_spatial_hp.png)

### Step 6: Cellpose Segmentation

Finally, Suite2p runs Cellpose on the processed image:

```python
masks = model.eval(img, channels=[0, 0], diameter=diameter,
                   cellprob_threshold=cellprob_threshold,
                   flow_threshold=flow_threshold)
```

## Direct Cellpose Pipeline (`lsp.cellpose()`)

`lsp.cellpose()` bypasses all Suite2p preprocessing and runs Cellpose directly:

![Direct Cellpose Pipeline](_images/cellpose_direct_pipeline.png)

### Step 1: Load Data

Load any format via `mbo.imread()`:

```python
arr = imread(input_path)  # supports TIFF, Zarr, HDF5, etc.
```

### Step 2: Temporal Projection

Compute a single projection image:

```python
# controlled by projection parameter
proj = np.max(arr, axis=0)  # 'max', 'mean', 'std', or 'percentile'
```

| Projection | Formula | Best for |
|------------|---------|----------|
| `max` | `np.max(arr, axis=0)` | Activity peaks |
| `mean` | `np.mean(arr, axis=0)` | Stable structure |
| `std` | `np.std(arr, axis=0)` | Activity variance |
| `percentile` | `np.percentile(arr, 99, axis=0)` | Robust peaks |

### Step 3: Cellpose Segmentation

Run Cellpose with full parameter control:

```python
masks, flows, styles = model.eval(
    proj,
    diameter=diameter,
    flow_threshold=flow_threshold,
    cellprob_threshold=cellprob_threshold,
    batch_size=batch_size,
    # ... all Cellpose params available
)
```

## When to Use Each

### Use Suite2p (`anatomical_only`) when:

- You want functional ROI detection with trace extraction
- Your data has significant noise that benefits from denoising
- You want automatic classification and ROI statistics
- You need to integrate with the full Suite2p workflow

### Use `lsp.cellpose()` when:

- You want fast anatomical segmentation only
- You need fine control over Cellpose parameters
- You want to compare different projections quickly
- You need Cellpose GUI-compatible output
- Your data is already preprocessed

## Parameter Comparison

### Suite2p Parameters

```python
lsp.pipeline(
    input_data,
    ops={
        "anatomical_only": 4,      # 1-4: image mode
        "nbinned": 1000,           # temporal binning
        "denoise": 1,              # PCA denoising
        "high_pass": 100,          # temporal HP filter
        "spatial_hp_cp": 0,        # spatial HP filter
        "diameter": 8,             # cell diameter (pixels)
        "cellprob_threshold": 0,   # cell probability
        "flow_threshold": 1.5,     # flow error threshold
    }
)
```

### Direct Cellpose Parameters

```python
lsp.cellpose(
    input_data,
    projection="max",              # 'max', 'mean', 'std', 'percentile'
    model_type="cyto3",            # Cellpose model
    diameter=8,                    # cell diameter (pixels)
    cellprob_threshold=0,          # cell probability
    flow_threshold=0.4,            # flow error threshold
    min_size=15,                   # minimum mask size
    batch_size=8,                  # GPU batch size
    do_3D=False,                   # 3D segmentation
)
```

## Output Comparison

### Suite2p Output

```
plane0/
├── stat.npy          # ROI statistics
├── iscell.npy        # classification
├── F.npy             # fluorescence traces
├── Fneu.npy          # neuropil traces
├── spks.npy          # deconvolved spikes
├── ops.npy           # parameters + images
└── data.bin          # registered movie
```

### `lsp.cellpose()` Output

```
cellpose/
├── masks_plane00.tif       # viewable label image
├── masks_plane00.npy       # numpy masks
├── stat_plane00.npy        # Suite2p-compatible stats
├── iscell_plane00.npy      # all accepted
├── projection_plane00.tif  # image used for segmentation
├── cellpose_seg_plane00.npy  # Cellpose GUI format
├── flows_plane00.npy       # flow fields
└── cellpose_meta.npy       # metadata
```
