(user_guide)=
# User Guide

This guide covers everything you need to process volumetric calcium imaging data with LBM-Suite2p-Python.

## Quick Start

### Prerequisites

- Python 3.12.7 to <3.12.10
- Planar timeseries TIFF files (T, Y, X format)
- Raw ScanImage TIFFs must be assembled first using [mbo_utilities](https://millerbrainobservatory.github.io/mbo_utilities/assembly.html)

### Basic Workflow

```python
import mbo_utilities as mbo
import lbm_suite2p_python as lsp
from pathlib import Path

# 1. Get planar outputs from mbo.imwrite
data_dir = Path(r"D://demo//local_nvme")

files = mbo.get_files(data_dir, "tiff", max_depth=3) # tiff
files = list(data_dir.glob("*.zarr*"))               # zarr
files = list(data_dir.glob("*data_raw.bin*"))        # suite2p binary

# 3. Process entire volume
save_dir = Path(r"D://demo//local_nvme//suite2p")
ops = {
    "two_step_registration": True # default is False
}

output_ops = lsp.run_volume(
    input_files=files,
    save_path=save_dir,
    ops=ops,
    keep_reg=True,      # Keep registered binaries (default: True)
    keep_raw=False,     # Delete raw binaries to save space (default: False)
    force_reg=False,    # Skip registration if already done (default: False)
    force_detect=False  # Skip detection if stat.npy exists (default: False)
)
```

### Processing a Single Plane

For testing parameters or processing individual planes:

```python
ops_file = lsp.run_plane(
    input_path=files[0],
    save_path=save_dir,
    ops=ops,
    chan2_file=None,     # Optional: structural channel for registration
    keep_raw=False,
    keep_reg=True,
    force_reg=False,
    force_detect=False,
    dff_window_size=300, # Rolling window for ΔF/F calculation
    dff_percentile=20    # Percentile for baseline F₀
)
```

---

## Understanding Outputs

### Per-Plane Outputs

Each plane gets its own directory with Suite2p outputs:

```
{save_path}/{plane_tag}/
├── ops.npy              # Full Suite2p parameters and results
├── stat.npy             # ROI definitions (pixel coordinates, weights, shape stats)
├── F.npy                # Fluorescence traces (n_rois, n_frames) - float32
├── Fneu.npy             # Neuropil fluorescence (n_rois, n_frames) - float32
├── spks.npy             # Deconvolved spike traces (n_rois, n_frames) - float32
├── iscell.npy           # Classification (n_rois, 2): [is_cell (0/1), probability]
├── data.bin             # Registered binary (if keep_reg=True) - int16
├── data_raw.bin         # Raw binary (if keep_raw=True) - int16
├── pc_metrics/          # Principal component analysis figures
└── *.png                # Visualization plots
```

```{figure} _images/planar_output_files.png
:name: fig-planar-files
:alt: Planar output files structure
:width: 80%

Example plane directory showing Suite2p outputs and visualization files.
```

#### Visualization Outputs

Processing generates several diagnostic plots automatically:

```{figure} _images/segmentation.png
:name: fig-segmentation
:alt: Cell segmentation results
:width: 100%

**Cell segmentation**: Detected ROI masks overlaid on mean image. Green = accepted cells, red = rejected.
```

```{figure} _images/max_projection_image.png
:name: fig-max-proj
:alt: Maximum projection
:width: 100%

**Maximum projection**: Brightest pixel values across all frames, revealing anatomical structure.
```

```{figure} _images/traces.png
:name: fig-traces
:alt: Fluorescence traces
:width: 100%

**Activity traces**: Raw fluorescence (F) and neuropil (Fneu) traces for detected cells.
```

```{figure} _images/rastermap.png
:name: fig-rastermap-plane
:alt: Rastermap clustering
:width: 100%

**Rastermap clustering**: Cells sorted by activity similarity, revealing functional structure.
```

**Conditional outputs (if `chan2_file` provided):**
```
├── data_chan2.bin       # Raw structural channel
├── data_chan2_reg.bin   # Registered structural channel
├── F_chan2.npy          # Channel 2 fluorescence
└── Fneu_chan2.npy       # Channel 2 neuropil
```

### Volumetric Outputs

After `run_volume()` completes:

```
{save_path}/
├── volume_stats.npy     # Per-plane statistics (n_cells, timing, correlation)
├── rastermap.png        # Clustered activity heatmap (if rastermap installed)
└── mean_volume_signal.png  # Signal strength vs z-plane
```

### Registration Quality Metrics

**Requires at least 1500 frames per plane**

Each plane directory contains PC-based registration quality metrics in `pc_metrics/`:

```
{plane_dir}/pc_metrics/
├── pc_metrics.csv              # Summary statistics
├── pc_metrics_panels.tif       # PC spatial patterns (Low/High)
└── pc_metrics_raw.npy          # Raw regDX array (5, 3)
```

<video width="100%" controls>
  <source src="../_images/pc_metrics.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**PC metrics visualization**: Top and bottom temporal halves for each principal component, showing spatial patterns used to assess registration quality.

**Example PC metrics table** (`pc_metrics.csv`):

```
   Rigid    Avg_NR    Max_NR
0    0.0  0.072801  0.316228   ← PC1 (strongest spatial pattern)
1    0.0  0.027177  0.141421   ← PC2
2    0.0  0.039081  0.200000   ← PC3
3    0.0  0.020034  0.141421   ← PC4
4    0.0  0.018639  0.141421   ← PC5 (weakest pattern)
```

**What this means:**

- **Rows 0-4**: First 5 Principal Components of the registered movie
- **Rigid**: Global shift between temporal halves (pixels)
- **Avg_NR**: Average non-rigid shift across blocks (pixels)
- **Max_NR**: Maximum non-rigid shift across blocks (pixels)

**Lower values = Better registration quality**

**Quality Benchmarks:**

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| **Rigid** | < 0.1 | 0.1-0.3 | 0.3-0.5 | > 0.5 |
| **Avg_NR** | < 0.1 | 0.1-0.2 | 0.2-0.4 | > 0.4 |
| **Max_NR** | < 0.5 | 0.5-1.0 | 1.0-2.0 | > 2.0 |

*All values in pixels*

**Interpreting the example above:**
- **Rigid = 0.0**: ✅ Excellent! No global shifts detected
- **Avg_NR = 0.02-0.07**: ✅ Excellent! Very small local motion
- **Max_NR = 0.14-0.32**: ✅ Good! Maximum block shifts well-controlled

**Common questions:**

- **Why is Rigid=0 for all PCs?** All PCs have negligible global shifts (< 0.001 pixels, rounded to 0.0)
- **Why do PC3 and PC4 have similar values?** Later PCs capture weaker spatial patterns and often have similar residual motion characteristics
- **Why do I see non-rigid metrics when `nonrigid=False`?** PC metrics always measure potential non-rigid motion to assess registration quality, even if non-rigid registration wasn't performed

**For detailed explanation**, see [Registration Metrics Guide](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/blob/master/docs/REGISTRATION_METRICS_EXPLAINED.md).

### Loading Results

```python
from lbm_suite2p_python import load_planar_results, dff_rolling_percentile

# Load a single plane's results
results = load_planar_results(ops_path, z_plane=0)
# Returns dict with: F, Fneu, spks, stat, iscell, cellprob, z_plane

# Calculate ΔF/F with rolling percentile baseline
dff = dff_rolling_percentile(
    results['F'],
    window_size=500,  # frames
    percentile=20     # baseline percentile
)

# Filter for accepted cells only
F_cells = results['F'][results['iscell']]
spks_cells = results['spks'][results['iscell']]
```

---

(parameters)=
## Critical Parameters

Understanding key Suite2p parameters is essential for good segmentation results.

```{admonition} Example Dataset
:class: dropdown

Example dataset collected by Will Snyder with Dr. Charles Gilbert @ Rockefeller University.

| Field        | Value                   |
|--------------|-------------------------|
| Date         | 2025-03-06              |
| Virus        | jGCaMP8s                |
| Framerate    | 17 Hz                   |
| FOV (Per ROI)| 448 µm × 896 µm         |
| Resolution   | 2 µm × 2 µm × 16 µm     |
| Num-Planes   | 14                      |
```

### Visual Parameter Comparisons

To see the effect of each parameter on segmentation results, it's helpful to start with default parameters as a baseline.

```{figure} _images/default_parameters.png
:name: fig-default-params
:alt: Default parameter segmentation results

Default parameters yield **324** accepted and **737** rejected neurons.
```

Visually it may be evident that we're missing a few obvious cells:

```{figure} _images/default_parameters_subset.png
:name: fig-default-params-subset
:alt: Zoomed view showing missed cells

Zoomed view showing several obvious cells that were not detected with default parameters.
```

There are generally 2 approaches toward curating a final dataset:

1. **Approach 1**: Tune parameters, thresholds and scaling factors to properly model your dataset
2. **Approach 2**: Use thresholds that maximize the number of cells detected, and use post-hoc correlation/spatial measures to curate cells

We will discuss both approaches.

### Detection Parameters

#### `diameter` (default: 0)
Expected cell diameter in pixels. Used by both Suite2p and Cellpose.
- **0**: Auto-estimate from data (Suite2p default)
- **6-15**: Typical range for neuronal somata at 2 µm/pixel
- **LBM override**: If `diameter` is 0/None/NaN and `anatomical_only > 0`, LBM sets it to 8

```python
ops["diameter"] = 8  # For ~16 µm cells at 2 µm/pixel resolution
```

#### `threshold_scaling` (default: 1.0)
Multiplier for detection threshold. **Lower values detect more ROIs.**
- **0.8-1.2**: Good starting range
- **<0.8**: May detect noise/background
- **>1.5**: May miss dim cells

```python
# More sensitive detection
ops["threshold_scaling"] = 0.9
```

```{figure} _images/default_params_thr.png
:name: fig-threshold-scaling
:alt: Effect of threshold_scaling on detected cells

Effect of varying `threshold_scaling` on detected cells. Notably, increasing this threshold actually led to several cells being detected that were not otherwise detected, and vice-versa.
```

#### `tau` (default: 1.0)
Calcium indicator decay time constant in seconds. **Critical for binning and deconvolution.**

GCaMP expression is slow, often taking between 100 ms to over 1 second for the signal to rise and decay. This is the timescale of the sensor, in seconds. We need this value because one of the main performance optimizations is [binning](https://en.wikipedia.org/wiki/Data_binning). We can bin our data **because of this slow timescale** - we set the bin-size to our sensor's timescale because we expect all frames in this window to be the same (on average).

- **GCaMP6f**: 0.7-1.0 s
- **GCaMP6s**: 1.5-2.0 s
- **GCaMP8s/m/f**: ~1.0 s
- **When in doubt, round up!**

Determines bin size: `bin_size = tau * fs`

```python
ops["tau"] = 1.0  # For GCaMP6f/8 at 17 Hz → ~17 frames/bin
```

```{figure} _images/default_parameters_tau.png
:name: fig-tau-effect
:alt: Effect of tau on detected cells

Effect of varying `tau` on segmentation results. Changing tau has a dramatic influence on detection.
```

```{admonition} Preview Binned Movie using Tau and Framerate
:class: dropdown

You can preview the movie as it will be binned like so:

\`\`\`python
import numpy as np
import suite2p

nframes = metadata["num_frames"]
bin_size = int(max(1, nframes // ops["nbinned"], np.round(ops["tau"] * ops["fs"])))

ops = lsp.load_ops(r"./grid_search/registration/two0/plane0/ops.npy")
bin_path = r"./grid_search/registration/two0/plane0/data.bin"

with suite2p.io.BinaryFile(filename=bin_path, Ly=ops["Ly"], Lx=ops["Lx"]) as f:
    binned_data = f.bin_movie(
        bin_size=bin_size,
        bad_frames=ops.get("badframes"),
        y_range=ops["yrange"],
        x_range=ops["xrange"],
    )
\`\`\`
```

#### `max_overlap` (default: 0.75)
Maximum allowed spatial overlap between ROIs (0-1). If two masks overlap by a fraction >max_overlap, they will be discarded/rejected.
- **0.75**: Default, reject ROIs with >75% overlap
- **1.0**: Keep all overlapping ROIs
- **0.5**: More stringent overlap rejection

```python
ops["max_overlap"] = 0.85  # Allow more overlap for dense regions
```

```{figure} _images/default_parameters_max_overlap.png
:name: fig-max-overlap
:alt: Effect of max_overlap on detected cells

Effect of varying `max_overlap` on detected cells.
```

#### `spatial_hp_detect` (default: 25)
There are several steps in the pipeline in which a `gaussian filter` is applied to the image before a downstream processing step. `spatial_hp_detect` is the gaussian filter applied immediately to each mean-subtracted image during cell detection and acts to decrease the background noise.

A good value for `spatial_hp_detect` will decrease the brightness of the background while increasing the contrast between background and neuron.

```{figure} _images/default_parameters_spatial_hp_detect.png
:name: fig-spatial-hp
:alt: Effect of spatial_hp_detect on detected cells

Effect of varying `spatial_hp_detect` on detected cells.
```

```{warning}
Physiologically relevant values for spatial high-pass filters are one of the biggest factors in quality detection.
It is additionally very easy to set this value in such a way that 0 cells will be detected.
A grid search is likely the most efficient way to test for the best spatial filter size.
```

### Registration Parameters

```{note}
The terms motion-correction and registration are often used interchangeably.
Similarly, non-rigid and piecewise-rigid are often used interchangeably.
Here, piecewise-rigid registration is the **method** used to correct for non-rigid motion.
```

We use [Suite2p Registration](https://suite2p.readthedocs.io/en/latest/registration.html) to ensure spatial alignment across all frames in a movie. This means a neuron that appears in one location in frame 0 remains in the same spot in frame N.

Suite2p first runs rigid registration (frame-wide shifts using phase correlation), followed by optional non-rigid registration (local shifts in blocks).

```{admonition} Recommended tuning
:class: tip
- Increase `nimg_init` if your template looks noisy or blurry
- If there is remaining motion after rigid, try `nonrigid = True`
- If registration looks unstable, try decreasing `maxregshift` or `maxregshiftNR`
```

```{important}
Visual inspection trumps all. Even if registration metrics look good, double check the video. And if it looks good but metrics are high, trust your eyes.
```

#### `do_registration` (default: 1)
Controls whether to run registration:
- **0 or False**: Skip registration entirely
- **1 or True**: Run only if not already done (checks for refImg/meanImg/offsets)
- **2**: Force re-registration even if already completed

```python
ops["do_registration"] = 1  # Automatic: register if needed
```

#### `nonrigid` (default: True)
Use block-wise non-rigid registration to handle local motion.
- **True**: Use non-rigid (recommended for drifting FOV)
- **False**: Use rigid registration only (faster, good for stable FOV)

```python
ops["nonrigid"] = True
ops["block_size"] = [128, 128]  # Block size for non-rigid
```

#### `align_by_chan` (default: 1)
Which channel to use for registration (1-based indexing):
- **1**: Register using functional channel (default)
- **2**: Register using structural channel (requires `chan2_file`)

```python
# Two-channel registration with structural reference
ops_file = lsp.run_plane(
    input_path=functional_file,
    chan2_file=structural_file,  # Triggers align_by_chan=2
    ops=ops
)
```

#### `maxregshift` (default: 0.1)
Maximum allowed registration shift as fraction of frame size.
- **0.1**: Allow up to 10% of frame width/height
- **0.05**: More conservative (5%)
- **0.2**: Permissive for large drifts

```python
ops["maxregshift"] = 0.15  # Allow 15% shift for large FOV drift
```

### Extraction Parameters

#### `neucoeff` (default: 0.7)
Neuropil subtraction coefficient for deconvolution.
- **0.7**: Standard value
- **0.5-1.0**: Typical range
- Higher values subtract more neuropil

```python
# Corrected fluorescence used in deconvolution:
# dF = F - neucoeff * Fneu
ops["neucoeff"] = 0.7
```

#### `allow_overlap` (default: False)
Whether ROIs can share pixels.
- **False**: Overlapping pixels removed (default)
- **True**: Pixels can belong to multiple ROIs (useful for dendrites)

```python
ops["allow_overlap"] = True  # For overlapping dendritic segments
```

#### `inner_neuropil_radius` (default: 2)
Inner radius (pixels) of neuropil annulus around each ROI.
- **2**: Standard 2-pixel buffer
- **0**: No buffer (neuropil immediately adjacent)
- **5+**: Larger buffer for very bright cells

### Anatomical Segmentation (Cellpose)

Set `anatomical_only` to use Cellpose instead of functional detection:

#### `anatomical_only` (default: 0)
Enables anatomical segmentation using [Cellpose](https://www.cellpose.org/), bypassing Suite2p's functional ROI detection. The value determines which image is used:

| Value | Image Used         | Description                                                                 |
|-------|--------------------|-----------------------------------------------------------------------------|
| `1`   | `max_proj / mean_img` | Ratio of the max projection to the mean image; highlights active areas relative to baseline. |
| `2`   | `mean_img`         | The average image over all frames; provides baseline structural contrast. |
| `3`   | `meanImgE`         | An enhanced version of the mean image using Suite2p's sharpening/filtering; highlights edges and features. |
| `4`   | `max_proj`         | Maximum projection across all frames. |
| `0` or False | Disabled   | Anatomical detection is off; functional detection (correlation-based) is used instead. |

```python
ops["anatomical_only"] = 3  # Use enhanced mean image
ops["diameter"] = 8         # Required for Cellpose
ops["sparse_mode"] = False  # Turn off sparse detection
```

#### `cellprob_threshold` (default: 0.0)
Probability threshold from Cellpose's output to determine whether a pixel belongs to a cell. More negative values include more pixels.
- **0.0**: Standard
- **-2.0**: More permissive

#### `flow_threshold` (default: 1.5)
Minimum Cellpose flow error to consider a region valid. Lower values include more ROIs.
- **1.5**: Standard
- **0.4**: More permissive

#### `spatial_hp_cp` (default: 0.0)
Amount of high-pass filtering applied to the image before Cellpose segmentation. A float between `0` and `1`.

---

## Parameter Grid Search

```{admonition} Example Dataset
:class: dropdown

Example dataset collected by Kevin Barber with Dr. Alipasha Vaziri @ Rockefeller University.

| Field        | Value                   |
|--------------|-------------------------|
| Animal       | mk301                   |
| Date         | 2025-03-01              |
| Virus        | jGCaMP8s                |
| Framerate    | 17 Hz                   |
| FOV          | 900 µm × 900 µm         |
| Resolution   | 2 µm × 2 µm × 16 µm     |
```

Test multiple parameter combinations systematically. When tuning segmentation parameters, the easiest knobs to turn are `threshold_scaling` and `max_overlap`. Lower `threshold_scaling` → more candidate ROIs. Higher `max_overlap` → more overlapping ROIs are kept. But their effects aren't linear or always intuitive, so it's often best to **grid search** them.

```python
import lbm_suite2p_python as lsp

# Define parameter grid
grid_params = {
    "threshold_scaling": [0.8, 1.0, 1.2],
    "tau": [0.7, 1.0, 1.5],
    "max_overlap": [0.7, 0.85]
}

# Run grid search on a single plane
lsp.run_grid_search(
    base_ops=ops,
    grid_search_dict=grid_params,
    input_file=test_plane_file,
    save_root=grid_search_dir,
    force_reg=False,
    force_detect=True  # Always re-run detection
)

# Output structure:
# grid_search_dir/
# ├── thr0.80_tau0.70_max0.70/
# ├── thr0.80_tau0.70_max0.85/
# ├── thr0.80_tau1.00_max0.70/
# ...
```

Each combination gets its own folder with full Suite2p outputs.

```{tip}
Some values (like `spatial_hp_cp`, `tau`, or `high_pass`) can interact in non-obvious ways.

Grid searching more than 2 parameters is really the only way to evaluate these interactions, though this can take up a lot of memory and disk space. We encourage making sure `ops['keep_movie_raw']=False` (default) and `ops['reg_tif'] = False` (default).
```

### Visualizing Grid Search Results

You can loop through the results using the saved `ops.npy` files:

```python
ops = lsp.load_ops("./grid_search/spatial/max0.75_thr1.00/plane0/ops.npy")
print("Accepted ROIs:", ops['iscell'].sum())
```

### Registration Grid Search

To evaluate what registration parameters you should use, you can try both enabling two-step registration and lowering the block-size for rigid registration.

```python
base_ops["roidetect"] = False  # Skip detection, only register

search_dict = {
    "two_step_registration": [False, True],
    "block_size": [[128, 128], [64, 64]]
}

lsp.run_grid_search(
    base_ops,
    search_dict,
    input_file=input_tiff,
    save_root=save_path.joinpath("registration")
)
```

---

## Post-Processing Filters

LBM-Suite2p-Python includes filters to refine cell selection:

```python
from lbm_suite2p_python.postprocessing import (
    filter_by_diameter,
    filter_by_area,
    filter_by_eccentricity,
    compute_event_exceptionality
)

# Load results
results = load_planar_results(ops_path)
iscell = results['iscell']
stat = results['stat']
ops = results['ops']

# Filter by size relative to median
iscell = filter_by_diameter(
    iscell, stat, ops,
    min_mult=0.3,  # Reject if <30% of median diameter
    max_mult=3.0   # Reject if >300% of median diameter
)

# Filter by area
iscell = filter_by_area(
    iscell, stat,
    min_mult=0.25,  # Reject if <25% of median area
    max_mult=4.0    # Reject if >400% of median area
)

# Filter elongated ROIs
iscell = filter_by_eccentricity(
    iscell, stat,
    max_ratio=5.0  # Reject if bounding box ratio >5:1
)

# Compute event exceptionality (for rare event detection)
fitness, erfc, sd_r, md = compute_event_exceptionality(
    results['spks'],
    N=5,  # Consecutive samples
    robust_std=False
)
```

---

## Multi-ROI Processing

LBM-Suite2p-Python automatically detects and merges ScanImage multi-ROI acquisitions.

**Automatic detection:** If "roi" appears in filename (case-insensitive), merging is triggered.

**Example filenames:**
```
plane01_roi00.tif
plane01_roi01.tif
plane02_roi00.tif
plane02_roi01.tif
```

After `run_volume()`:
```python
# ROIs are automatically merged per plane
# Output structure:
# save_path/
# ├── plane01_roi00/  # Individual ROI results
# ├── plane01_roi01/
# ├── merged_mrois/
# │   ├── plane01/    # Merged results for plane 1
# │   └── plane02/    # Merged results for plane 2
# └── volume_stats.npy  # Uses merged results
```

Images are stitched horizontally, stat arrays concatenated.

---

## ΔF/F Calculation

The gold-standard formula for measuring cellular activity is "Delta F over F₀", or the change in fluorescence intensity normalized by the baseline activity:

```{math}
:class: center large

\Delta F/F_0 = \frac{F - F_0}{F_0}
```

Here, F₀ is a user-defined baseline that may be static (e.g., median over all frames) or dynamically estimated using a rolling window.

::::{grid}
:::{grid-item-card} Key Takeaways
:columns: 12

- $\Delta F/F$ reflects intracellular calcium levels, but often in a nonlinear way
- GCaMP behaves differently due to its low baseline brightness and its nonlinearity
- There is no single recipe on how to compute $\Delta F/F$
- Computation of $\Delta F/F$ must be adapted to cell types, activity patterns, and noise

:::
::::

```{figure} _images/plot_traces_dff_50n.png
:alt: Cell Activity (ΔF/F)
:name: ug-fig-dff-activity
:width: 100%

ΔF/F traces for 50 cells, sorted by similarity using the [Rastermap](https://github.com/MouseLand/rastermap) algorithm.
```

### Baseline F₀ Strategies

Choosing the correct baseline strategy depends on many factors:

- Cell type
- Brain location
- Frame-rate
- Virus Kinetics
- Scientific question

| **Method**                         | **How It's Performed**                                                                 | **Pros**                                                      | **Cons**                                                             |
|-----------------------------------|-----------------------------------------------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------------|
| **Percentile Baseline Subtraction**| Use a moving window to define F₀ as low percentile (e.g., 10–20th) of the trace.       | Adaptive baseline; handles long-term drift.                    | Choice of window size/percentile affects sensitivity.                |
| **Z-Score Thresholding**          | Normalize trace by mean/SD, define events above N standard deviations.                | Removes baseline drift; good for noisy data.                   | Sensitive to noise if SD is low; assumes Gaussian distribution.      |
| **dF/F Thresholding**             | Compute (F − F₀)/F₀ and define a fixed or adaptive threshold for events.              | Widely used; compatible with GCaMP, Fluo dyes.                | Sensitive to F₀ definition; arbitrary thresholds can bias results.  |
| **Standard Deviation (SD) Masking**| Define active frames/regions where ΔF exceeds N×SD of baseline.                        | Objective thresholding for event detection.                    | Threshold choice heavily affects results.                            |
| **Image Subtraction (Frame-to-Frame)**| Compute ΔF = Fₙ − Fₙ₋₁ or F − background to detect sudden changes.                   | Simple, fast; used for wave detection.                         | Sensitive to noise; misses gradual changes.                          |

```{figure} _images/dff_baseline_strategies2.png
:alt: Comparison of dF/F baseline strategies
:name: ug-fig-dff-strategies
:width: 100%

Comparison of different ΔF/F₀ baseline correction methods across selected cells.
```

```{figure} _images/dff_baseline_strategies_dff.png
:alt: Resulting DFF values from baseline strategies
:name: ug-fig-dff-dff
:width: 100%

The resulting ΔF/F₀ traces can look different depending on the chosen baseline strategy.
```

```{figure} _images/dff_baseline_strategies_events.png
:alt: Resulting DFF values from baseline strategies with Events
:name: ug-fig-dff-dff-ev
:width: 100%

The resulting ΔF/F₀ traces with detected events overlaid.
```

### Standardized Noise Levels

```{math}
:class: center large

\nu = \frac{\mathrm{median}_t\left( \left| \frac{\Delta F_t}{F_0} - \frac{\Delta F_{t+1}}{F_0} \right| \right)}{\sqrt{f_r}}
```

- Compare shot-noise levels across datasets, recordings, experiments
- Takes advantage of the slow calcium signal, which is typically similar between adjacent frames
- Median to exclude outliers stemming from fast onset
- Norm by sqrt(framerate) makes metric comparable across datasets with different framerates
- Shot-noise decreases with #samples with a square root dependency

The resulting units, `% for dF/F, divided by the square root of seconds`, is strange but should be used relative to other datasets.

```{figure} _images/noise_comp.png
:alt: High vs Low Noise Levels
:name: ug-fig-noise-high-low
:width: 100%

Top N neurons with highest and lowest standardized noise levels.
```

### Pipeline Comparison

Different pipelines handle ΔF/F₀ differently:

| **Pipeline** | **F₀ Method**                       | **ΔF/F₀**                 | **Neuropil**                            |
| ------------ | ----------------------------------- | ------------------------- | ------------------------------------------------ |
| **CaImAn**   | 8th percentile, 500-frame window    | Yes, in pipeline          | Modeled via CNMF, no manual subtraction          |
| **Suite2p**  | Maximin (default) or 8th percentile | No, user divides post hoc | 0.7 × F<sub>neu</sub> subtracted before baseline |
| **EXTRACT**  | User-defined (e.g. 10th percentile) | No, user computes         | Implicitly handled via robust model              |

#### CaImAn

CaImAn computes ΔF/F₀ using a **running low-percentile baseline**. By default, it uses the **8th percentile** over a **500 frame** window. The idea is to track the lower envelope of the signal to get F₀ without being biased by transients.

**Neuropil/background:** CaImAn handles this as part of its CNMF model. Background and neuropil are explicitly separated into distinct spatial/temporal components, so the output traces are background subtracted during this factorization.

```{figure} _images/dff_baseline_strategies_caiman.png
:alt: CaImAn default DF/F strategy
:name: ug-fig-dff-caiman
:width: 100%

[CaImAn](https://github.com/flatironinstitute/CaImAn) uses a default baseline of the `lower 8th percentile` and a moving window of `200 frames`.
```

#### Suite2p

Suite2p does **not** output traces in ΔF/F₀ format directly. Instead, it gives you the raw trace and the neuropil, along with spike estimates if you ran deconvolution. The neuropil represents fluorescence from the surrounding non-somatic tissue. As an optional step, many experimenters apply a fixed subtraction:

```python
# F is an [n_neurons x time] array of raw signal
# Fneu is an [n_neurons x time] array of neuropil
F_corrected = F - 0.7 * Fneu
```

The 0.7 is an empirically chosen scalar to account for the partial contamination. Essentially you subtract 70% of the signal contained in the surrounding neuropil.

```{figure} _images/dff_oasis.png
:name: fig-dff-oasis
:width: 100%

Raw, neuropil, ΔF/F₀ and resulting deconvolved spikes as output by [Suite2p](https://github.com/MouseLand/suite2p).
```

### Using LBM-Suite2p-Python ΔF/F Functions

#### Rolling Percentile Baseline (Recommended)

```python
from lbm_suite2p_python import dff_rolling_percentile

dff = dff_rolling_percentile(
    F,
    window_size=500,    # Frames (e.g., ~30s at 17 Hz)
    percentile=20,      # Use 20th percentile as baseline
    use_median_floor=False  # Optional: set min F₀ at 1% of median
)
```

**When to use:** Most datasets, handles slow baseline drifts.

#### Median Filter Baseline

```python
from lbm_suite2p_python import dff_median_filter

dff = dff_median_filter(F)
# Uses 1% of median as F₀ (simple but less adaptive)
```

**When to use:** Quick baseline for stable recordings.

#### Shot Noise Estimation

```python
from lbm_suite2p_python import dff_shot_noise

noise_levels = dff_shot_noise(dff, fr=17.0)  # Frame rate in Hz
# Returns noise level per neuron in %/√Hz units
```

Quantifies SNR for comparing across datasets.

---

## Spike Deconvolution & Tau

Most calcium imaging pipelines model neural activity as an exponential decay process following each spike, making τ (tau) a key hyperparameter.

### Indicator Dynamics

| **GCaMP Variant**     | **Optimal Tau (s)** | **Notes / Sources** |
|-----------------------|---------------------|----------------------|
| **GCaMP6f (fast)**    | ~0.5–0.7 s          | Suite2p: ~0.7 s. OASIS/CNMF: ~0.5–0.7 s. CaImAn: ~0.4 s. |
| **GCaMP6m (medium)**  | ~1.0–1.25 s         | Suite2p: ~1.0 s. OASIS: ~1.25 s. CaImAn: ~1.0 s. |
| **GCaMP6s (slow)**    | ~1.5–2.0 s          | Suite2p: 1.25–1.5 s. OASIS/Suite2p: ~2.0 s. CaImAn: ~1.5–2.0 s. |
| **GCaMP7f (fast)**    | ~0.5 s              | Similar to GCaMP6f. |
| **GCaMP7m (medium)**  | ~1.0 s (est.)       | Estimated by analogy to GCaMP6m. |
| **GCaMP7s (slow)**    | ~1.0–1.5 s          | In vivo half-decay ~0.7 s. Tau ≈ 1.0 s. |
| **GCaMP8f (fast)**    | ~0.3 s              | Fastest decay; tenfold faster than 6f/7f. |
| **GCaMP8m (medium)**  | ~0.3 s              | Slightly slower than 8f, still ~0.3 s. |
| **GCaMP8s (slow)**    | ~0.7 s              | Faster than 6s. |

::::{grid}
:::{grid-item-card} GCaMP6 Family
:columns: 4
Fast (6f): 0.5 s
Medium (6m): 1.1 s
Slow (6s): 1.8 s
:::
:::{grid-item-card} GCaMP7 Family
:columns: 4
7f: 0.45 s
7s: 1.0 s
7c: 0.8 s
:::
:::{grid-item-card} GCaMP8 Family
:columns: 4
8f: 0.25 s
8m: 0.3 s
8s: 0.5 s
:::
::::

### Pipeline Comparison

| **Pipeline** | **Uses τ?** | **Range (s)** | **Method** |
|---------------|-------------|----------------|-------------|
| **FOOPSI** | Yes | ~1.0 | Fixed exponential |
| **OASIS / CNMF** | Yes | 0.3 – 2.0 | AR(1/2) model |
| **Suite2p** | Yes | 0.7 – 1.5 | OASIS internal |
| **CaImAn** | Yes | 0.4 – 2.0 | CNMF-E fit |
| **CASCADE** | Implicit | 0.3 – 2.0 | Learned dynamics |

::::{grid}
:::{grid-item-card} Key Takeaways
:columns: 12

- τ defines calcium transient decay and sets temporal resolution of spike inference
- Optimal τ depends on both **indicator kinetics** and **frame rate**
- Pipelines like Suite2p and CaImAn require τ tuning per GECI
- CASCADE bypasses explicit τ by learning it implicitly
- GCaMP8 series are ~3× faster than GCaMP6
:::
::::

```{note}
All τ values summarized here reflect *in vivo* mammalian calcium imaging (typically ~30 Hz frame rate).
In vitro or temperature-controlled decay times (e.g., 37 °C) can be >10× shorter.
Choosing an incorrect τ biases both spike amplitude and inferred firing rate.
```

---

## Troubleshooting

### No Cells Detected

**Possible causes:**
1. **`threshold_scaling` too high** → Lower to 0.8-0.9
2. **`tau` too small** → Increase (when in doubt, round up!)
3. **`spatial_hp_detect` too large** → Try 15-25 instead of default
4. **`diameter` wrong** → Check actual cell sizes in pixels

**Diagnosis:**
```python
# Check binned movie
import suite2p
bin_size = int(ops["tau"] * ops["fs"])
with suite2p.io.BinaryFile(filename=data_bin, Ly=ops["Ly"], Lx=ops["Lx"]) as f:
    binned = f.bin_movie(bin_size=bin_size, y_range=ops["yrange"], x_range=ops["xrange"])
# Visualize binned data to see if cells are visible
```

### Registration Artifacts

**Wobbling/warping at edges:**
- Increase `spatial_taper` (default 40 → 60-80)
- Decrease `block_size` (default [128,128] → [64,64])

**Large drift not corrected:**
- Increase `maxregshift` (default 0.1 → 0.15-0.2)
- Check `refImg` quality

### Memory Issues

**Processing large volumes:**
```python
# Process planes individually
for file in files:
    ops_file = lsp.run_plane(input_path=file, ...)
    gc.collect()  # Force garbage collection

# Or: delete binaries immediately
ops_file = lsp.run_plane(keep_raw=False, keep_reg=False, ...)
```

### Known Issues

**Docstring inconsistencies:** The `run_plane()` docstring has incorrect default values. Trust the function signature:
- `dff_window_size`: 300 (not 10)
- `dff_percentile`: 20 (not 8)
- `save_json`: False (not True)
- `keep_reg`: True (not false)
- Return type: Path (not dict)

See [CLAUDE.md](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/blob/master/CLAUDE.md#known-issues-and-source-code-discrepancies) for complete list.

---

## Advanced Usage

### Custom Ops Overrides

```python
# Start with defaults
ops = lsp.default_ops()

# Override specific parameters
ops.update({
    "threshold_scaling": 0.9,
    "tau": 1.2,
    "nonrigid": True,
    "block_size": [96, 96],
    "max_overlap": 0.85,
    "allow_overlap": True
})

# Run with custom ops
output_ops = lsp.run_volume(input_files, save_path, ops)
```

### Bypassing Binary Validation

Force recreation of binaries even if they exist:

```python
ops_file = lsp.run_plane(
    input_path=file,
    force_reg=True,  # Force binary rewrite and registration
    ops=ops
)
```

### Processing Pre-Registered Data

If you have `data.bin` already:

```python
from lbm_suite2p_python import run_plane_bin

# Directly process existing binary
ops["raw_file"] = str(plane_dir / "data_raw.bin")
ops["ops_path"] = str(plane_dir / "ops.npy")
ops["do_registration"] = 0  # Skip registration
ops["roidetect"] = True

run_plane_bin(ops)
```

---

## Best Practices

1. **Start small:** Test parameters on 1-2 representative planes before processing full volumes
2. **Use grid search:** Systematically explore parameter space
3. **Save registered binaries:** Set `keep_reg=True` for re-analysis without re-registration
4. **Delete raw binaries:** Set `keep_raw=False` to save ~50% disk space
5. **Monitor registration quality:** Check `ops["corrXY"]` and `meanImg` for artifacts
6. **Validate diameter:** Measure actual cell sizes in your max projection
7. **Document parameters:** Save `ops.json` alongside results for reproducibility

```python
from lbm_suite2p_python import ops_to_json
ops_to_json(ops_file, outpath="ops.json")
```

---

## Further Reading

- {doc}`API Reference <api>` - Function signatures and docstrings
- {doc}`Glossary <glossary>` - Term definitions
- [Suite2p Documentation](https://suite2p.readthedocs.io/) - Detailed Suite2p parameter guide
- [Cellpose Documentation](https://cellpose.readthedocs.io/) - Anatomical segmentation details
- [CLAUDE.md](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/blob/master/CLAUDE.md) - Technical deep-dive into data flow

---

## Example Workflows

### Workflow 1: Standard Volumetric Processing

```python
import mbo_utilities as mbo
import lbm_suite2p_python as lsp

# Setup
files = mbo.get_files(data_dir, "tiff", max_depth=3)
metadata = mbo.get_metadata(files[0])
ops = mbo.params_from_metadata(metadata)

# Tune parameters
ops["threshold_scaling"] = 0.9
ops["tau"] = 1.0
ops["diameter"] = 8

# Process
output_ops = lsp.run_volume(files, save_dir, ops, keep_raw=False)

# Load and analyze
results = [lsp.load_planar_results(f) for f in output_ops]
```

### Workflow 2: Two-Channel Registration

```python
# Functional and structural channels
func_files = mbo.get_files(func_dir, "tiff")
struct_files = mbo.get_files(struct_dir, "tiff")

for func, struct in zip(func_files, struct_files):
    ops_file = lsp.run_plane(
        input_path=func,
        chan2_file=struct,  # Register functional to structural
        save_path=save_dir / func.stem,
        ops=ops
    )
```

### Workflow 3: Cellpose Anatomical Segmentation

```python
ops = lsp.default_ops()
ops["anatomical_only"] = 3  # Use meanImgE
ops["diameter"] = 8
ops["sparse_mode"] = False
ops["cellprob_threshold"] = -2.0  # More permissive

output_ops = lsp.run_volume(files, save_dir, ops)
```
