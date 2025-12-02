(user_guide)=
# User Guide

```{toctree}
:maxdepth: 2
```

This guide covers everything you need to process volumetric calcium imaging data with LBM-Suite2p-Python.

```{tip}
For interactive examples, see the [Quickstart Notebook](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/blob/master/demos/notebooks/quickstart.ipynb) and [Grid Search Notebook](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/blob/master/demos/notebooks/grid_search.ipynb).
```

## Input formats

The pipeline accepts all filetypes at [mbo_utilities imread()](https://millerbrainobservatory.github.io/mbo_utilities/array_types.html) accepts.

```bash
uv run mbo formats

Supported input formats:
  .tif, .tiff  - TIFF files (BigTIFF, OME-TIFF, ScanImage)
  .zarr        - Zarr v3 arrays
  .bin         - Suite2p binary format (with ops.npy)
  .h5, .hdf5   - HDF5 files
  .npy         - NumPy arrays
  .json        - Zarr array metadata (loads parent .zarr)

Supported output formats:
  .tiff        - Multi-page BigTIFF
  .zarr        - Zarr v3 with optional OME-NGFF metadata
  .bin         - Suite2p binary format
  .h5          - HDF5 format
  .npy         - NumPy array (with .json metadata)

```

## Unified Pipeline

The `lsp.pipeline()` function is the recommended entry point for all processing. It automatically handles:

- **Any input type**: Files, directories, or pre-loaded arrays from `mbo_utilities`
- **Multi-plane data**: Processes each z-plane independently
- **Multi-ROI data**: Stitches or splits ScanImage multi-ROI acquisitions
- **Metadata extraction**: Auto-populates frame rate, pixel resolution, etc.

```python
import lbm_suite2p_python as lsp

# Process a directory of raw ScanImage TIFFs
results = lsp.pipeline(
    input_data="D:/data/raw_tiffs",
    save_path="D:/results",
)

# Process specific planes from a volume
results = lsp.pipeline(
    input_data="D:/data/volume.zarr",
    save_path="D:/results",
    planes=[1, 5, 10],  # 1-indexed
)

# Process a pre-loaded array (e.g., from mbo_utilities GUI)
import mbo_utilities as mbo
arr = mbo.imread("D:/data/raw")
results = lsp.pipeline(
    input_data=arr,
    save_path="D:/results",
    roi=0,  # Split all ROIs into separate outputs
)

# Custom Suite2p parameters
results = lsp.pipeline(
    input_data="D:/data",
    ops={"diameter": 8, "threshold_scaling": 0.8},
)
```

### Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | path, list, array | required | File, directory, list of files, or lazy array |
| `save_path` | path | None | Output directory (auto-detected if None) |
| `ops` | dict | None | Suite2p parameters (uses defaults if None) |
| `planes` | int, list | None | Which planes to process (1-indexed, None=all) |
| `roi` | int | None | ROI handling: None=stitch, 0=split, N=specific ROI |
| `keep_reg` | bool | True | Keep registered binary after processing |
| `keep_raw` | bool | False | Keep raw binary after processing |
| `force_reg` | bool | False | Force re-registration |
| `force_detect` | bool | False | Force ROI detection |
| `dff_window_size` | int | None | Window for ΔF/F baseline (auto-calculated from tau and framerate) |
| `dff_percentile` | int | 20 | Percentile for baseline F₀ |
| `dff_smooth_window` | int | None | Temporal smoothing for dF/F traces (auto-calculated) |
| `save_json` | bool | False | Save ops as JSON in addition to .npy |

## Planar Pipeline

For testing parameters or processing individual planes:

```python
results = lsp.pipeline(
    input_data=files[0],        # path to .zarr, .tiff, or .bin file
    save_path=None,             # default: save next to input file
    ops=None,                   # default: use MBO-optimized parameters
    planes=1,                   # process single plane (1-indexed)
    roi=None,                   # default: stitch multi-ROI data
    keep_reg=True,              # default: keep data.bin (registered binary)
    keep_raw=False,             # default: delete data_raw.bin after processing
    force_reg=False,            # default: skip if already registered
    force_detect=False,         # default: skip if stat.npy exists
    dff_window_size=None,       # default: auto-calculate from tau and framerate
    dff_percentile=20,          # default: 20th percentile for baseline
    dff_smooth_window=None,     # default: auto-calculate from tau and framerate
    save_json=False,            # default: only save ops.npy
)
```

### Planar Outputs

Each z-plane directory contains:

#### Data Files

| File | Shape | Description |
|------|-------|-------------|
| `ops.npy` | dict | Processing parameters and metadata |
| `stat.npy` | (n_rois,) | ROI definitions (pixel coordinates, weights, shape stats) |
| `F.npy` | (n_rois, n_frames) | Raw fluorescence traces |
| `Fneu.npy` | (n_rois, n_frames) | Neuropil fluorescence traces |
| `spks.npy` | (n_rois, n_frames) | Deconvolved spike estimates |
| `iscell.npy` | (n_rois, 2) | Cell classification: `[:, 0]` = is_cell (0/1), `[:, 1]` = probability |
| `data.bin` | (n_frames, Ly, Lx) | Registered movie (if `keep_reg=True`) |
| `data_raw.bin` | (n_frames, Ly, Lx) | Raw movie (if `keep_raw=True`) |

#### Visualization Files

Files are numbered to ensure proper ordering when viewing in file browsers.

| File | Description |
|------|-------------|
| `01_correlation.png` | Pixel-wise correlation image |
| `01_correlation_segmentation.png` | Correlation image with ROI overlay |
| `02_max_projection.png` | Maximum intensity projection |
| `02_max_projection_segmentation.png` | Max projection with ROI overlay |
| `03_mean.png` | Temporal mean image |
| `03_mean_segmentation.png` | Mean image with ROI overlay |
| `04_mean_enhanced.png` | Enhanced mean image (edge sharpened) |
| `04_mean_enhanced_segmentation.png` | Enhanced mean with ROI overlay |
| `05_quality_diagnostics.png` | ROI size, SNR, compactness metrics |
| `06_registration.png` | Registration quality visualization |
| `07_traces_raw.png` | Sample raw fluorescence traces |
| `08_traces_dff.png` | Sample ΔF/F traces |
| `09_traces_noise.png` | Noise estimation traces (rejected ROIs) |
| `10_noise_accepted.png` | Shot noise histogram (accepted) |
| `11_noise_rejected.png` | Shot noise histogram (rejected) |
| `12_rastermap.png` | Activity sorted by similarity |
| `pc_metrics.csv` | Registration quality metrics |
| `pc_metrics_panels.tif` | PC metric visualization panels |

#### Reference Images

The pipeline generates several reference images that serve as both quality checks and visualization aids:

```{figure} _images/02_max_projection.png
:alt: Maximum Intensity Projection
:name: ug-fig-max-proj
:width: 50%

**Maximum intensity projection** across all frames. Highlights the brightest pixels over time, useful for identifying highly active regions and checking for motion artifacts.
```

```{figure} _images/04_mean_enhanced.png
:alt: Enhanced Mean Image
:name: ug-fig-mean-enhanced
:width: 50%

**Enhanced mean image** (meanImgE) with edge sharpening applied. This is the recommended image for Cellpose anatomical detection (`anatomical_only=3`). Cell boundaries are more defined than the standard mean image.
```

#### Segmentation Overlays

Each reference image has a corresponding segmentation overlay showing detected ROIs:

```{figure} _images/04_mean_enhanced_segmentation.png
:alt: Segmentation Overlay
:name: ug-fig-segmentation
:width: 50%

**Segmentation overlay** on the enhanced mean image. Accepted ROIs are shown as colored outlines. Use this to visually verify that detected cells match actual cell bodies.
```

#### Quality Diagnostics

The quality diagnostics panel provides a multi-metric view of ROI quality:

```{figure} _images/05_quality_diagnostics.png
:alt: Quality Diagnostics
:name: ug-fig-quality-diag
:width: 100%

**ROI quality diagnostics** showing:
- **Size distribution**: Histogram of ROI areas (pixels) for accepted vs rejected cells
- **SNR distribution**: Signal-to-noise ratio for each ROI
- **Compactness**: How circular each ROI is (1.0 = perfect circle)
- **Skewness**: Positive skew indicates calcium transients

Use these metrics to tune parameters like `max_overlap`, `threshold_scaling`, and post-hoc filters.
```

#### Fluorescence Traces

```{figure} _images/07_traces_raw.png
:alt: Raw Fluorescence Traces
:name: ug-fig-traces-raw
:width: 100%

**Raw fluorescence traces** for the top 20 neurons sorted by quality score. The y-axis shows fluorescence intensity (arbitrary units), x-axis shows time in seconds.
```

```{figure} _images/08_traces_dff.png
:alt: ΔF/F Traces
:name: ug-fig-traces-dff
:width: 100%

**ΔF/F traces** computed using rolling percentile baseline. These normalized traces are directly comparable across cells and experiments. The quality-sorted display surfaces the most reliable signals first.
```

(understanding-trace-quality-sorting)=
#### Trace Quality Sorting

The trace plots (`07_traces_raw.png`, `08_traces_dff.png`) display the **top 20 neurons sorted by quality score**.
This weighted score combines three metrics to surface the best traces:

| Metric | Weight | Description |
|--------|--------|-------------|
| **SNR** | 1.0 | Signal-to-noise ratio (higher = better) |
| **Skewness** | 0.8 | Positive skew indicates calcium transients |
| **Shot Noise** | 0.5 | Frame-to-frame variability (lower = better) |

**Using Quality Sorting in Your Analysis:**

```python
import lbm_suite2p_python as lsp

# Load planar results
results = lsp.load_planar_results("path/to/plane01")
F = results["F"]
Fneu = results["Fneu"]
stat = results["stat"]
iscell = results["iscell"]

# Filter for accepted cells
iscell_mask = iscell[:, 0].astype(bool)
F_acc = F[iscell_mask]
Fneu_acc = Fneu[iscell_mask]
stat_acc = [s for s, m in zip(stat, iscell_mask) if m]

# Compute quality scores
quality = lsp.compute_trace_quality_score(
    F_acc, Fneu_acc, stat_acc,
    fs=30.0,  # Your frame rate
    weights={'snr': 1.0, 'skewness': 0.8, 'shot_noise': 0.5}  # Default weights
)

# Sort traces by quality (best first)
sort_idx = quality['sort_idx']
F_sorted = F_acc[sort_idx]

# View individual metrics
print(f"SNR range: {quality['snr'].min():.2f} - {quality['snr'].max():.2f}")
print(f"Best neuron score: {quality['score'][sort_idx[0]]:.2f}")

# Plot top 50 neurons
lsp.plot_traces(
    F_sorted,
    num_neurons=50,
    fps=30.0,
    scale_bar_label="Raw F",
    title="Top 50 Neurons by Quality Score"
)
```

**Custom Weights:**

Adjust weights based on your priorities:

```python
# Prioritize low noise over skewness
quality = lsp.compute_trace_quality_score(
    F, Fneu, stat, fs=30.0,
    weights={'snr': 1.0, 'skewness': 0.3, 'shot_noise': 1.0}
)
```

(understanding-δff-traces)=
#### Activity ΔF/F Traces

The ΔF/F (delta F over F) traces in `08_traces_dff.png` show the change in fluorescence normalized by baseline. This is the standard metric for comparing neural activity across cells and experiments.

**Window Size Selection for ΔF/F:**

The `dff_window_size` parameter controls the rolling window used to estimate baseline F₀. Choosing the right window size is critical:

```
window_size = 10 × tau × framerate
```

| Indicator | Tau (s) | Framerate | Recommended Window |
|-----------|---------|-----------|-------------------|
| GCaMP6f | 0.7 | 30 Hz | ~210 frames |
| GCaMP6s | 1.8 | 30 Hz | ~540 frames |
| GCaMP7s | 1.0 | 17 Hz | ~170 frames |
| GCaMP8f | 0.25 | 30 Hz | ~75 frames |
| GCaMP8s | 0.5 | 30 Hz | ~150 frames |

**Why 10× tau × framerate?**

- The window must span **multiple calcium transients** so the percentile filter can find true baseline between events
- Too small: baseline contaminated by transients → underestimated ΔF/F
- Too large: slow drifts not tracked → baseline mismatch

```python
# Example: GCaMP7s at 17 Hz
tau = 1.0  # seconds
fs = 17.0  # Hz
window_size = int(10 * tau * fs)  # = 170 frames

dff = lsp.dff_rolling_percentile(F, window_size=window_size, percentile=20)
```

(understanding-shot-noise-levels)=
#### Shot Noise Levels

The noise histograms (`10_noise_accepted.png`, `11_noise_rejected.png`) show **standardized shot noise levels** for each ROI. This metric helps you:

- Compare noise levels across datasets, recordings, and experiments
- Identify problematic ROIs with unusually high noise
- Assess overall recording quality

**Shot Noise Formula:**

```{math}
\nu = \frac{\mathrm{median}_t\left( \left| \frac{\Delta F_t}{F_0} - \frac{\Delta F_{t+1}}{F_0} \right| \right)}{\sqrt{f_r}}
```

- **Median of frame-to-frame differences**: Takes advantage of slow calcium dynamics (adjacent frames should be similar)
- **Median (not mean)**: Excludes outliers from fast transient onsets
- **Normalized by √framerate**: Makes metric comparable across different acquisition rates

**Interpreting Noise Values:**

| Noise Level | Quality | Interpretation |
|-------------|---------|----------------|
| < 0.5 %/√Hz | Excellent | Very clean signal |
| 0.5-1.0 %/√Hz | Good | Typical for healthy recordings |
| 1.0-2.0 %/√Hz | Fair | May need filtering or careful analysis |
| > 2.0 %/√Hz | Poor | Consider excluding or investigating |

```{figure} _images/noise_comp.png
:alt: High vs Low Noise Levels
:name: ug-fig-noise-planar
:width: 100%

Top N neurons with highest and lowest standardized noise levels. Use this to identify outlier ROIs.
```

**Conditional outputs (if `chan2_file` provided):**

| File | Description |
|------|-------------|
| `data_chan2.bin` | Raw structural channel |
| `data_chan2_reg.bin` | Registered structural channel |
| `F_chan2.npy` | Channel 2 fluorescence |
| `Fneu_chan2.npy` | Channel 2 neuropil |

## Volumetric Pipeline

Instead of a single file, provide a list of z-planes or a directory:

```python
results = lsp.pipeline(
    input_data="D:/data/volume",    # directory or list of plane files
    save_path=None,                 # default: save next to input
    ops=None,                       # default: use MBO-optimized parameters
    planes=None,                    # default: process all planes
    roi=None,                       # default: stitch multi-ROI data
    keep_reg=True,                  # default: keep data.bin (registered binary)
    keep_raw=False,                 # default: delete data_raw.bin after processing
    force_reg=False,                # default: skip if already registered
    force_detect=False,             # default: skip if stat.npy exists
    dff_window_size=None,           # default: auto-calculate from tau and framerate
    dff_percentile=20,              # default: 20th percentile for baseline
    dff_smooth_window=None,         # default: auto-calculate from tau and framerate
    save_json=False,                # default: only save ops.npy
)
```

### Volumetric Outputs

When processing multiple planes with `lsp.pipeline()`, additional files are generated in the root save directory:

| File | Description |
|------|-------------|
| `all_planes_masks.png` | Grid showing ROI masks overlaid on mean images for all planes |
| `volume_quality_metrics.png` | Compactness, skewness, ROI size, and radius per plane (mean ± std) |
| `volume_trace_analysis.png` | Example traces, SNR and fluorescence per plane, activity heatmap |
| `volume_summary.csv` | Per-plane statistics table (ROIs, SNR, acceptance rate) |
| `mean_volume_signal.png` | Mean signal intensity across z-depth |
| `rastermap.png` | Activity sorted by similarity across all planes |
| `orthoslices.png` | Orthogonal slices through the volume (XY, XZ, YZ) |
| `roi_map_3d_snr.png` | 3D scatter plot of ROI centroids colored by SNR |
| `volume_stats.npy` | Per-plane statistics dictionary |

#### All Planes Masks

```{figure} _images/all_planes_masks.png
:alt: All Planes Masks
:name: ug-fig-all-planes
:width: 100%

**ROI masks across all z-planes** arranged in a grid. Each panel shows the enhanced mean image with segmentation overlay for one plane. Use this to quickly assess segmentation quality across the entire volume and identify planes with poor detection.
```

#### Orthogonal Slices

```{figure} _images/orthoslices.png
:alt: Orthoslices
:name: ug-fig-orthoslices
:width: 80%

**Orthogonal slices** through the mean intensity volume. Shows XY (axial), XZ (coronal), and YZ (sagittal) views with crosshairs indicating the slice positions. Useful for verifying volume alignment and identifying depth-dependent signal changes.
```

#### 3D ROI Map

```{figure} _images/roi_map_3d_snr.png
:alt: 3D ROI Map
:name: ug-fig-roi-3d
:width: 80%

**3D scatter plot of ROI centroids** with each point representing a detected cell. Points are colored by SNR (signal-to-noise ratio), revealing the spatial distribution of signal quality through the volume. Brighter colors indicate higher quality cells.
```

#### Volume Trace Analysis

```{figure} _images/volume_trace_analysis.png
:alt: Volume Trace Analysis
:name: ug-fig-vol-trace
:width: 100%

**Volume-wide trace analysis** combining multiple views:
- **Left**: Example ΔF/F traces from representative cells across planes
- **Middle**: SNR and mean fluorescence as a function of z-depth
- **Right**: Activity heatmap showing temporal patterns across all cells

This visualization helps identify depth-dependent signal quality and synchronized activity patterns.
```

#### Rastermap (Volume-wide)

```{figure} _images/rastermap.png
:alt: Rastermap
:name: ug-fig-rastermap-vol
:width: 100%

**Activity sorted by similarity** using the [Rastermap](https://github.com/MouseLand/rastermap) algorithm. Cells from all planes are combined and reordered so that neurons with similar activity patterns appear adjacent. This reveals functional clusters and population dynamics that may span multiple z-planes.
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
# Returns dict with: F, Fneu, spks, stat, iscell, z_plane
# iscell is (n_rois, 2): column 0 is classification (0/1), column 1 is probability

# Calculate ΔF/F with rolling percentile baseline
dff = dff_rolling_percentile(
    results['F'],
    window_size=500,  # frames
    percentile=20     # baseline percentile
)

# Filter for accepted cells only
iscell_mask = results['iscell'][:, 0].astype(bool)
F_cells = results['F'][iscell_mask]
spks_cells = results['spks'][iscell_mask]
```

---

(parameters)=
## Critical Parameters

Understanding key Suite2p parameters is essential for good segmentation results.

```{admonition} Recommended: Anatomical Segmentation
:class: tip

For LBM data, we recommend **anatomical segmentation with Cellpose** over functional detection. Anatomical detection uses structural features (cell morphology) rather than activity correlations, which works better for densely labeled tissue and datasets with variable activity levels.
```

### Anatomical Segmentation (Cellpose) - Recommended

Anatomical segmentation uses [Cellpose](https://www.cellpose.org/) to detect cell bodies based on morphology rather than functional activity. This is the recommended approach for LBM data.

#### Quick Start: Anatomical Detection

```python
ops = {
    "anatomical_only": 3,      # Use enhanced mean image (recommended)
    "diameter": 6,             # Expected cell diameter in pixels
    "cellprob_threshold": 0.0, # Cell probability threshold
    "flow_threshold": 0.4,     # Flow error threshold
}
```

#### `anatomical_only` (default: 0)

Enables anatomical segmentation using Cellpose. The value determines which image is used for detection:

| Value | Image Used | Description |
|-------|------------|-------------|
| `0` | Disabled | Functional detection (correlation-based) |
| `1` | `max_proj / mean_img` | Ratio highlighting active areas |
| `2` | `mean_img` | Average image over all frames |
| `3` | `meanImgE` | **Recommended**: Enhanced mean with edge sharpening |
| `4` | `max_proj` | Maximum projection across frames |

```python
ops["anatomical_only"] = 3  # Use enhanced mean image
```

#### `diameter` (default: 0)

Expected cell diameter in pixels. **Required for Cellpose**.

- **0**: Auto-estimate (Suite2p default, but not recommended for Cellpose)
- **4-8**: Typical for LBM data at 2 µm/pixel resolution
- **LBM override**: If `diameter` is 0/None/NaN and `anatomical_only > 0`, LBM sets it to 8

```python
ops["diameter"] = 6  # For ~12 µm cells at 2 µm/pixel
```

#### `cellprob_threshold` (default: 0.0)

Probability threshold from Cellpose output to determine cell boundaries. More negative values include more pixels.

- **0.0**: Standard threshold
- **-2.0**: More permissive (include dimmer cells)
- **2.0**: More stringent (only bright, clear cells)

```python
ops["cellprob_threshold"] = 0.0  # Standard
```

#### `flow_threshold` (default: 1.5)

Minimum Cellpose flow error to consider a region valid. Lower values include more ROIs.

- **1.5**: Standard (Suite2p default)
- **0.4**: More permissive (recommended for LBM)
- **0.1**: Very permissive

```python
ops["flow_threshold"] = 0.4  # More permissive for LBM data
```

#### `spatial_hp_cp` (default: 0.0)

High-pass filtering applied before Cellpose segmentation. A float between 0 and 1.

- **0.0**: No filtering (default)
- **0.5**: Moderate high-pass (reduces background)
- **1.0**: Strong high-pass

```python
ops["spatial_hp_cp"] = 0.0  # No pre-filtering
```

---

### Functional Detection Parameters

For datasets where anatomical detection doesn't work well, or when you need activity-based ROI detection, use functional parameters.

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

#### Visual Parameter Comparisons

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

#### `threshold_scaling` (default: 1.0)

Multiplier for detection threshold. **Lower values detect more ROIs.**

- **0.8-1.2**: Good starting range
- **<0.8**: May detect noise/background
- **>1.5**: May miss dim cells

```python
ops["threshold_scaling"] = 0.9  # More sensitive detection
```

```{figure} _images/default_params_thr.png
:name: fig-threshold-scaling
:alt: Effect of threshold_scaling on detected cells

Effect of varying `threshold_scaling` on detected cells. Notably, increasing this threshold actually led to several cells being detected that were not otherwise detected, and vice-versa.
```

#### `tau` (default: 1.0)

Calcium indicator decay time constant in seconds. **Critical for binning and deconvolution.**

GCaMP expression is slow, often taking between 100 ms to over 1 second for the signal to rise and decay. This is the timescale of the sensor, in seconds. We need this value because one of the main performance optimizations is [binning](https://en.wikipedia.org/wiki/Data_binning). We can bin our data **because of this slow timescale** - we set the bin-size to our sensor's timescale because we expect all frames in this window to be the same (on average).

| Indicator | Tau (seconds) |
|-----------|---------------|
| GCaMP6f | 0.7-1.0 |
| GCaMP6s | 1.5-2.0 |
| GCaMP7/8 variants | ~1.0 |

**When in doubt, round up!**

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

Maximum allowed spatial overlap between ROIs (0-1). If two masks overlap by a fraction > max_overlap, they will be discarded/rejected.

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

Gaussian filter size applied during functional cell detection to reduce background noise.

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

#### Registration Output Files

Registration produces these files in addition to the main outputs:

| File/Key | Description |
|----------|-------------|
| `data.bin` | Motion-corrected movie (channel 1) |
| `data_chan2.bin` | Registered channel 2 (if `nchannels=2`) |
| `ops['xoff']` | Rigid X shifts for each frame |
| `ops['yoff']` | Rigid Y shifts for each frame |
| `ops['corrXY']` | Phase-correlation scores (frame vs reference) |
| `ops['refImg']` | Reference image used for registration |

#### Registration Metrics (`ops['regDX']`)

Suite2p computes PC-based metrics to assess registration quality:

| Metric | Description |
|--------|-------------|
| `regPC` | Spatial principal components used for QC |
| `tPC` | Time courses of principal components |
| `regDX` | Shift distance between split PCs (lower = better) |

You can compute metrics manually:

```python
from suite2p.registration import metrics
ops = metrics.get_pc_metrics(ops, use_red=False)
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
lsp.grid_search(
    input_file=test_plane_file,
    save_path=grid_search_dir,
    grid_params=grid_params,
    ops=ops,
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

lsp.grid_search(
    input_file=input_tiff,
    save_path=save_path / "registration",
    grid_params=search_dict,
    ops=base_ops,
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

#### EXTRACT

[EXTRACT](https://github.com/schnitzer-lab/EXTRACT-public) outputs raw fluorescence signals without built-in ΔF/F₀ calculation. You compute it yourself using something like a low-percentile (e.g. 10%) as F₀. Most users apply a global or sliding percentile window.

**Neuropil:** Handled implicitly. The algorithm uses robust factorization to ignore background and neuropil. There's no explicit subtraction or coefficient to tune—it isolates only what fits a consistent spatial footprint and suppresses outliers by design.

### Using LBM-Suite2p-Python ΔF/F Functions

```{tip}
For detailed guidance on choosing `window_size` based on your indicator and framerate, see [Understanding ΔF/F Traces](#understanding-δff-traces) in the Planar Outputs section.
```

#### Rolling Percentile Baseline (Recommended)

```python
from lbm_suite2p_python import dff_rolling_percentile

# Calculate window size: 10 × tau × framerate
tau = 1.0   # GCaMP7s decay time constant (seconds)
fs = 17.0   # Your acquisition framerate (Hz)
window_size = int(10 * tau * fs)  # = 170 frames

dff = dff_rolling_percentile(
    F,
    window_size=window_size,
    percentile=20,      # Use 20th percentile as baseline
    use_median_floor=False  # Optional: set min F₀ at 1% of median
)
```

**Window size rule of thumb:** Use ~10× the indicator decay time constant (tau) × frame rate.
For example, with jGCaMP7s (tau≈1.0s) at 30Hz: 10 × 1.0 × 30 = 300 frames.
This ensures the window spans multiple calcium transients so the percentile filter
can find the baseline between events. See [Suite2p Cell Detection](https://suite2p.readthedocs.io/en/latest/celldetection.html) for related binning logic.

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

---

## Further Reading

- {doc}`API Reference <api>` - Function signatures and docstrings
- {doc}`Glossary <glossary>` - Term definitions
- [Suite2p Documentation](https://suite2p.readthedocs.io/) - Detailed Suite2p parameter guide
- [Cellpose Documentation](https://cellpose.readthedocs.io/) - Anatomical segmentation details
