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

# 1. Get assembled planar TIFF files
files = mbo.get_files(data_dir, "tiff", max_depth=3)

# 2. Extract metadata and create ops
metadata = mbo.get_metadata(files[0])
ops = mbo.params_from_metadata(metadata)  # Auto-fills fs, dx, dy, Ly, Lx

# 3. Process entire volume
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
    input_path=plane_file,
    save_path=plane_dir,
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

#### `tau` (default: 1.0)
Calcium indicator decay time constant in seconds. **Critical for binning and deconvolution.**
- **GCaMP6f**: 0.7-1.0 s
- **GCaMP6s**: 1.5-2.0 s
- **GCaMP8s/m/f**: ~1.0 s
- **When in doubt, round up!**

Determines bin size: `bin_size = tau * fs`

```python
ops["tau"] = 1.0  # For GCaMP6f/8 at 17 Hz → ~17 frames/bin
```

#### `max_overlap` (default: 0.75)
Maximum allowed spatial overlap between ROIs (0-1).
- **0.75**: Default, reject ROIs with >75% overlap
- **1.0**: Keep all overlapping ROIs
- **0.5**: More stringent overlap rejection

```python
ops["max_overlap"] = 0.85  # Allow more overlap for dense regions
```

### Registration Parameters

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
Which image to use for Cellpose segmentation:
- **0 or False**: Disabled, use Suite2p functional detection
- **1**: `max_proj / mean_img` ratio (highlights active regions)
- **2**: `mean_img` (average intensity)
- **3**: `meanImgE` (enhanced/sharpened mean image)
- **4**: `max_proj` (maximum projection)

```python
ops["anatomical_only"] = 3  # Use enhanced mean image
ops["diameter"] = 8         # Required for Cellpose
ops["sparse_mode"] = False  # Turn off sparse detection
```

#### `cellprob_threshold` (default: 0.0)
Cellpose cell probability threshold. More negative = more inclusive.
- **0.0**: Standard
- **-2.0**: More permissive

#### `flow_threshold` (default: 1.5)
Cellpose flow error threshold. Lower = more ROIs.
- **1.5**: Standard
- **0.4**: More permissive

---

## Parameter Grid Search

Test multiple parameter combinations systematically:

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

Multiple methods for computing ΔF/F:

### Rolling Percentile Baseline (Recommended)

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

### Median Filter Baseline

```python
from lbm_suite2p_python import dff_median_filter

dff = dff_median_filter(F)
# Uses 1% of median as F₀ (simple but less adaptive)
```

**When to use:** Quick baseline for stable recordings.

### Shot Noise Estimation

```python
from lbm_suite2p_python import dff_shot_noise

noise_levels = dff_shot_noise(dff, fr=17.0)  # Frame rate in Hz
# Returns noise level per neuron in %/√Hz units
```

Quantifies SNR for comparing across datasets.

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
