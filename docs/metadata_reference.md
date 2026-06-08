# Metadata Processing Reference

Complete reference for all metadata handling in LBM-Suite2p-Python.

---

## Table of Contents

1. [Core Metadata Sources](#1-core-metadata-sources--extraction)
2. [Suite2p ops Dictionary](#2-suite2p-ops-dictionary-fields)
3. [stat.npy Array Fields](#3-statnpy-array-fields-per-roi-metadata)
4. [Processing History](#4-processing-history-metadata)
5. [Format Conversion](#5-format-conversion-metadata)
6. [Validation & Comparison](#6-validation--comparison-metadata)
7. [Cell Filtering](#7-cell-filtering-metadata)
8. [Multi-ROI Merging](#8-multi-roi-merging-metadata)
9. [Normcorre Registration](#9-normcorre-registration-metadata)
10. [dF/F Computation](#10-dff-computation-metadata)
11. [Cellpose Training](#11-cellpose-training-metadata)
12. [File Format Storage](#12-file-format-metadata-storage)
13. [Lazy Array Access](#13-lazy-array-metadata-access)
14. [Volume Statistics](#14-volume-statistics-metadata)
15. [Image Cropping Logic](#15-image-croppingexpansion-logic)
16. [Normalization Functions](#16-metadata-normalization-functions)
17. [Metadata Flow Diagram](#17-metadata-flow-diagram)

---

## 1. Core Metadata Sources & Extraction

| Function | File:Line | Input | Output Fields | Notes |
|----------|-----------|-------|---------------|-------|
| `get_param(metadata, key)` | mbo_utilities | ScanImage metadata dict | Single value (fs, nframes, etc.) | primary metadata accessor from mbo_utilities |
| `get_voxel_size(metadata)` | mbo_utilities | Metadata dict or ops | VoxelSize object with `.dx`, `.dy`, `.dz`, `.pixel_resolution` | returns microns/pixel; defaults to 1.0 if unavailable |
| `detect_stack_type(metadata)` | mbo_utilities | ScanImage metadata | `"lbm"`, `"piezo"`, `"single"` | determines acquisition mode from SI metadata |
| `default_ops(metadata, ops)` | default_ops.py:170-222 | Optional metadata dict | ops dict with `fs`, `dx`, `dy` populated | merges defaults with extracted metadata |
| `s2p_ops()` | default_ops.py:6-167 | None | Complete Suite2p default parameters | 80+ fields for registration, detection, extraction |

---

## 2. Suite2p ops Dictionary Fields

### 2.1 Spatial/Dimensional Metadata

| Field | Type | Source | Purpose |
|-------|------|--------|---------|
| `Ly` | int | Binary write | image height in pixels |
| `Lx` | int | Binary write | image width in pixels |
| `nframes` | int | Binary size / metadata | total frame count |
| `nframes_chan1` | int | Suite2p | frames in functional channel |
| `nplanes` | int | default_ops | number of z-planes |
| `nchannels` | int | default_ops | channels per acquisition |
| `yrange` | [int, int] | Registration | valid Y crop region after registration |
| `xrange` | [int, int] | Registration | valid X crop region after registration |
| `dx` | float | get_voxel_size | X pixel size in microns |
| `dy` | float | get_voxel_size | Y pixel size in microns |
| `dz` | float | get_voxel_size | Z-plane spacing in microns |
| `pixel_resolution` | [dx, dy, dz] | grid_search | combined voxel dimensions |
| `aspect` | float | default_ops | aspect ratio (um/px X / um/px Y) |

### 2.2 Temporal Metadata

| Field | Type | Source | Purpose |
|-------|------|--------|---------|
| `fs` | float | get_param(metadata, "fs") | frame rate in Hz (per plane) |
| `tau` | float | default_ops (1.0) | calcium indicator decay constant (seconds) |
| `num_timepoints` | int | pipeline | actual volume timepoints (frames/planes) |
| `frames_include` | int | default_ops (-1) | number of frames to process (-1=all) |

### 2.3 Image Metadata (Processing Results)

| Field | Shape | Notes |
|-------|-------|-------|
| `meanImg` | (Ly, Lx) | full-size mean projection |
| `meanImgE` | (Ly, Lx) | full-size enhanced mean (neuropil) |
| `refImg` | (Ly, Lx) | full-size registration reference |
| `max_proj` | (yrange, xrange) | cropped max projection |
| `Vcorr` | (yrange, xrange) | cropped correlation map |
| `sdmov` | (Ly, Lx) | standard deviation of movie |

### 2.4 Registration Metadata

| Field | Type | Purpose |
|-------|------|---------|
| `do_registration` | bool/int | whether registration was performed |
| `do_bidiphase` | bool | compute bidirectional phase offset |
| `bidiphase` | float | phase offset value in pixels |
| `bidi_corrected` | bool | whether bidirectional correction applied |
| `nonrigid` | bool | use piecewise-rigid registration |
| `block_size` | [int, int] | patch size for nonrigid registration |
| `maxregshift` | float | max allowed shift as fraction of frame |
| `smooth_sigma` | float | gaussian smoothing before registration |
| `smooth_sigma_time` | float | temporal smoothing for reference |
| `two_step_registration` | int | enable two-step registration (coarse then fine) |

### 2.5 Detection/Classification Metadata

| Field | Type | Purpose |
|-------|------|---------|
| `roidetect` | bool | whether to run ROI detection |
| `anatomical_only` | int | 0=functional; 1/2/4 enable cellpose and select `cellpose_settings.img` (1=`max_proj / meanImg`, 2=`meanImg`, 4=`max_proj`); 3 (enhanced mean) removed |
| `diameter` | int/list | expected cell diameter in pixels |
| `diameter_user` | int/list | original user-specified diameter (preserved) |
| `spatial_scale` | int | ROI scale: 0=multi, 1=6px, 2=12px, 3=24px, 4=48px |
| `threshold_scaling` | float | detection threshold multiplier |
| `cellprob_threshold` | float | cellpose probability threshold |
| `flow_threshold` | float | cellpose flow error threshold |
| `sparse_mode` | bool | use sparse detection algorithm |
| `connected` | bool | keep ROIs fully connected |

### 2.6 Path Metadata

| Field | Purpose |
|-------|---------|
| `save_path` | plane output directory |
| `save_path0` | parent results directory |
| `save_folder` | results folder name |
| `data_path` | input data path |
| `ops_path` | path to ops.npy |
| `raw_file` | path to data_raw.bin |
| `reg_file` | path to data.bin |
| `chan2_file` | path to structural channel binary |

---

## 3. stat.npy Array Fields (Per-ROI Metadata)

| Field | Type | Purpose |
|-------|------|---------|
| `ypix` | np.int32[] | Y pixel coordinates of ROI |
| `xpix` | np.int32[] | X pixel coordinates of ROI |
| `zpix` | np.int32[] | Z pixel coordinates (if 3D) |
| `npix` | int | number of pixels in ROI |
| `lam` | np.float32[] | intensity weights (sum=1) |
| `overlap` | bool[] | mask of overlapping pixels |
| `med` | [y, x] | centroid position |
| `med_z` | float | median Z position (if 3D) |
| `radius` | float | fitted radius in pixels |
| `aspect_ratio` | float | elongation metric (height/width) |
| `compact` | float | compactness (npix / πr²) |
| `mean_intensity` | float | mean pixel value in ROI |
| `max_intensity` | float | max pixel value in ROI |
| `footprint` | int | source algorithm (0=suite2p, 1=cellpose) |
| `mrs` | float | mean registration shift |
| `mrs0` | float | mean rigid shift |
| `mrs1` | float | mean nonrigid shift |

---

## 4. Processing History Metadata

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `_add_processing_step()` | run_lsp.py:82-130 | appends step to `ops["processing_history"]` |

**Each history entry contains:**

| Field | Type | Content |
|-------|------|---------|
| `step` | str | step name (binary_write, registration, detection, etc.) |
| `timestamp` | str | ISO format datetime |
| `lbm_suite2p_python_version` | str | package version |
| `suite2p_version` | str | suite2p version |
| `input_files` | list[str] | source file paths |
| `duration_seconds` | float | processing time |
| `extra` | dict | step-specific metadata |

---

## 5. Format Conversion Metadata

### 5.1 Suite2p to Cellpose

| Function | File:Line | Input | Output |
|----------|-----------|-------|--------|
| `suite2p_to_cellpose()` | conversion.py:269-349 | ops.npy, stat.npy | masks.npy, cellpose_seg.npy |
| `export_for_gui()` | conversion.py:552-642 | Suite2p dir | {name}.tif + {name}_seg.npy |
| `stat_to_masks()` | conversion.py:221-246 | stat array, shape | label mask (0=bg, N=ROI_ID) |

**cellpose_seg.npy structure:**

```python
{
    "img": projection_image,      # summary image
    "masks": label_mask,          # uint16 label image
    "outlines": outline_mask,     # binary outlines
    "chan_choose": [0, 0],        # channel selection
    "ismanual": bool_array,       # manual edit flags
    "filename": projection_path,  # source image path
    "flows": flow_fields,         # optical flow (or None)
    "est_diam": diameter,         # estimated diameter
    "cellprob_threshold": float,  # detection threshold
    "flow_threshold": float,      # flow error threshold
}
```

### 5.2 Cellpose to Suite2p

| Function | File:Line | Input | Output |
|----------|-----------|-------|--------|
| `cellpose_to_suite2p()` | conversion.py:352-465 | masks.npy, cellpose_seg.npy | ops.npy, stat.npy, iscell.npy |
| `import_from_gui()` | conversion.py:645-728 | edited _seg.npy | updated stat.npy, iscell.npy |
| `masks_to_stat()` | conversion.py:249-266 | label mask, image | stat array |

**conversion_meta.npy structure:**

```python
{
    "source_format": "suite2p" | "cellpose",
    "source_path": str,
    "converted_at": ISO_timestamp,
    "n_rois": int,
    "shape": [Ly, Lx],
    "img_key": str,              # for suite2p to cellpose
    "traces_extracted": bool,    # for cellpose to suite2p
}
```

---

## 6. Validation & Comparison Metadata

| Function | File:Line | Returns |
|----------|-----------|---------|
| `validate_format()` | conversion.py:31-119 | `{valid, format, files, n_rois, shape, warnings}` |
| `compare_detections()` | conversion.py:731-811 | `{n_a, n_b, n_matched, matched_pairs, unique_to_a, unique_to_b, mean_iou}` |

---

## 7. Cell Filtering Metadata

| Function | File:Line | Metadata Used | Notes |
|----------|-----------|---------------|-------|
| `filter_by_max_diameter()` | postprocessing.py:179-314 | `stat[i]["radius"]`, `ops["dx"]`, `ops["dy"]` | converts µm to px via voxel size |
| `filter_by_area()` | postprocessing.py:317-412 | `len(stat[i]["xpix"])` | pixel count per ROI |
| `filter_by_eccentricity()` | postprocessing.py:415-494 | `stat[i]["ypix"]`, `stat[i]["xpix"]` ranges | aspect ratio from bounding box |
| `apply_filters()` | postprocessing.py:497-600 | filter config list | returns iscell_filtered, removed_mask, filter_results |

**Filter result metadata:**

```python
{
    "filter_name": str,
    "n_removed": int,
    "n_remaining": int,
    "threshold": value,
    "removed_indices": array,
}
```

---

## 8. Multi-ROI Merging Metadata

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `group_plane_rois()` | merging.py:11-44 | groups directories by plane from pattern `planeXX_roiYY` |
| `merge_mrois()` | merging.py:105-232 | merges ROI directories into single plane |

**Coordinate transformations during merge:**

- `stat["xpix"] += x_offset` (horizontal concatenation)
- `stat["med"][1] += x_offset` (centroid update)
- traces concatenated along ROI axis

---

## 9. Normcorre Registration Metadata

| Function | File:Line | Parameters |
|----------|-----------|------------|
| `normcorre_ops()` | normcorre.py:35-76 | default registration parameters |

**normcorre_ops fields:**

| Field | Default | Purpose |
|-------|---------|---------|
| `max_shifts` | (10, 10) | max (y, x) shift in pixels |
| `strides` | (48, 48) | patch size for piecewise-rigid |
| `overlaps` | (24, 24) | patch overlap |
| `upsample_factor` | 10 | subpixel precision |
| `max_deviation_rigid` | 3 | threshold for rigid vs. piecewise |
| `border_nan` | "copy" | how to handle borders |
| `template_method` | "median" | template computation method |
| `template_max_frames` | 500 | max frames for template |

---

## 10. dF/F Computation Metadata

| Function | File:Line | Metadata Parameters |
|----------|-----------|---------------------|
| `dff_rolling_percentile()` | postprocessing.py:794-900+ | `fs`, `tau`, `window_size`, `percentile`, `smooth_window` |

**Auto-calculation from metadata:**

- `window_size = ~10 × tau × fs` (baseline window)
- `smooth_window = ~0.5 × tau × fs` (temporal smoothing)

---

## 11. Cellpose Training Metadata

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `train_cellpose()` | cellpose.py | train custom model from pretrained cpsam |
| `prepare_training_data()` | cellpose.py | export images + masks for training |
| `annotate()` | cellpose.py | launch GUI for manual annotation |

**Training requires paired files:**

- `{name}.tif` - image file
- `{name}_seg.npy` - matching segmentation (naming convention critical)

---

## 12. File Format Metadata Storage

| Format | Metadata Location | Access Method |
|--------|-------------------|---------------|
| `.bin` | Separate `ops.npy` | dimensions from `ops["Ly"]`, `ops["Lx"]`, file size |
| `.npy` | Embedded (allow_pickle=True) | `np.load(path, allow_pickle=True).item()` |
| `.tif` | TIFF tags + ScanImage metadata | mbo_utilities extraction |
| `.zarr` | `.zattrs` JSON | `arr.metadata` attribute |
| `.h5` | HDF5 attributes | `ops["h5py_key"]` for dataset access |
| `.json` | ops.json export | `ops_to_json()` for human-readable |

---

## 13. Lazy Array Metadata Access

| Array Type | Metadata Attribute | Source |
|------------|-------------------|--------|
| `MboRawArray` | `.metadata` | ScanImage TIFF headers |
| `ScanImageArray` | `.metadata` | ScanImage JSON embedded |
| `LBMArray` | `.metadata` | inherited from base |
| `PiezoArray` | `.metadata` | inherited from base |
| `Suite2pArray` | `.metadata` | loaded from ops.npy |
| `ZarrArray` | `.metadata` | `.zattrs` file |
| `H5Array` | `.metadata` | HDF5 attributes |
| `BinArray` | `.metadata` | associated ops.npy |

**Additional MboRawArray properties:**

- `.fix_phase` - phase correction enabled
- `.use_fft` - FFT subpixel correction
- `.roi` - current ROI mode
- `.num_rois` - total ROI count

---

## 14. Volume Statistics Metadata

| Function | File:Line | Collected Data |
|----------|-----------|----------------|
| `get_volume_stats()` | volume.py:276+ | per-plane ops aggregation |

**Volume metadata structure:**

```python
{
    "planes": [plane_indices],
    "n_rois_per_plane": [counts],
    "total_rois": int,
    "voxel_size": (dx, dy, dz),
    "volume_shape": (nz, ny, nx),
    "processing_times": {plane: seconds},
}
```

---

## 15. Image Cropping/Expansion Logic

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `_get_summary_image()` | conversion.py:130-206 | handles full vs. cropped images |

**Size handling:**

- Full-size images (`meanImg`, `meanImgE`, `refImg`): Shape = (Ly, Lx)
- Cropped images (`max_proj`, `Vcorr`): Shape = (yrange[1]-yrange[0], xrange[1]-xrange[0])
- Expansion: cropped images padded with zeros to (Ly, Lx) using `yrange`, `xrange`

---

## 16. Metadata Normalization Functions

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `_normalize_iscell()` | postprocessing.py:13-17 | converts (n, 2) to (n,) boolean |
| `load_ops()` | postprocessing.py:1223-1249 | loads ops from path, dir, or returns dict |
| `ops_to_json()` | postprocessing.py:695-747 | serializes numpy types for JSON export |

**JSON serialization handles:**

- `np.ndarray` → `list`
- `np.integer` → `int`
- `np.floating` → `float`
- `np.bool_` → `bool`
- `Path` → `str`

---

## 17. Metadata Flow Diagram

```
Input Files (TIFF/Zarr/HDF5)
        │
        ▼
┌─────────────────────┐
│   mbo_utilities     │
│  - get_param()      │
│  - get_voxel_size() │
│  - detect_stack_type│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    default_ops()    │
│  Merge: defaults +  │
│  metadata + user    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│     pipeline()      │──────┐
│  - plane_ops copy   │      │
│  - processing_history      │
└─────────┬───────────┘      │
          │                  │
          ▼                  ▼
┌─────────────────┐  ┌──────────────┐
│  Binary Write   │  │ run_plane_bin│
│  Updates: Ly,Lx │  │ Suite2p core │
│  nframes, shape │  └──────┬───────┘
└─────────────────┘         │
                            ▼
                   ┌────────────────┐
                   │  Registration  │
                   │ Updates: yrange│
                   │ xrange, refImg │
                   └────────┬───────┘
                            │
                            ▼
                   ┌────────────────┐
                   │   Detection    │
                   │ Creates: stat  │
                   │ iscell, F, Fneu│
                   └────────┬───────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ Postprocessing │
                   │ - Cell filters │
                   │ - dF/F compute │
                   │ - Plot generate│
                   └────────┬───────┘
                            │
                            ▼
                   ┌────────────────┐
                   │   Output Files │
                   │ ops.npy, stat  │
                   │ F, Fneu, spks  │
                   │ iscell, norm   │
                   └────────────────┘
```

---

## Summary

The `ops` dictionary serves as the central metadata hub in Suite2p, with additional structured data in:

- `stat.npy` - per-ROI spatial and intensity metadata
- `iscell.npy` - classification results (n_rois, 2)
- `F.npy`, `Fneu.npy`, `spks.npy`, `norm_traces.npy` - trace data (n_rois, nframes)
- `processing_history` - append-only log within ops

All metadata flows through the pipeline with consistent access patterns via mbo_utilities for input extraction and numpy serialization for output storage.
