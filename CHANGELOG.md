# Changelog

## v3.1.0 — 2026-06-08

First release on the v3 line since v2.5.2. Highlights below; for the full diff see
`v2.5.2...v3.1.0`.

### Added
- GPU-accelerated Suite2p registration and detection. Device selection honors the
  `mbo_gpu` environment variable.
- Cellpose anatomical segmentation (`cpsam` model) via the optional `[cellpose]`
  extra, plus a `cellpose-gui` entry point and a runnable Cellpose input-image
  evaluation script.
- Per-ROI statistics written to `roi_stats.npy` (SNR, skew, shot noise, mean
  fluorescence, compactness, radius), computed automatically after dF/F.
- Multiplane parallel processing.
- Accepted/rejected segmentation figures and volume trace figures; figures 10–15
  now render in dark mode.

### Changed
- Z-score is now the default trace normalization. The dF/F output file is renamed
  `dff.npy` → `norm_traces.npy`, and `norm_method` selects the normalization.
- Parameters may be passed as either flat or nested dicts.
- `grid_search` copies input data instead of forcing re-registration; its argument
  was renamed `input_path` → `input_data`.
- Output filenames now reflect data dimensions and timepoint selection.
- Compatible with `mbo_utilities` 3.1.0, including 5D arrays and forward
  compatibility with the v4 `LazyArray` API.

### Fixed
- Suite2p `lx`/`ly` padding bug worked around with a padding shim; axial
  registration now uses valid (unpadded) regions for mean images.
- Force re-registration of stale `data_raw.bin`.
- Plane separation and channel-slicing bugs; correct z-plane numbering on load.
- `anatomical_only=3` now warns instead of failing silently.
