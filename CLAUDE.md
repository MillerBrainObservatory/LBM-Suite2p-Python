# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A volumetric 2-photon calcium imaging pipeline for Light Beads Microscopy (LBM) data, built as a **compatibility layer over upstream MouseLand/suite2p**. The package wraps suite2p's registration, detection, and extraction so an entire imaging *volume* (many z-planes) can be processed plane-by-plane with one set of parameters, plus diagnostics and plotting. I/O, lazy arrays, and metadata come from the sibling package `mbo_utilities`.

## Commands

```bash
# dev install (uv recommended; pip works too)
uv pip install -e ".[dev]"

# run the test suite
uv run pytest tests
pytest tests/test_streaming.py                      # one file
pytest tests/test_streaming.py::test_setitem_is_discarded   # one test

# lint / format (ruff: line-length 88, numpy docstrings, select=ALL)
ruff format .
ruff check .

# CLI
lsp <input> <output> [options]      # run the pipeline
lsp --list-ops                      # list every settable suite2p parameter
lsp --version                       # fast; does not import suite2p
lsp convert <dir> --to cellpose     # format conversion subcommand
lsp detect <path>                   # format detection subcommand
```

Note: the GitHub Actions test step is **commented out** in `.github/workflows/test_python.yml`; CI currently only syncs deps and runs `ruff format`. Run tests locally. Many tests in `tests/test_streaming*.py` are `skipif` when suite2p/cv2 are absent.

## Architecture

### Call graph

`pipeline()` is the single public entry point. It loads the input as a lazy array (via `mbo.imread`), detects 3D (single plane) vs 4D (volume), and delegates:

```
pipeline()                       run_lsp.py  — input detection, deprecation handling, db/settings flattening
  ├─ run_volume()                run_lsp.py  — iterates planes; sequential OR ProcessPoolExecutor workers; aggregates volume_stats + volumetric plots
  │    └─ run_plane()  (per plane)
  └─ run_plane()                 run_lsp.py  — single plane: imwrite binary OR stream, then dispatch to one of:
       ├─ run_plane_bin()        runs suite2p against data_raw.bin / data.bin on disk
       └─ run_plane_stream()     streams frames straight from the lazy array (no binaries)
            └─ _call_upstream_pipeline()      the bridge to suite2p.run_s2p.pipeline
```

The pipeline **always processes one plane and one channel at a time** — `default_ops()` forces `nplanes=1`, `nchannels=1`. "Volumetric" means looping `run_plane` over z, not handing suite2p a 4D array.

### The upstream-compatibility seam (most important to understand)

The fork uses a **flat `ops` dict** everywhere (`ops["meanImg"]`, `ops["yrange"]`, `ops["fs"]`, ...). Upstream suite2p moved to **nested `db` + `settings` dicts** and a `pipeline()` that returns a 12-tuple instead of a merged ops. Two modules absorb that mismatch — touch them whenever a suite2p upgrade breaks things:

- **`_call_upstream_pipeline()`** (`run_lsp.py`): splits flat ops into `(db, settings)`, overlays them onto a fresh `default_settings()` (so new upstream keys fall back to upstream defaults rather than crashing), resolves the torch device, calls upstream `pipeline()`, then folds the returned `reg_outputs` / `detect_outputs` dicts back into the flat ops dict.
- **`db_settings.py`**: the bidirectional flat↔nested translation (`db_settings_to_ops`, `ops_to_db_settings`, `merge_db_settings_into_ops`). Section membership tables (`_SETTINGS_SECTIONS`, etc.) mirror upstream's `default_settings()`/`default_db()` shapes and must stay in sync with them. Fork-only keys (`dff_*`, `keep_raw`, registration outputs, mbo metadata) live in `ops.npy` only and do not round-trip.

`compute_enhanced_mean_image()` is similarly a shim for an upstream rename (`highpass_mean_image`). Expect more such shims when bumping suite2p.

### Ops resolution

`default_ops.py` defines the parameter baseline: `s2p_ops()` returns suite2p's own defaults flattened through `db_settings_to_ops`; `default_ops()` overlays a small `_LBM_DETECTION_DEFAULTS` set on top. A caller-supplied `ops=` is respected as-is (not overlaid). `pipeline()`/`run_plane()` accept **either** flat `ops=` **or** nested `db=`/`settings=`; when both are given, the nested pair is flattened first and explicit `ops` keys win. `reconcile_detection_keys()` keeps the fork's `anatomical_only`/`sparse_mode` spellings and suite2p's `algorithm` spelling consistent and valid.

### Streaming mode (`stream=True`)

`streaming.py` / `streaming_array.py` let suite2p run without ever writing the full `data_raw.bin` / `data.bin`:

- **`StreamingBinaryFile`** is a duck-typed stand-in for suite2p's `BinaryFile`. As `f_raw` it returns int16 frames sliced from the lazy array; as `f_reg` (`registered=True`) it **discards writes** and reconstitutes registered frames on read by replaying the shifts suite2p saved to `reg_outputs.npy` (bidiphase + rigid roll + nonrigid `transform_data`). Reconstituted frames are byte-identical to `data.bin` (verified in `tests/test_streaming.py`) because the same raw source, shifts, block layout, and int16 cast are reused. The only persisted binary artifact is the tiny `reg_outputs.npy`.
- Streaming is **functional-channel-only**; a chan2/structural file or a `.bin` input falls back to the binary path. `two_step_registration` is auto-disabled.
- **`RegisteredStreamArray`** (`streaming_array.py`) is the mbo_utilities side: registered via the `mbo_utilities.lazy_arrays` entry point in `pyproject.toml`, it lets `mbo <plane_dir>` / `imread(<plane_dir>)` open a binary-less plane dir (ops.npy + reg_outputs.npy, no .bin) as a lazy registered array. It is deliberately isolated from `streaming.py` so a load failure can never break mbo_utilities.

### Lazy imports

`__init__.py` uses PEP 562 `__getattr__` to map every public name to its submodule, imported on first access. Importing the package does **not** pull in suite2p / cv2 / matplotlib, so `lsp --version` / `--help` stay fast. The CLI (`cli.py`) mirrors this: `build_parser(include_ops=False)` skips enumerating suite2p ops (which would import suite2p) for the `--help` / no-arg / `--version` / `--list-ops` fast paths. When adding a public function, register it in both `_LAZY_ATTRS` and `__all__`.

### Parallelism & device

- `run_volume(workers=...)`: `1` = sequential; `None`/`<=0` = auto `min(num_planes, cpu_count//2, 8)`; `>1` = that many `ProcessPoolExecutor` workers. Parallel mode requires a **path-based** input (workers re-`imread` the source; lazy arrays may not survive spawn) and writes to disjoint per-plane subdirs. Worker logs are funneled back through a multiprocessing log queue. Cellpose on GPU can OOM with multiple workers.
- `MBO_GPU` env var → `CUDA_VISIBLE_DEVICES` (resolved before torch init via `_resolve_gpu_env()`); `threads_per_worker` caps BLAS/OMP/numba/torch threads.

### Naming & metadata conventions

- Plane output dirs are named `zplaneNN[_tpSTART-STOP][_suffix]` via `generate_plane_dirname()` using mbo_utilities dimension tags.
- **Frame-count aliases**: mbo_utilities and this package both write a set of synonym keys (`nframes`, `num_frames`, `num_timepoints`, `n_frames`, `T`, `nt`, `timepoints`) that must stay in lockstep — use `_set_frame_count_aliases()` to fan a count out to all of them.
- The canonical user-facing selection params are `timepoints` (1-based) and `planes` (1-based). `frames` / `frame_indices` (0-based) and `roi` / `num_frames` are **deprecated aliases** that emit `DeprecationWarning` (see `_resolve_timepoints`).

## Conventions & gotchas

- `torch` and `torchvision` are pinned to the **cu126** index in `pyproject.toml` (`[tool.uv.sources]`). They are direct deps — removing them reverts torch to the CPU build.
- This is an upstream **fork-tracking** project: when suite2p is bumped, the breakage is almost always in `_call_upstream_pipeline` / `db_settings.py` (new required `settings` keys, renamed functions, changed return shapes). The overlay-onto-`default_settings()` pattern is intentional insurance against new keys.
- suite2p / cellpose loggers have no handlers by default; `_attach_external_loggers()` wires them to stdout so registration/detection progress is visible.
