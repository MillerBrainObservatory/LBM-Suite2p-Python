# cell filtering bugs

## context

suite2p detects ROIs and classifies them via its built-in classifier into
`iscell.npy` (shape `(n_rois, 2)` where col 0 = classification 0/1, col 1 = probability).
example: 119 ROIs detected, 9 approved, 110 rejected.

our pipeline then optionally:
1. accepts all cells (`accept_all_cells=True` sets col 0 to 1 for all ROIs)
2. applies custom filters (diameter, area, eccentricity)
3. saves filtered iscell back to disk
4. generates diagnostic plots

the user sees `13_filtered_cells.png` showing "11 cells, 1 removed" when
suite2p actually rejected 110 cells. multiple bugs contribute to this.

---

## bug 1: `iscell_original` is loaded AFTER `accept_all_cells` overwrites it

**file:** `run_lsp.py:1495-1509`

```python
# line 1500: overwrites iscell.npy on disk — all 119 ROIs now marked as accepted
iscell[:, 0] = 1
np.save(iscell_file, iscell)

# line 1509: loads "original" from disk — but it's already been overwritten!
iscell_original = np.load(plane_dir / "iscell.npy", allow_pickle=True)
```

`iscell_original` now has all 119 ROIs marked as cells. so when filtering
removes only a few (e.g. 1 by diameter), the plot shows "119 -> 118 cells
(1 removed)" instead of reflecting suite2p's 110 rejections.

**fix:** load `iscell_original` BEFORE the `accept_all_cells` block, or load
from the backup `iscell_suite2p.npy`.

---

## bug 2: backup `iscell_suite2p.npy` is saved AFTER overwrite

**file:** `run_lsp.py:1499-1500`

```python
np.save(plane_dir / "iscell_suite2p.npy", iscell)  # save backup
iscell[:, 0] = 1  # then overwrite in-place
```

`iscell` is a numpy array — the `iscell[:, 0] = 1` on line 1500 modifies
the same object that was saved on line 1499. numpy `save` writes to disk
immediately so the backup IS correct (saved before mutation). however this
is fragile and confusing — a simple reorder would break it.

**fix:** explicitly copy before mutation: `np.save(..., iscell.copy())` or
save the backup first, then reload it separately.

---

## bug 3: plot shows OUR filter results, not suite2p's classification

**file:** `zplane.py:3854-3860`

the `13_filtered_cells.png` plot compares `iscell_original` vs `iscell_filtered`.
but because of bug 1, `iscell_original` has all ROIs accepted (post-`accept_all_cells`).
so the plot only shows what OUR filters removed, not what suite2p rejected.

the user expects this plot to show suite2p's 110 rejected cells in red.
instead it shows only 1 cell removed (from our diameter filter).

**fix:** the plot should use the true suite2p classification as `iscell_original`
(loaded before `accept_all_cells`), OR the plot should be split into two:
one for suite2p classification, one for our custom filters.

---

## bug 4: `_save_filtered_iscell` loses probability data in fallback path

**file:** `postprocessing.py:87-92`

```python
iscell_2d = np.column_stack([
    iscell_filtered.astype(float),
    iscell_filtered.astype(float)  # prob=1 for cells, 0 for non-cells
])
```

when `iscell_original` is not available, the probability column is set to
the same value as the classification column. all accepted cells get prob=1.0,
all rejected get prob=0.0. this loses suite2p's actual classifier confidence.

**fix:** preserve the original probability column. this fallback should load
from `iscell_suite2p.npy` backup if available.

---

## bug 5: `apply_filters` doesn't pass `iscell_original` to `_save_filtered_iscell`

**file:** `postprocessing.py:614`

```python
_save_filtered_iscell(plane_dir, iscell_current)
# missing: iscell_original=iscell (the raw 2D array)
```

`_save_filtered_iscell` accepts an `iscell_original` parameter to preserve
the probability column, but `apply_filters` doesn't pass it. this means
`_save_filtered_iscell` reloads from disk — which at this point is already
the `accept_all_cells`-modified version (all probs are the same).

**fix:** pass the original 2D iscell array through to `_save_filtered_iscell`.

---

## bug 6: `plot_filtered_cells` reloads from disk unnecessarily

**file:** `zplane.py:3836`

```python
res = load_planar_results(plane_dir)  # reloads iscell.npy from disk
```

the function receives `iscell_original` and `iscell_filtered` as parameters,
but also loads results from disk. `iscell.npy` on disk has already been
overwritten by `apply_filters(save=True)` at this point. the function uses
the passed-in parameters for cell counting (correct), but loads `stat` from
`res` which is fine. however if anyone changes the code to use `res["iscell"]`
instead of the parameters, counts will be wrong.

**severity:** low (currently not causing incorrect behavior, but fragile).

---

## summary

| bug | file | severity | core issue |
|-----|------|----------|------------|
| 1 | run_lsp.py:1509 | critical | iscell_original loaded after accept_all_cells overwrites it |
| 2 | run_lsp.py:1499 | low | backup save relies on mutation ordering |
| 3 | zplane.py:3854 | critical | plot shows our filters, not suite2p rejections |
| 4 | postprocessing.py:87 | medium | probability column lost in fallback |
| 5 | postprocessing.py:614 | medium | iscell_original not passed to save function |
| 6 | zplane.py:3836 | low | unnecessary disk reload, fragile |

bugs 1 and 3 together explain the user-reported issue: "119 cells found,
9 approved by suite2p, but plot shows 11 cells with 1 removed."
