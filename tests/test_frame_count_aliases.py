"""
Frame-count alias consistency tests.

mbo_utilities writes ops.npy with seven aliases for "number of timepoints
in this output" — every reader (mbo, lbm_suite2p_python, suite2p, downstream
analysis code) reads a different one. Whenever lbm_suite2p_python updates
the count (e.g. dual-channel trim), it must fan out to ALL aliases or
downstream code gets contradictory answers from the same ops dict.

These tests pin that contract using small synthetic binaries — no external
test data required.
"""

import numpy as np
import pytest

from lbm_suite2p_python.run_lsp import (
    _FRAME_COUNT_ALIASES,
    _set_frame_count_aliases,
)


# minimum set of aliases that must all agree after any frame-count update
EXPECTED_ALIASES = (
    "nframes",
    "num_frames",
    "num_timepoints",
    "n_frames",
    "T",
    "nt",
    "timepoints",
)


class TestSetFrameCountAliases:
    """Direct unit tests for the helper."""

    def test_sets_every_alias(self):
        ops = {}
        _set_frame_count_aliases(ops, 700)
        for key in EXPECTED_ALIASES:
            assert ops[key] == 700, f"{key} not set"

    def test_overwrites_existing(self):
        """Existing values must be replaced, not preserved."""
        ops = {key: 1574 for key in EXPECTED_ALIASES}
        _set_frame_count_aliases(ops, 700)
        for key in EXPECTED_ALIASES:
            assert ops[key] == 700, f"{key} stuck at old value"

    def test_constant_lists_canonical_aliases(self):
        """The module-level constant must include every key downstream
        readers actually look at. If a new alias is added to mbo, it
        belongs here too.
        """
        for key in EXPECTED_ALIASES:
            assert key in _FRAME_COUNT_ALIASES, \
                f"{key} missing from _FRAME_COUNT_ALIASES"


class TestRunPlaneBinAliasConsistency:
    """Integration test: write a tiny synthetic plane dir, run
    `run_plane_bin` (which calls suite2p's pipeline internally), and
    assert every alias in the resulting ops.npy is consistent — both
    with each other and with the actual binary file size.

    The dual-channel trim case is the regression target: chan1 has more
    frames than chan2, lbm_suite2p_python clamps n_align = min(...), and
    used to update only `nframes` while leaving 6 other aliases at the
    chan1 source value.
    """

    def _build_plane(self, plane_dir, nt_chan1, nt_chan2=None,
                     ly=32, lx=32):
        """Write data_raw.bin (+ optional data_chan2.bin) and an
        ops.npy with all aliases at the chan1 count.
        """
        from suite2p.default_ops import default_ops

        rng = np.random.RandomState(0)
        chan1 = rng.randint(0, 4096, size=(nt_chan1, ly, lx),
                            dtype=np.int16)
        chan1.tofile(plane_dir / "data_raw.bin")

        if nt_chan2 is not None:
            chan2 = rng.randint(0, 4096, size=(nt_chan2, ly, lx),
                                dtype=np.int16)
            chan2.tofile(plane_dir / "data_chan2.bin")

        ops = default_ops()
        ops.update({
            "Ly": ly,
            "Lx": lx,
            "fs": 30.0,
            "dz": 15.0,  # sentinel — must survive
            "raw_file": str(plane_dir / "data_raw.bin"),
            "reg_file": str(plane_dir / "data.bin"),
            "ops_path": str(plane_dir / "ops.npy"),
            "save_path": str(plane_dir),
            # skip detection so the test stays fast and doesn't depend
            # on suite2p classifier files
            "roidetect": False,
            "do_registration": 1,
            # all 7 aliases at the chan1 count to start
            "nframes": nt_chan1,
            "nframes_chan1": nt_chan1,
            "num_frames": nt_chan1,
            "num_timepoints": nt_chan1,
            "n_frames": nt_chan1,
            "T": nt_chan1,
            "nt": nt_chan1,
            "timepoints": nt_chan1,
        })
        if nt_chan2 is not None:
            ops["chan2_file"] = str(plane_dir / "data_chan2.bin")
            ops["nframes_chan2"] = nt_chan2

        np.save(plane_dir / "ops.npy", ops)
        return ops

    def _run_and_reload(self, plane_dir, ops):
        from lbm_suite2p_python.run_lsp import run_plane_bin
        run_plane_bin(ops)
        return np.load(plane_dir / "ops.npy", allow_pickle=True).item()

    def test_single_channel_preserves_aliases(self, tmp_path):
        """No trimming → all aliases unchanged from input."""
        plane = tmp_path / "plane01"
        plane.mkdir()
        ops = self._build_plane(plane, nt_chan1=100)
        after = self._run_and_reload(plane, ops)

        for key in EXPECTED_ALIASES:
            assert after.get(key) == 100, \
                f"{key}: expected 100, got {after.get(key)!r}"
        # sentinel survived
        assert after.get("dz") == 15.0

    def test_dual_channel_trim_fans_out(self, tmp_path):
        """chan1=100, chan2=80 → all aliases must report 80, not 100."""
        plane = tmp_path / "plane01"
        plane.mkdir()
        ops = self._build_plane(plane, nt_chan1=100, nt_chan2=80)
        after = self._run_and_reload(plane, ops)

        for key in EXPECTED_ALIASES:
            assert after.get(key) == 80, \
                f"{key}: expected 80 (trimmed), got {after.get(key)!r}"
        # both channel-specific keys also clamped
        assert after.get("nframes_chan1") == 80
        assert after.get("nframes_chan2") == 80

    def test_aliases_internally_consistent(self, tmp_path):
        """Whatever the value, all 7 aliases must equal each other."""
        plane = tmp_path / "plane01"
        plane.mkdir()
        ops = self._build_plane(plane, nt_chan1=100, nt_chan2=80)
        after = self._run_and_reload(plane, ops)

        values = {k: after.get(k) for k in EXPECTED_ALIASES}
        unique = set(values.values())
        assert len(unique) == 1, \
            f"aliases disagree: {values}"

    def test_dz_survives_pipeline(self, tmp_path):
        """User-set dz must not be lost when suite2p mutates ops."""
        plane = tmp_path / "plane01"
        plane.mkdir()
        ops = self._build_plane(plane, nt_chan1=100)
        after = self._run_and_reload(plane, ops)
        assert after.get("dz") == 15.0


# strided fs/dz scaling for run_volume + run_plane

class TestStridedSelections:
    """`pipeline`/`run_volume`/`run_plane` accept `frame_indices` for
    explicit (possibly strided) frame selection. The implicit T-stride
    must scale `fs` (via mbo's writer), and the implicit Z-stride from
    the `planes` selection must scale `dz` (via lbm's helper, since the
    full plane list isn't visible at the writer layer).

    Regression: prior to this work, the only timepoint control was
    `num_timepoints` (truncation only — no stride awareness), and `dz`
    in per-plane ops.npy was always the source value regardless of
    plane stride. So every-other-timepoint produced ops.npy with the
    raw source `fs`, and every-other-plane produced ops.npy with the
    raw source `dz`. Both wrong by a factor of 2.
    """

    def test_helper_skips_dz_when_no_plane_indices(self):
        """When called with frame_indices but no plane_indices (the
        per-plane layer in run_plane), the helper must NOT touch dz —
        otherwise it clobbers a correctly-scaled value already set by
        the outer run_volume call.
        """
        from lbm_suite2p_python.run_lsp import _apply_reactive_metadata
        ops = {"dz": 20.0}  # already-scaled from outer layer
        _apply_reactive_metadata(
            ops=ops,
            source_metadata={"fs": 30.0, "dz": 10.0},
            source_shape=(200, 1, 8, 32, 32),
            frame_indices=list(range(0, 200, 2)),
            plane_indices=None,  # no plane context here
        )
        assert ops["dz"] == 20.0, \
            "helper must NOT clobber dz when plane_indices is None"

    def test_helper_scales_dz_with_plane_indices(self):
        """With plane_indices supplied, the helper scales dz correctly."""
        from lbm_suite2p_python.run_lsp import _apply_reactive_metadata
        ops = {}
        _apply_reactive_metadata(
            ops=ops,
            source_metadata={"fs": 30.0, "dz": 10.0},
            source_shape=(200, 1, 8, 32, 32),
            frame_indices=list(range(0, 200, 2)),
            plane_indices=[0, 2, 4, 6],
        )
        assert ops["dz"] == 20.0, f"expected dz=20, got {ops.get('dz')}"

    def test_helper_does_not_scale_fs(self):
        """fs scaling is owned by mbo's writer (which sees `frames=`).
        The lbm helper must NOT touch fs, otherwise it gets double-
        scaled when the writer also runs.
        """
        from lbm_suite2p_python.run_lsp import _apply_reactive_metadata
        ops = {"fs": 30.0}
        _apply_reactive_metadata(
            ops=ops,
            source_metadata={"fs": 30.0, "dz": 10.0},
            source_shape=(200, 1, 8, 32, 32),
            frame_indices=list(range(0, 200, 2)),
            plane_indices=[0, 2, 4, 6],
        )
        assert ops["fs"] == 30.0, "helper must NOT touch fs"

    def test_run_plane_strided_frames(self, tmp_path):
        """Direct `run_plane` with `frame_indices` writes the strided
        binary AND the per-plane ops.npy carries the scaled `fs`.
        """
        import numpy as np
        from mbo_utilities.arrays import NumpyArray
        from lbm_suite2p_python import run_plane

        nt = 200
        rng = np.random.RandomState(0)
        data = rng.randint(0, 4096, size=(nt, 1, 1, 32, 32), dtype=np.int16)
        arr = NumpyArray(data)
        arr.metadata = {"fs": 30.0, "dz": 10.0, "dx": 0.5, "dy": 0.5}

        ops_path = run_plane(
            input_data=arr,
            save_path=tmp_path / "out",
            ops={"roidetect": False, "do_registration": 1},
            frame_indices=list(range(0, nt, 2)),  # every other frame
            plane_name="plane01",
            keep_raw=True,
        )
        ops = np.load(ops_path, allow_pickle=True).item()
        assert ops.get("fs") == 15.0, f"fs: expected 15.0, got {ops.get('fs')}"
        assert ops.get("nframes") == 100
        # binary file size matches the strided count
        bin_files = list(ops_path.parent.glob("data_raw.bin"))
        assert bin_files
        actual = bin_files[0].stat().st_size // (32 * 32 * 2)
        assert actual == 100, f"binary should have 100 frames, got {actual}"

    def test_run_volume_strided_planes_and_frames(self, tmp_path):
        """`run_volume` with both `planes=[1,3,5,7]` (every-other-plane)
        and `frame_indices=range(0,N,2)` (every-other-frame). Each
        per-plane ops.npy must have BOTH `fs` halved AND `dz` doubled.
        """
        import numpy as np
        import tifffile
        from mbo_utilities import imread
        from lbm_suite2p_python import run_volume

        nt, nz = 200, 8
        rng = np.random.RandomState(0)
        vol_dir = tmp_path / "src"
        vol_dir.mkdir()
        for z in range(nz):
            tifffile.imwrite(
                vol_dir / f"plane{z+1:02d}.tif",
                rng.randint(0, 4096, (nt, 32, 32), dtype=np.uint16),
            )
        arr = imread(vol_dir)
        arr.metadata = {**arr.metadata, "fs": 30.0, "dz": 10.0,
                        "dx": 0.5, "dy": 0.5}

        ops_paths = run_volume(
            input_data=arr,
            save_path=tmp_path / "out",
            ops={"roidetect": False, "do_registration": 1},
            planes=[1, 3, 5, 7],            # every other plane
            frame_indices=list(range(0, nt, 2)),  # every other frame
            keep_raw=True,
        )
        assert len(ops_paths) == 4

        for op in ops_paths:
            ops = np.load(op, allow_pickle=True).item()
            assert ops.get("fs") == 15.0, \
                f"plane {ops.get('plane')}: fs={ops.get('fs')}, expected 15.0"
            assert ops.get("dz") == 20.0, \
                f"plane {ops.get('plane')}: dz={ops.get('dz')}, expected 20.0"
            assert ops.get("nframes") == 100, \
                f"plane {ops.get('plane')}: nframes={ops.get('nframes')}, expected 100"
            # binary file matches
            bin_files = list(op.parent.glob("data_raw.bin"))
            assert bin_files
            actual = bin_files[0].stat().st_size // (32 * 32 * 2)
            assert actual == 100, \
                f"plane {ops.get('plane')}: bin has {actual} frames"
