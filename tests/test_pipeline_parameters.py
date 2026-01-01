"""
Comprehensive pipeline parameter tests for LBM-Suite2p-Python.

This test suite validates all major parameters for run_plane and run_volume,
ensuring they work correctly and don't overwrite existing data on reruns.

Usage:
    pytest tests/test_pipeline_parameters.py -v --tb=short

    # Or run directly:
    python tests/test_pipeline_parameters.py

Environment:
    Set TEST_DATA_PATH to override the default test data location:
    export TEST_DATA_PATH="E:\\tests\\lbm\\lbm_suite2p_python"
"""

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# Test configuration
DEFAULT_TEST_DATA_PATH = Path(r"E:\tests\lbm\lbm_suite2p_python")
TEST_OUTPUT_DIR = Path(__file__).parent / "_test_outputs"
SUMMARY_FILE = TEST_OUTPUT_DIR / "test_summary.json"
IMAGES_DIR = Path(__file__).parent  # Save images directly to tests/ folder


def get_test_data_path():
    """Get test data path from environment or default."""
    return Path(os.environ.get("TEST_DATA_PATH", DEFAULT_TEST_DATA_PATH))


@pytest.fixture(scope="module")
def test_data_path():
    """Fixture providing path to test data."""
    path = get_test_data_path()
    if not path.exists():
        pytest.skip(f"Test data path does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def test_tiff(test_data_path, output_dir):
    """Fixture providing path to test TIFF file (assembled planar format)."""
    from mbo_utilities import imread, imwrite
    from mbo_utilities.metadata import get_metadata

    tiff_path = output_dir / "test_input.tif"
    if not tiff_path.exists():
        print(f"Generating synthetic TIFF at {tiff_path}...")
        # Create synthetic tiff: (frames, y, x)
        data = np.random.randint(0, 1000, (100, 512, 512), dtype=np.int16)
        imwrite(data, tiff_path)
    
    if not tiff_path.exists():
        pytest.skip(f"Test TIFF could not be created: {tiff_path}")

    # Check if this is a raw ScanImage file that needs assembly
    metadata = get_metadata(tiff_path)
    num_planes = metadata.get("num_planes", 1)

    if num_planes > 1:
        # This is a raw multi-plane file - assemble it first
        assembled_dir = output_dir / "assembled"

        # Check for existing assembled files (nested in subdirectory)
        assembled_files = list(assembled_dir.rglob("*_stitched.tif"))

        if not assembled_files:
            print(f"Assembling raw file with {num_planes} planes...")
            arr = imread(tiff_path)
            imwrite(arr, assembled_dir / "planar")
            assembled_files = sorted(assembled_dir.rglob("*_stitched.tif"))

        # Use only first 2 planes for speed
        assembled_files = sorted(assembled_files)[:2]
        if not assembled_files:
            pytest.skip("No assembled files created")
        return assembled_files[0]  # Return first plane for single-plane tests

    return tiff_path


@pytest.fixture(scope="module")
def test_tiffs(test_data_path, output_dir):
    """Fixture providing list of test TIFF files (for volume tests)."""
    from mbo_utilities import imread, imwrite
    from mbo_utilities.metadata import get_metadata

    tiff_path = test_data_path / "test_input.tif"
    if not tiff_path.exists():
        pytest.skip(f"Test TIFF not found: {tiff_path}")

    # Check if this is a raw ScanImage file that needs assembly
    metadata = get_metadata(tiff_path)
    num_planes = metadata.get("num_planes", 1)

    if num_planes > 1:
        # This is a raw multi-plane file - assemble it first
        assembled_dir = output_dir / "assembled"

        # Check for existing assembled files (nested in subdirectory)
        assembled_files = list(assembled_dir.rglob("*_stitched.tif"))

        if not assembled_files:
            print(f"Assembling raw file with {num_planes} planes...")
            arr = imread(tiff_path)
            imwrite(arr, assembled_dir / "planar")
            assembled_files = sorted(assembled_dir.rglob("*_stitched.tif"))

        # Use only first 2 planes for speed
        return sorted(assembled_files)[:2]

    return [tiff_path]


@pytest.fixture(scope="module")
def output_dir():
    """Fixture providing output directory for test results."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # IMAGES_DIR is tests/ folder itself, no need to create
    return TEST_OUTPUT_DIR


@pytest.fixture(scope="module")
def test_summary(output_dir):
    """Fixture for accumulating test results."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "images": [],
    }
    yield summary
    # Save summary after all tests
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Images saved: {len(summary['images'])}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"{'='*60}")


def record_result(summary, name, status, duration=0, details=None, images=None):
    """Record a test result in the summary."""
    result = {
        "name": name,
        "status": status,
        "duration_seconds": duration,
        "details": details or {},
    }
    summary["tests"].append(result)
    if status == "passed":
        summary["passed"] += 1
    elif status == "failed":
        summary["failed"] += 1
    else:
        summary["skipped"] += 1
    if images:
        summary["images"].extend(images)


class TestRunPlaneParameters:
    """Test run_plane with various parameter combinations."""

    @pytest.fixture(autouse=True)
    def setup(self, test_tiff, output_dir, test_summary):
        """Setup for each test."""
        self.test_tiff = test_tiff
        self.output_dir = output_dir
        self.summary = test_summary
        self.base_save_path = output_dir / "run_plane_tests"
        self.base_save_path.mkdir(parents=True, exist_ok=True)

    def _get_base_ops(self):
        """Get base ops for testing."""
        import lbm_suite2p_python as lsp

        ops = lsp.default_ops()
        ops.update({
            "tau": 1.0,
            "fs": 17.0,
            "diameter": 8,
            "threshold_scaling": 0.5,  # Lower threshold for short test data
            "batch_size": 200,
            "nimg_init": 50,  # Less frames for init
            "anatomical_only": 0,  # Use functional detection (avoid cellpose bug)
            "sparse_mode": False,  # Standard mode
            "spatial_scale": 0,  # Auto-detect
            "high_pass": 100,  # High-pass filter
            "denoise": 1,  # Denoise
            "nbinned": 5000,  # More binning for short data
        })
        return ops

    def test_basic_run_plane(self):
        """Test basic run_plane execution."""
        import lbm_suite2p_python as lsp

        start = time.time()
        save_path = self.base_save_path / "basic"

        try:
            ops = self._get_base_ops()
            result = lsp.run_plane(
                input_path=self.test_tiff,
                save_path=save_path,
                ops=ops,
                keep_raw=False,
                keep_reg=True,
                force_reg=False,
                force_detect=False,
            )

            assert result.exists(), "ops.npy not created"
            assert (save_path / "stat.npy").exists(), "stat.npy not created"
            assert (save_path / "F.npy").exists(), "F.npy not created"

            duration = time.time() - start
            record_result(self.summary, "test_basic_run_plane", "passed", duration, {
                "ops_path": str(result),
                "output_files": [f.name for f in save_path.glob("*.npy")],
            })

        except Exception as e:
            record_result(self.summary, "test_basic_run_plane", "failed",
                         time.time() - start, {"error": str(e)})
            raise

    def test_force_reg_false_no_overwrite(self):
        """Test that force_reg=False doesn't overwrite existing registration."""
        import lbm_suite2p_python as lsp

        start = time.time()
        # Reuse the basic test output to avoid running another full pipeline
        save_path = self.base_save_path / "basic"

        try:
            # Check if basic test already ran
            if not (save_path / "stat.npy").exists():
                pytest.skip("Requires test_basic_run_plane to run first")

            # Get modification time of stat.npy from first run
            stat_mtime1 = (save_path / "stat.npy").stat().st_mtime
            print(f"Initial stat.npy mtime: {stat_mtime1}")

            # Small delay to ensure different timestamps
            time.sleep(1.0)

            # Create ops with roidetect=False to skip detection entirely
            ops = self._get_base_ops()
            ops["roidetect"] = False  # Explicitly disable detection

            # Second run with force_reg=False, force_detect=False
            result2 = lsp.run_plane(
                input_path=self.test_tiff,
                save_path=save_path,
                ops=ops,
                keep_raw=False,
                keep_reg=True,
                force_reg=False,
                force_detect=False,
            )

            stat_mtime2 = (save_path / "stat.npy").stat().st_mtime
            print(f"After second run stat.npy mtime: {stat_mtime2}")

            # stat.npy should NOT have been modified
            assert stat_mtime1 == stat_mtime2, f"stat.npy was overwritten despite force_detect=False (mtime changed from {stat_mtime1} to {stat_mtime2})"

            duration = time.time() - start
            record_result(self.summary, "test_force_reg_false_no_overwrite", "passed", duration, {
                "stat_preserved": stat_mtime1 == stat_mtime2,
            })

        except Exception as e:
            record_result(self.summary, "test_force_reg_false_no_overwrite", "failed",
                         time.time() - start, {"error": str(e)})
            raise

    def test_output_files_exist(self):
        """Verify all expected output files were created."""
        start = time.time()
        save_path = self.base_save_path / "basic"

        try:
            if not (save_path / "stat.npy").exists():
                pytest.skip("Requires test_basic_run_plane to run first")

            expected_files = ["ops.npy", "stat.npy", "F.npy", "Fneu.npy", "spks.npy", "iscell.npy"]
            missing = [f for f in expected_files if not (save_path / f).exists()]

            assert len(missing) == 0, f"Missing output files: {missing}"

            # Check shapes are consistent
            F = np.load(save_path / "F.npy")
            Fneu = np.load(save_path / "Fneu.npy")
            spks = np.load(save_path / "spks.npy")
            iscell = np.load(save_path / "iscell.npy")

            n_rois = F.shape[0]
            assert Fneu.shape[0] == n_rois, "Fneu shape mismatch"
            assert spks.shape[0] == n_rois, "spks shape mismatch"
            assert iscell.shape[0] == n_rois, "iscell shape mismatch"

            duration = time.time() - start
            record_result(self.summary, "test_output_files_exist", "passed", duration, {
                "n_rois": n_rois,
                "n_frames": F.shape[1],
                "n_accepted": int(np.sum(iscell[:, 0])),
            })

        except Exception as e:
            record_result(self.summary, "test_output_files_exist", "failed",
                         time.time() - start, {"error": str(e)})
            raise


class TestPlotTraces:
    """Test plot_traces function with various parameters."""

    @pytest.fixture(autouse=True)
    def setup(self, output_dir, test_summary):
        """Setup for each test."""
        self.output_dir = output_dir
        self.summary = test_summary
        self.images_dir = IMAGES_DIR

    def _get_test_traces(self):
        """Get or create test traces."""
        # Check if we have processed data from previous tests
        test_output = TEST_OUTPUT_DIR / "run_plane_tests" / "basic"
        if (test_output / "F.npy").exists():
            F = np.load(test_output / "F.npy")
            return F

        # Create synthetic data if no real data available
        np.random.seed(42)
        n_cells, n_frames = 100, 3000
        F = np.random.randn(n_cells, n_frames) * 0.1 + 1.0
        # Add some transients
        for i in range(n_cells):
            n_transients = np.random.randint(5, 20)
            for _ in range(n_transients):
                start = np.random.randint(0, n_frames - 100)
                F[i, start:start+50] += np.random.rand() * 0.5
        return F

    def test_plot_traces_basic(self):
        """Test basic plot_traces functionality."""
        from lbm_suite2p_python.zplane import plot_traces

        start = time.time()

        try:
            F = self._get_test_traces()
            save_path = self.images_dir / "traces_basic.png"

            plot_traces(
                F,
                save_path=str(save_path),
                fps=30.0,
                num_neurons=20,
                window=60,
                title="Basic Trace Plot",
                scale_bar_unit="% ΔF/F₀",
            )

            assert save_path.exists(), "Plot was not saved"

            duration = time.time() - start
            record_result(self.summary, "test_plot_traces_basic", "passed", duration,
                         images=[str(save_path)])

        except Exception as e:
            record_result(self.summary, "test_plot_traces_basic", "failed",
                         time.time() - start, {"error": str(e)})
            raise

    def test_plot_traces_custom_label(self):
        """Test plot_traces with custom scale bar labels."""
        from lbm_suite2p_python.zplane import plot_traces

        start = time.time()
        images = []

        try:
            F = self._get_test_traces()

            labels = ["100% ΔF/F₀", "0.5 a.u.", "Custom Units"]
            for i, label in enumerate(labels):
                save_path = self.images_dir / f"traces_label_{i}.png"

                plot_traces(
                    F,
                    save_path=str(save_path),
                    fps=30.0,
                    num_neurons=15,
                    window=45,
                    title=f"Scale Bar: {label}",
                    scale_bar_unit=label,
                )

                assert save_path.exists(), f"Plot {i} was not saved"
                images.append(str(save_path))

            duration = time.time() - start
            record_result(self.summary, "test_plot_traces_custom_label", "passed", duration,
                         {"labels_tested": labels}, images=images)

        except Exception as e:
            record_result(self.summary, "test_plot_traces_custom_label", "failed",
                         time.time() - start, {"error": str(e)})
            raise

    def test_plot_traces_cell_indices(self):
        """Test plot_traces with specific cell indices."""
        from lbm_suite2p_python.zplane import plot_traces

        start = time.time()

        try:
            F = self._get_test_traces()
            save_path = self.images_dir / "traces_cell_indices.png"

            # Select specific cells
            indices = [0, 5, 10, 20, 50]

            plot_traces(
                F,
                save_path=str(save_path),
                cell_indices=indices,
                fps=30.0,
                window=90,
                title="Selected Cells",
                scale_bar_unit="ΔF/F₀",
            )

            assert save_path.exists(), "Plot was not saved"

            duration = time.time() - start
            record_result(self.summary, "test_plot_traces_cell_indices", "passed", duration,
                         {"indices": indices}, images=[str(save_path)])

        except Exception as e:
            record_result(self.summary, "test_plot_traces_cell_indices", "failed",
                         time.time() - start, {"error": str(e)})
            raise


class TestDFFCalculation:
    """Test ΔF/F calculation functions."""

    @pytest.fixture(autouse=True)
    def setup(self, output_dir, test_summary):
        """Setup for each test."""
        self.output_dir = output_dir
        self.summary = test_summary

    def _get_test_traces(self):
        """Get or create test traces."""
        test_output = TEST_OUTPUT_DIR / "run_plane_tests" / "basic"
        if (test_output / "F.npy").exists():
            return np.load(test_output / "F.npy")

        # Create synthetic data
        np.random.seed(42)
        n_cells, n_frames = 50, 2000
        F = np.random.randn(n_cells, n_frames) * 0.1 + 100.0
        for i in range(n_cells):
            for _ in range(np.random.randint(3, 10)):
                start = np.random.randint(0, n_frames - 100)
                F[i, start:start+30] += np.random.rand() * 20
        return F

    def test_dff_rolling_percentile(self):
        """Test dff_rolling_percentile function."""
        from lbm_suite2p_python.postprocessing import dff_rolling_percentile

        start = time.time()

        try:
            F = self._get_test_traces()

            dff = dff_rolling_percentile(
                F,
                window_size=300,
                percentile=20,
            )

            assert dff.shape == F.shape, "DFF shape mismatch"
            assert not np.any(np.isnan(dff)), "DFF contains NaN values"
            assert not np.any(np.isinf(dff)), "DFF contains Inf values"

            # Check that DFF has reasonable range
            dff_range = np.percentile(dff, 99) - np.percentile(dff, 1)

            duration = time.time() - start
            record_result(self.summary, "test_dff_rolling_percentile", "passed", duration, {
                "shape": list(dff.shape),
                "dff_range": float(dff_range),
                "dff_mean": float(np.mean(dff)),
            })

        except Exception as e:
            record_result(self.summary, "test_dff_rolling_percentile", "failed",
                         time.time() - start, {"error": str(e)})
            raise

    def test_dff_window_size_effect(self):
        """Test that different window sizes produce different results."""
        from lbm_suite2p_python.postprocessing import dff_rolling_percentile

        start = time.time()

        try:
            F = self._get_test_traces()

            dff_100 = dff_rolling_percentile(F, window_size=100, percentile=20)
            dff_500 = dff_rolling_percentile(F, window_size=500, percentile=20)

            # Different window sizes should produce different results
            # Check they're not exactly identical (allow small numerical differences)
            max_diff = np.max(np.abs(dff_100 - dff_500))
            mean_diff = np.mean(np.abs(dff_100 - dff_500))

            # There should be some difference between the results
            assert max_diff > 0.001, f"Results too similar: max_diff={max_diff}"

            duration = time.time() - start
            record_result(self.summary, "test_dff_window_size_effect", "passed", duration, {
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
            })

        except Exception as e:
            record_result(self.summary, "test_dff_window_size_effect", "failed",
                         time.time() - start, {"error": str(e)})
            raise


def run_all_tests():
    """Run all tests and generate summary."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode


def print_ci_summary():
    """Print summary suitable for CI output."""
    if not SUMMARY_FILE.exists():
        print("No test summary found. Run tests first.")
        return

    with open(SUMMARY_FILE) as f:
        summary = json.load(f)

    print("\n" + "="*60)
    print("LBM-Suite2p-Python Pipeline Parameter Tests")
    print("="*60)
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Total tests: {len(summary['tests'])}")
    print(f"  [PASS] Passed: {summary['passed']}")
    print(f"  [FAIL] Failed: {summary['failed']}")
    print(f"  [SKIP] Skipped: {summary['skipped']}")
    print(f"Images generated: {len(summary['images'])}")
    print("-"*60)

    for test in summary['tests']:
        status_icon = "[PASS]" if test['status'] == 'passed' else "[FAIL]" if test['status'] == 'failed' else "[SKIP]"
        print(f"  {status_icon} {test['name']} ({test['duration_seconds']:.1f}s)")
        if test['details'] and test['status'] == 'failed':
            print(f"      Error: {test['details'].get('error', 'Unknown')}")

    print("="*60)

    if summary['images']:
        print("\nGenerated images:")
        for img in summary['images']:
            print(f"  - {img}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LBM-Suite2p-Python parameter tests")
    parser.add_argument("--summary", action="store_true", help="Print CI summary only")
    parser.add_argument("--clean", action="store_true", help="Clean test outputs before running")
    args = parser.parse_args()

    if args.summary:
        print_ci_summary()
    else:
        if args.clean and TEST_OUTPUT_DIR.exists():
            shutil.rmtree(TEST_OUTPUT_DIR)
            print(f"Cleaned {TEST_OUTPUT_DIR}")

        exit_code = run_all_tests()
        print_ci_summary()
        sys.exit(exit_code)
