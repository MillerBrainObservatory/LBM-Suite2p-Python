#!/usr/bin/env python
"""
Re-plot figures for an existing Suite2p output folder.

Usage:
    uv run demos/scripts/replot_figures.py

Opens a file dialog to select a plane folder (containing ops.npy, stat.npy, etc.)
and regenerates all figures without re-running registration or segmentation.
"""

import tkinter as tk
from tkinter import filedialog
from pathlib import Path


def main():
    # Hide the root window
    root = tk.Tk()
    root.withdraw()

    # Open folder dialog
    folder = filedialog.askdirectory(
        title="Select Suite2p plane folder (containing ops.npy)"
    )

    if not folder:
        print("No folder selected, exiting.")
        return

    plane_dir = Path(folder)

    # Verify it's a valid plane folder
    if not (plane_dir / "ops.npy").exists():
        print(f"Error: {plane_dir} does not contain ops.npy")
        print("Please select a folder containing Suite2p output files.")
        return

    print(f"Re-plotting figures for: {plane_dir}")

    import lbm_suite2p_python as lsp

    # Regenerate all figures
    lsp.plot_zplane_figures(
        plane_dir=plane_dir,
        dff_percentile=20,
        dff_window_size=300,
        run_rastermap=False,  # Skip if already computed
    )

    print("Done!")


if __name__ == "__main__":
    main()
