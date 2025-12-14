"""
Cellpose GUI wrapper with format conversion.

Provides a unified interface for editing Suite2p or Cellpose results
in the Cellpose GUI, with automatic format detection and conversion.

Usage:
    # as script
    uv run python -m lbm_suite2p_python.gui /path/to/results

    # or via lsp command
    lsp gui /path/to/results
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


def _patch_qt_compat():
    """Patch Qt5/Qt6 compatibility issues."""
    try:
        from qtpy.QtWidgets import QCheckBox
        if not hasattr(QCheckBox, "checkStateChanged"):
            QCheckBox.checkStateChanged = QCheckBox.stateChanged
    except ImportError:
        pass


def prepare_for_gui(source_path, output_dir=None):
    """
    Prepare any format for Cellpose GUI editing.

    Parameters
    ----------
    source_path : str or Path
        Suite2p plane directory or Cellpose output directory.
    output_dir : str or Path, optional
        Where to save GUI-ready files. Defaults to source_path/gui_edit.

    Returns
    -------
    dict
        Preparation result with seg_file path and source info.
    """
    from lbm_suite2p_python.conversion import (
        detect_format,
        export_for_gui,
        validate_format,
    )

    source_path = Path(source_path)
    fmt = detect_format(source_path)

    if fmt == "unknown":
        raise ValueError(f"Unknown format at {source_path}")

    validation = validate_format(source_path)
    print(f"Detected format: {fmt}")
    print(f"  ROIs: {validation['n_rois']}")
    print(f"  Shape: {validation['shape']}")

    if fmt in ("suite2p", "suite2p_minimal"):
        # export suite2p to cellpose GUI format
        if output_dir is None:
            output_dir = source_path / "gui_edit"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        seg_file = export_for_gui(source_path, output_dir / "cellpose_seg.npy")
        return {
            "seg_file": seg_file,
            "source_format": fmt,
            "source_path": source_path,
            "output_dir": output_dir,
        }

    elif fmt == "cellpose":
        # find or create seg file
        seg_files = list(source_path.glob("*_seg.npy")) + list(source_path.glob("cellpose_seg*.npy"))
        if seg_files:
            seg_file = seg_files[0]
        else:
            # need to create from masks
            from lbm_suite2p_python.conversion import suite2p_to_cellpose
            if output_dir is None:
                output_dir = source_path
            suite2p_to_cellpose(source_path, output_dir)
            seg_file = output_dir / "cellpose_seg.npy"

        return {
            "seg_file": seg_file,
            "source_format": fmt,
            "source_path": source_path,
            "output_dir": output_dir or source_path,
        }


def save_gui_edits(seg_file, original_path, output_dir=None, target_format=None):
    """
    Save GUI edits back to original format.

    Parameters
    ----------
    seg_file : str or Path
        Path to edited _seg.npy from GUI.
    original_path : str or Path
        Original source directory.
    output_dir : str or Path, optional
        Where to save. Defaults to original_path/edited.
    target_format : str, optional
        Target format: "suite2p", "cellpose", or "both". Defaults to original.

    Returns
    -------
    dict
        Save result with paths and change summary.
    """
    from lbm_suite2p_python.conversion import (
        detect_format,
        import_from_gui,
        cellpose_to_suite2p,
    )

    seg_file = Path(seg_file)
    original_path = Path(original_path)
    original_format = detect_format(original_path)

    if target_format is None:
        target_format = original_format if original_format != "unknown" else "both"

    results = {}

    if target_format in ("suite2p", "both", "suite2p_minimal"):
        suite2p_out = output_dir or original_path / "edited"
        result = import_from_gui(seg_file, original_path, suite2p_out)
        results["suite2p"] = result

    if target_format in ("cellpose", "both"):
        # save edited seg file
        cellpose_out = output_dir or original_path
        seg = np.load(seg_file, allow_pickle=True).item()
        np.save(cellpose_out / "cellpose_seg_edited.npy", seg)
        np.save(cellpose_out / "masks_edited.npy", seg["masks"])
        results["cellpose"] = {"path": str(cellpose_out)}

    return results


def launch_gui(seg_file=None, image=None, masks=None, wait=True):
    """
    Launch Cellpose GUI.

    Parameters
    ----------
    seg_file : str or Path, optional
        Path to _seg.npy to load.
    image : ndarray, optional
        Image to display.
    masks : ndarray, optional
        Masks to overlay.
    wait : bool
        If True, blocks until GUI closes.

    Returns
    -------
    bool
        True if GUI launched successfully.
    """
    _patch_qt_compat()

    try:
        from cellpose.gui import gui
    except ImportError:
        print("Error: Cellpose GUI not available.")
        print("Install with: pip install cellpose[gui]")
        return False

    if seg_file is not None:
        seg_file = str(Path(seg_file))

    # launch GUI
    try:
        if seg_file:
            print(f"Loading in GUI: {seg_file}")
            gui.run(image=seg_file)
        elif image is not None:
            gui.run(image=image, mask=masks)
        else:
            gui.run()
        return True
    except Exception as e:
        print(f"GUI error: {e}")
        return False


def run_gui_workflow(
    source_path,
    output_dir=None,
    save_on_exit=True,
    target_format=None,
):
    """
    Full GUI workflow: prepare → edit → save.

    Parameters
    ----------
    source_path : str or Path
        Path to Suite2p or Cellpose results.
    output_dir : str or Path, optional
        Where to save edits.
    save_on_exit : bool
        If True, prompts to save edits after GUI closes.
    target_format : str, optional
        Output format: "suite2p", "cellpose", or "both".

    Returns
    -------
    dict
        Workflow result with paths and changes.
    """
    source_path = Path(source_path)

    # prepare files for GUI
    print(f"\nPreparing for GUI: {source_path}")
    prep = prepare_for_gui(source_path, output_dir)
    seg_file = prep["seg_file"]

    print(f"\nLaunching Cellpose GUI...")
    print("  - Edit cells as needed")
    print("  - Save with Ctrl+S or File > Save")
    print("  - Close GUI when done")
    print()

    # launch GUI (blocks until closed)
    success = launch_gui(seg_file, wait=True)

    if not success:
        return {"success": False, "error": "GUI failed to launch"}

    # check if file was modified
    if save_on_exit:
        print("\nGUI closed. Checking for edits...")

        # reload and check for changes
        seg = np.load(seg_file, allow_pickle=True).item()
        n_cells = int(seg["masks"].max())
        n_manual = seg.get("ismanual", []).sum() if "ismanual" in seg else 0

        print(f"  Current cells: {n_cells}")
        if n_manual > 0:
            print(f"  Manual edits: {n_manual}")

        # save back to original format
        save_result = save_gui_edits(
            seg_file,
            prep["source_path"],
            output_dir,
            target_format or prep["source_format"],
        )

        return {
            "success": True,
            "prep": prep,
            "save_result": save_result,
            "n_cells": n_cells,
        }

    return {"success": True, "prep": prep}


def main():
    """CLI entry point for GUI wrapper."""
    parser = argparse.ArgumentParser(
        prog="lsp-gui",
        description="Cellpose GUI with Suite2p/Cellpose format support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lsp gui /path/to/suite2p/plane00     # edit suite2p results
  lsp gui /path/to/cellpose_output     # edit cellpose results
  lsp gui /path --output /path/edited  # save edits to specific dir
  lsp gui /path --format both          # save as both formats
        """,
    )

    parser.add_argument(
        "path",
        nargs="?",
        help="Path to Suite2p plane or Cellpose output directory",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for edited results",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["suite2p", "cellpose", "both"],
        help="Output format (default: same as input)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't prompt to save edits on exit",
    )
    parser.add_argument(
        "--seg-file",
        help="Load specific _seg.npy file directly",
    )

    args = parser.parse_args()

    # direct seg file loading
    if args.seg_file:
        print(f"Loading: {args.seg_file}")
        launch_gui(args.seg_file)
        return 0

    # no path - just launch empty GUI
    if not args.path:
        print("Launching Cellpose GUI...")
        launch_gui()
        return 0

    # full workflow
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {path}")
        return 1

    result = run_gui_workflow(
        path,
        output_dir=args.output,
        save_on_exit=not args.no_save,
        target_format=args.format,
    )

    if result.get("success"):
        print("\nDone!")
        return 0
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
