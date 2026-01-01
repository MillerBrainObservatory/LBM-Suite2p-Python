"""
Cellpose GUI wrapper with format conversion.

Provides a unified interface for editing Suite2p or Cellpose results
in the Cellpose GUI, with automatic format detection and conversion.

Usage:
    uv run cellpose-gui /path/to/results
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from mbo_utilities.util import load_npy


def detect_format(path: str | Path) -> str:
    """
    Detect whether path contains suite2p or cellpose results.

    Returns 'suite2p', 'cellpose', or 'unknown'.

    Note: This is a simplified wrapper around conversion.detect_format
    that also handles _seg.npy files directly.
    """
    from lbm_suite2p_python.conversion import detect_format as _detect_format

    path = Path(path)
    if not path.exists():
        return "unknown"

    # Handle _seg.npy files directly (cellpose GUI format)
    if path.is_file() and path.name.endswith("_seg.npy"):
        return "cellpose"

    # Delegate to conversion module
    result = _detect_format(path)

    # Normalize suite2p_minimal to suite2p for GUI purposes
    if result == "suite2p_minimal":
        return "suite2p"

    return result


def suite2p_to_seg(
    suite2p_dir: str | Path,
    output_dir: str | Path = None,
) -> Path:
    """
    Convert suite2p results to cellpose _seg.npy format.

    Parameters
    ----------
    suite2p_dir : Path
        Directory containing stat.npy, iscell.npy, ops.npy.
    output_dir : Path, optional
        Where to save the _seg.npy. Defaults to suite2p_dir.

    Returns
    -------
    Path
        Path to the created _seg.npy file.
    """
    from lbm_suite2p_python.cellpose import stat_to_masks, _masks_to_outlines

    suite2p_dir = Path(suite2p_dir)
    output_dir = Path(output_dir) if output_dir else suite2p_dir

    # load suite2p files
    stat = np.load(suite2p_dir / "stat.npy", allow_pickle=True)
    iscell = np.load(suite2p_dir / "iscell.npy") if (suite2p_dir / "iscell.npy").exists() else None
    ops = load_npy(suite2p_dir / "ops.npy").item()

    # get image dimensions
    Ly = ops.get("Ly", 512)
    Lx = ops.get("Lx", 512)

    # filter by iscell if available
    if iscell is not None:
        accepted = iscell[:, 0].astype(bool)
        stat = stat[accepted]

    # convert to masks
    masks = stat_to_masks(stat, (Ly, Lx))

    # get mean image for display
    img = ops.get("meanImg", ops.get("max_proj", np.zeros((Ly, Lx))))

    # create seg data
    n_rois = int(masks.max())
    seg_data = {
        "img": img.astype(np.float32),
        "masks": masks.astype(np.uint32),
        "outlines": _masks_to_outlines(masks),
        "chan_choose": [0, 0],
        "ismanual": np.zeros(n_rois, dtype=bool),
        "filename": str(output_dir / "meanImg.tif"),
        "flows": None,
        "est_diam": None,
    }

    # save image for gui
    try:
        import tifffile
        tifffile.imwrite(output_dir / "meanImg.tif", img.astype(np.float32))
    except ImportError:
        pass

    # save seg file
    seg_file = output_dir / "suite2p_seg.npy"
    np.save(seg_file, seg_data, allow_pickle=True)

    return seg_file


def seg_to_suite2p(
    seg_path: str | Path,
    suite2p_dir: str | Path,
    original_ops: dict = None,
):
    """
    Convert cellpose _seg.npy back to suite2p format.

    Parameters
    ----------
    seg_path : Path
        Path to _seg.npy file.
    suite2p_dir : Path
        Directory to save suite2p files.
    original_ops : dict, optional
        Original ops.npy contents to preserve.
    """
    from lbm_suite2p_python.cellpose import masks_to_stat

    seg_path = Path(seg_path)
    suite2p_dir = Path(suite2p_dir)

    # load seg data
    seg_data = np.load(seg_path, allow_pickle=True).item()
    masks = seg_data["masks"]
    img = seg_data.get("img")

    # convert masks to stat
    stat = masks_to_stat(masks, img)

    # create iscell (all accepted after gui editing)
    n_rois = len(stat)
    iscell = np.ones((n_rois, 2), dtype=np.float32)

    # save
    np.save(suite2p_dir / "stat.npy", stat)
    np.save(suite2p_dir / "iscell.npy", iscell)

    # update ops if provided
    if original_ops:
        original_ops["nrois"] = n_rois
        np.save(suite2p_dir / "ops.npy", original_ops)

    print(f"saved {n_rois} rois to {suite2p_dir}")


def launch_gui(seg_path: str | Path = None, image_path: str | Path = None):
    """
    Launch cellpose gui with optional pre-loaded results.

    Parameters
    ----------
    seg_path : Path, optional
        Path to _seg.npy file to load.
    image_path : Path, optional
        Path to image file to load.
    """
    # patch QCheckBox for Qt5/Qt6 compatibility
    try:
        from qtpy.QtWidgets import QCheckBox
        if not hasattr(QCheckBox, 'checkStateChanged'):
            QCheckBox.checkStateChanged = QCheckBox.stateChanged
    except ImportError:
        pass

    from cellpose.gui import gui

    if seg_path:
        seg_path = Path(seg_path)
        seg_data = np.load(seg_path, allow_pickle=True).item()
        img_file = seg_data.get("filename")
        if img_file and Path(img_file).exists():
            gui.run(image=str(img_file))
        elif image_path and Path(image_path).exists():
            gui.run(image=str(image_path))
        else:
            print(f"warning: image file not found, opening empty gui")
            gui.run()
    elif image_path:
        gui.run(image=str(image_path))
    else:
        gui.run()


def run_gui_workflow(
    path: str | Path = None,
    output_dir: str | Path = None,
    output_format: str = None,
    save_on_exit: bool = True,
    seg_file: str | Path = None,
):
    """
    Run the full gui workflow with format detection and conversion.

    Parameters
    ----------
    path : Path, optional
        Path to suite2p plane dir or cellpose output dir.
    output_dir : Path, optional
        Where to save edited results. Defaults to input location.
    output_format : str, optional
        Output format: 'suite2p', 'cellpose', or 'both'.
        Defaults to same as input.
    save_on_exit : bool
        Whether to prompt for save on exit.
    seg_file : Path, optional
        Specific _seg.npy file to load directly.
    """
    if seg_file:
        # load specific seg file directly
        seg_path = Path(seg_file)
        launch_gui(seg_path)
        return

    if path is None:
        # launch empty gui
        launch_gui()
        return

    path = Path(path)
    if not path.exists():
        print(f"error: path not found: {path}")
        return

    # detect format
    fmt = detect_format(path)
    print(f"detected format: {fmt}")

    if fmt == "unknown":
        print("could not detect format, launching empty gui")
        launch_gui()
        return

    output_dir = Path(output_dir) if output_dir else path
    output_format = output_format or fmt

    if fmt == "suite2p":
        # convert to seg format for gui
        print("converting suite2p to cellpose format...")

        # create temp dir for gui files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # convert
            seg_path = suite2p_to_seg(path, tmpdir)
            print(f"created: {seg_path}")

            # store original ops for later
            ops = load_npy(path / "ops.npy").item()

            # launch gui
            print("\nlaunching cellpose gui...")
            print("edit cells, then save and close the gui")
            launch_gui(seg_path)

            if save_on_exit:
                # check if user saved changes (seg file modified)
                if seg_path.exists():
                    print("\nsaving changes...")
                    if output_format in ("suite2p", "both"):
                        seg_to_suite2p(seg_path, output_dir, ops)
                    if output_format in ("cellpose", "both"):
                        # copy seg file to output
                        import shutil
                        out_seg = output_dir / "cellpose_seg.npy"
                        shutil.copy(seg_path, out_seg)
                        print(f"saved: {out_seg}")

    elif fmt == "cellpose":
        # find seg file
        if path.is_file() and path.name.endswith("_seg.npy"):
            seg_path = path
        else:
            seg_files = list(path.glob("*_seg.npy"))
            if seg_files:
                seg_path = seg_files[0]
            else:
                print("no _seg.npy file found")
                launch_gui()
                return

        print(f"loading: {seg_path}")
        launch_gui(seg_path)

        if save_on_exit and output_format == "suite2p":
            print("\nconverting to suite2p format...")
            seg_to_suite2p(seg_path, output_dir)


def main():
    """CLI entry point for GUI wrapper."""
    parser = argparse.ArgumentParser(
        prog="cellpose-gui",
        description="Cellpose GUI with Suite2p/Cellpose format support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cellpose-gui /path/to/suite2p/plane00     # edit suite2p results
  cellpose-gui /path/to/cellpose_output     # edit cellpose results
  cellpose-gui /path --output /path/edited  # save edits to specific dir
  cellpose-gui /path --format both          # save as both formats
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

    run_gui_workflow(
        path=args.path,
        output_dir=args.output,
        output_format=args.format,
        save_on_exit=not args.no_save,
        seg_file=args.seg_file,
    )


if __name__ == "__main__":
    main()
