from pathlib import Path
import fastplotlib as fpl
import mbo_utilities as mbo
from mbo_utilities.graphics.run_gui import run_gui

if __name__ == "__main__":
    filepath = Path(r"E:\tests\lbm\mbo_utilities\test_input.tif")
    run_gui(filepath)
