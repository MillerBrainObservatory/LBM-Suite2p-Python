import time

from pathlib import Path
from mbo_utilities import get_files, imread, imwrite

if __name__ == "__main__":
    import lbm_suite2p_python as lsp

    base_path = Path(r"D:\demo\multichannel")

    structural = get_files(base_path / "structural")
    functional = get_files(base_path / "functional")

    s_data = imread(structural[0])
    f_data = imread(functional[0])

    lsp.run_plane(
        input_path=functional[0],
        chan2_path=structural,
        save_path=base_path / "test"
    )
