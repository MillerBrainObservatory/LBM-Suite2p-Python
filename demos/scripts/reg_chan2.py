import time

from pathlib import Path
from mbo_utilities import get_files, imread, imwrite

if __name__ == "__main__":
    import lbm_suite2p_python as lsp

    base_path = Path(r"D:\demo\multichannel")

    structural = get_files(base_path / "structural")
    functional = get_files(base_path / "functional")

    start = time.time()
    s_data = imread(structural[0])
    f_data = imread(functional[0])
    end = time.time()
    print(f"{end - start} seconds")

    outpath = structural[0].parent.joinpath("out")
    imwrite(f_data, outpath, ext=".bin", structural=False)
    imwrite(s_data, outpath, ext=".bin", structural=True) # names the file data_chan2.bin

    ops = {
        "nchannels": 1,
        "align_by_chan": 2,
    }

    ops_files = get_files(outpath, "ops.npy", 3)
    ops1 = lsp.load_ops(ops_files[0])
    lsp.run_plane(outpath)
