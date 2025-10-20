import time

from suite2p import pipeline

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
    # imwrite(f_data, outpath, ext=".bin", structural=False)
    #
    # ops = {
    #     "nchannels": 1,
    #     "functional_channel": 2,  # not used but set to avoid warnings}
    #     "align_by_chan": 2,
    # }
    ops_files = get_files(outpath, "ops.npy", 3)
    ops1 = lsp.load_ops(ops_files[0])
    lsp.run_plane(outpath)
