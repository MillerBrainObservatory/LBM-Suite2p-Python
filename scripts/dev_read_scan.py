from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import suite2p
import mbo_utilities as mbo
import fastplotlib as fpl
import lbm_suite2p_python as lsp

scan = mbo.read_scan(r"/home/flynn/lbm_data/demo/raw")
iw = mbo.run_gui(scan)
iw.show()
mbo.save_as(scan, r"D:/masknmf/assembled", ext=".tif", planes=[8], target_chunk_mb=100)
out_file = Path(r"D:/masknmf/assembled/plane_08.tif")
# print(out_file.is_file())
# lsp.run_plane(out_file, save_path=r"D:/masknmf/output", keep_raw=True, keep_reg=True)
# iw = fpl.ImageWidget(scan)

if __name__ == "__main__":
    fpl.loop.run()