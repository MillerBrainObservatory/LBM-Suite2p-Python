from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import suite2p
import mbo_utilities as mbo
import fastplotlib as fpl
import lbm_suite2p_python as lsp

animal_path = Path(r"D:\W2_DATA\wsnyder\2025_03_06")
raw_path = animal_path.joinpath("raw/dot_lv6")
assembled_path = animal_path.joinpath("assembled")
save_path = animal_path.joinpath("grid_search")  # where the z-plane tiff files live

dpath = Path(r"~/repos/nb_sandbox/2025_04/data/")
files = mbo.get_files(dpath, 'tif', 2)
print(files)