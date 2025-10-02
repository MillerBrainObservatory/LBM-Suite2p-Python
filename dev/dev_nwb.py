import mbo_utilities as mbo
from pathlib import Path
from suite2p.io import nwb as s2pnwb
import numpy as np

fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355")
plane_folders = fpath.joinpath("plane7")
ops = np.load(plane_folders.joinpath("ops.npy"), allow_pickle=True).item()
ops["save_path"] = str(plane_folders)
np.save(plane_folders.joinpath("ops.npy"), ops)
# ops["data_path"] = fpath.joinpath("stitched", "plane07_stitched.tif")

# s2pnwb.save_nwb(fpath)

# stat, ops, F, Fneu, spks, iscell, probcell, redcell, probredcell
nwb_obj = s2pnwb.read_nwb(fpath.joinpath("ophys.nwb"))
x = 4