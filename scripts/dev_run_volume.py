from pathlib import Path
import lbm_suite2p_python as lsp
import mbo_utilities as mbo

save_path = r"D:\W2_DATA\santi\stitched"
files = mbo.get_files("D:\W2_DATA\santi\stitched", 'tif', 3)

lsp.run_volume(input_files=files, save_path=save_path)