from pathlib import Path
import mbo_utilities as mbo
import lbm_suite2p_python as lsp

ops = {"dff_percentile": 50}
input_files = mbo.get_files(r"D:\W2_DATA\wsynder", 'tif')
lsp.run_volume(input_files, r"D:\W2_DATA\wsynder\results", ops=ops)