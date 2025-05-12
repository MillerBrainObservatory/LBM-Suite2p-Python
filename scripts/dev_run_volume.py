import lbm_suite2p_python as lsp
import mbo_utilities as mbo

# as tiff

save_path = "/home/flynn/lbm_data/demo/tiff"
files = mbo.get_files("/home/flynn/lbm_data/demo/assembled", 'tif', 3)

lsp.run_volume(input_files=files, save_path=save_path)
