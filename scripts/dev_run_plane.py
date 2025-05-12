from pathlib import Path
import lbm_suite2p_python as lsp
import mbo_utilities as mbo

from lbm_suite2p_python import run_volume

raw_path = Path("/home/flynn/lbm_data/raw")
bin_out = raw_path.parent.joinpath("bin")

scan = mbo.read_scan(raw_path)
# mbo.save_as(scan, bin_out, planes=[7], ext="bin")
#
# bin_file = bin_out.joinpath("plane7", "data_raw.bin")
# ops1 = lsp.run_plane(
#     input_path=bin_file,
#     keep_raw=True
# )

tiff_out = raw_path.parent.joinpath("tiff")
res_out = raw_path.parent.joinpath("results")
mbo.save_as(scan, tiff_out, planes=[7, 13], ext="tif")
files = mbo.get_files(tiff_out, "tif", 2)
ops = run_volume(files,)

for file in files:
    ops2 = lsp.run_plane(
        input_path=file,
        keep_reg=True,
        keep_raw=False,
        force_reg=False,
        force_detect=True,
    )

# res2 = lsp.load_planar_results(ops2)
# x = 5