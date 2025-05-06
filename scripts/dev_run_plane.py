from pathlib import Path
import lbm_suite2p_python as lsp
import mbo_utilities as mbo

raw_path = Path("/home/flynn/lbm_data/demo/raw")

bin_out = raw_path.parent.joinpath("bin")
tiff_out = raw_path.parent.joinpath("tiff")

scan = mbo.read_scan(raw_path)
mbo.save_as(scan, bin_out, planes=[7], ext="bin")

bin_file = bin_out.joinpath("plane7", "data_raw.bin")
ops1 = lsp.run_plane(
    input_path=bin_file,
    keep_raw=True
)

res1 = lsp.load_planar_results(ops1)
print(sum(res1["iscell"]))

mbo.save_as(scan, tiff_out, planes=[7], ext="tif")
bin_file = tiff_out.joinpath("plane7", "data_raw.bin")
ops2 = lsp.run_plane(
    input_path=bin_file,
    save_path=tiff_out.joinpath("res"),
    keep_reg=True,
    keep_raw=False,
    force_reg=False,
    force_detect=True,
)

res2 = lsp.load_planar_results(ops1)
x = 5