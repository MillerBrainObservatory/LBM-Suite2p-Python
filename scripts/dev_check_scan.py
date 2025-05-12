from pathlib import Path

from numba.core.extending import overload_method

import lbm_suite2p_python as lsp
import mbo_utilities as mbo

from lbm_suite2p_python import run_volume

raw_path = Path("/home/flynn/lbm_data/raw")

scan = mbo.read_scan(raw_path)
tiff_out = raw_path.parent.joinpath("tiff")
tiff_out.mkdir(exist_ok=True)

mbo.save_as(scan, tiff_out.joinpath("false"), planes=[12], ext="tif", fix_phase=False, overwrite=True)
mbo.save_as(scan, tiff_out.joinpath("true"), planes=[12], ext="tif", fix_phase=True, overwrite=True)
mbo.save_as(scan, tiff_out.joinpath("50"), planes=[12], ext="tif", fix_phase=True,target_chunk_mb=50, overwrite=True)
mbo.save_as(scan, tiff_out.joinpath("500"), planes=[12], ext="tif", fix_phase=True,target_chunk_mb=500, overwrite=True)

res_out = raw_path.parent.joinpath("results")

## BINFILES

# bin_out = raw_path.parent.joinpath("bin")
# bin_file = bin_out.joinpath("plane7", "data_raw.bin")
# ops1 = lsp.run_plane(
#     input_path=bin_file,
#     keep_raw=True
# )
# files = mbo.get_files(tiff_out, "tif", 2)
# ops = run_volume(files,)
#
# for file in files:
#     ops2 = lsp.run_plane(
#         input_path=file,
#         keep_reg=True,
#         keep_raw=False,
#         force_reg=False,
#         force_detect=True,
#     )

# res2 = lsp.load_planar_results(ops2)
# x = 5