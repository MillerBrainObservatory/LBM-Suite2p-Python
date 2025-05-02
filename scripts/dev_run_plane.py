import lbm_suite2p_python as lsp
import mbo_utilities as mbo

save_path = "/home/flynn/lbm_data/demo/amol"
scan = mbo.read_scan("/home/flynn/lbm_data/demo/raw")
mbo.save_as(scan, "/home/flynn/lbm_data/demo/amol", ext=".bin", planes=[7, 8])
input_path = "/home/flynn/lbm_data/demo/amol/plane7/data_raw.bin"

ops1 = lsp.run_plane(
    input_path=input_path,
    save_path=save_path,
    force_reg=True,
    keep_raw=True,
)

res1 = lsp.load_planar_results(ops1)
print(sum(res1["iscell"]))

ops2 = lsp.run_plane(
    input_path=input_path,
    save_path=save_path,
    keep_reg=True,
    keep_raw=False,
    force_reg=False,
    force_detect=True,
)

res2 = lsp.load_planar_results(ops1)

ops1path = "/home/flynn/lbm_data/demo/ops1.npy"
ops2path = "/home/flynn/lbm_data/demo/ops2.npy"
import numpy as np
np.save(ops1path, ops1)
np.save(ops2path, ops2)