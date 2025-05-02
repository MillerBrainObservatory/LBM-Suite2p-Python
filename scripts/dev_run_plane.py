import lbm_suite2p_python as lsp

save_path = "/home/flynn/lbm_data/demo/output"
input_path = "/home/flynn/lbm_data/demo/assembled/plane_07.tif"

ops1 = lsp.run_plane_any(
    input_path=input_path,
    save_path=save_path,
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