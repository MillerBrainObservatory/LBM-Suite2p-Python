from pathlib import Path
import lbm_suite2p_python as lsp
import mbo_utilities as mbo

raw_path = Path(r"/home/flynn/lbm_data/raw")
# scan = mbo.read_scan(raw_path)
# scan.roi = None
# mbo.save_as(
#     scan,
#     raw_path / "testing",
# )
path = Path(r"/home/flynn/lbm_data/raw/testing/plane7.tif")
user_ops = {"classifier_path": "/home/flynn/lbm_data/mbo_v3.npy"}

ops = lsp.run_plane(
    input_path=path,
    save_path=path.parent,  # unused for single-z bin files
    ops=user_ops,
    keep_reg=True,
    keep_raw=True,
    force_reg=False,
    force_detect=True,
    debug=True
)

res = lsp.load_planar_results(ops)
print(res)