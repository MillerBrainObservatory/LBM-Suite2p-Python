from pathlib import Path
import lbm_suite2p_python as lsp

path = Path(r"/home/flynn/lbm_data/assembled/roi2/plane7.tif")
save_path = Path(r"/home/flynn/lbm_data/assembled/roi2/plane7_default")
user_ops = {
    "anatomical_only": 3,
    "nbinned" : 220,
    "cellprob_threshold": 0.0,
    "flow_threshold": 0.9,
    "diameter": 6,
}

ops = lsp.run_plane(
    input_path=path,
    save_path=path.parent,
    ops=user_ops,
    keep_reg=True,
    keep_raw=True,
    force_reg=False,
    force_detect=True,
    debug=True
)
x = 2
