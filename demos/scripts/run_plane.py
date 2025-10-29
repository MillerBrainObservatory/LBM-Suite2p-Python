from pathlib import Path
import lbm_suite2p_python as lsp

new_ops = {
    "anatomical_only": 3,
    "diameter": 4,
    "cellprob_threshold": -6,
    "flow_threshold": -6,
    "two_step_registration": True,
}

input_file = r"D:\demo\local_staging\plane06_stitched\data_raw.bin"
save_path = Path(r"D://demo//local_staging//suite2p")  # will be created if it doesn't exist

output_ops = lsp.run_plane(
    input_path=input_file,  # run_plane takes a single input fileppath
    # save_path=save_path,
    ops=new_ops,         # default is None
    keep_raw=True,       # default is False
    keep_reg=True,       # default is True
    force_reg=False,     # default is False
    force_detect=False,  # default is False
)
