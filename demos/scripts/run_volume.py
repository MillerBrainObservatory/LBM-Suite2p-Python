from pathlib import Path
import lbm_suite2p_python as lsp

base_dir = r"D:\W2_DATA\wsynder"
query_str = ".zarr"

input_files = list(Path(base_dir).rglob(f"*{query_str}*"))

new_ops = {
    "do_registration": 0,
    "anatomical_only": 4,
    "cellprob_threshold": -2,
    "flow_threshold": -6,
    "diameter": 5,
    "pretrained_model": "cpsam",
    "lam_percentile": 0,
}

lsp.run_volume(
    input_files=[input_files[0]],
    save_path=None, # saves next to input files
    ops=new_ops,
    keep_raw=True,
    keep_reg=True,
    force_detect=True
)

x = 2
