from pathlib import Path
import lbm_suite2p_python as lsp

base_dir = r"D:\demo\ome_v2\sharded"
query_str = ".zarr"

input_files = list(Path(base_dir).rglob(f"*{query_str}*"))

new_ops = {
    "anatomical_only": 3,
    "diameter": 4,
    "cellprob_threshold": -6,
    "flow_threshold": 0,
    "spatial_hp_cp": 0.5,
}

lsp.run_volume(
    input_files=input_files,
    save_path=None, # saves next to input files
    ops=new_ops,
    keep_raw=True,
    keep_reg=True,
)
