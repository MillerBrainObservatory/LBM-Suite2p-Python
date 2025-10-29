from pathlib import Path
import lbm_suite2p_python as lsp

base_dir = r"D:\demo\local_nvme"
extension = ".bin"

input_files = list(Path(base_dir).rglob(f"*{extension}*"))

new_ops = {
    "anatomical_only": 3,
    "diameter": 4,
    "cellprob_threshold": -6,
    "flow_threshold": 0,
    "two_step_registration": True,
}

lsp.run_volume(
    input_files=input_files,
    ops=new_ops,
)
