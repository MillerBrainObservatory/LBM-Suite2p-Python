from pathlib import Path
import mbo_utilities as mbo

import lbm_suite2p_python as lsp

base_dir = r"D://demo//single_channel//mrois_fft"
zarr_files = list(Path(base_dir).glob("*.zarr"))

new_ops = {
    "anatomical_only": 3,
    "two_step_registration": True,
    "roidetect": False
}

save_path = Path(r"D://demo//mrois_fft")  # will be created if it doesn't exist

lsp.run_volume(
    input_files=zarr_files,
    save_path=save_path,
    ops=new_ops,
)
