from pathlib import Path
import mbo_utilities as mbo
from mbo_utilities import get_files

import lbm_suite2p_python as lsp
from lbm_suite2p_python.merging import merge_mrois

input_files = mbo.get_files("D://demo//mrois", 'tif')

new_ops = {
    "anatomical_only": 3,
    "two_step_registration": True,
}

save_path = Path(r"D://demo//volumetric_suite2p")  # will be created if it doesn't exist

lsp.run_volume(
    input_files=input_files,
    save_path=save_path,
    ops=new_ops,
)

if "roi" in Path(input_files[0]).stem.lower():
    print("Detected mROI data, merging ROIs for each z-plane...")

    merged_savepath = save_path.joinpath("merged_mrois")
    merge_mrois(save_path, merged_savepath)
    all_ops = sorted(get_files(merged_savepath, "ops.npy", 2))
