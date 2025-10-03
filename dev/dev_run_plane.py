from pathlib import Path
import numpy as np
import mbo_utilities as mbo
from lbm_suite2p_python.run_lsp import run_plane

if __name__ == "__main__":

    fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\data_roi")

    inpath = Path(r"D:\W2_DATA\kbarber\2025-03-01\green")
    extracted_outpath = inpath.parent.joinpath("green.processed")
    files = [
        x for x in extracted_outpath.iterdir() if x.suffix == ".zarr"
    ]

    # file = fpath  # Use a specific file for testing
    for anat in [0, 1, 2, 3]:
        print(f"Running anatomical_only = {anat}")

        user_ops = {
            "anatomical_only": anat,
            "cellprob_threshold": -6,
            "flow_threshold": -6,
            "do_regmetrics": True,
            "two_step_registration": True,
        }

        outpath = inpath.parent.joinpath(f"anatomical_{anat}")

        _ = run_plane(
            input_path=files[8],
            save_path=outpath,
            ops=user_ops,
            keep_reg=True,
            keep_raw=False,
            force_reg=False,
            force_detect=False,
            save_json=False,
        )

