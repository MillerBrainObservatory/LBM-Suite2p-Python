from pathlib import Path
from lbm_suite2p_python.run_lsp import run_volume
# import numpy as np
# from mbo_utilities import imread, imwrite

if __name__ == "__main__":

    inpath = Path(r"D:\W2_DATA\kbarber\2025-03-01\green")
    extracted_outpath = inpath.parent.joinpath("green.processed")
    files = [
        x for x in extracted_outpath.iterdir() if x.suffix == ".zarr"
    ]

    outpath = inpath.parent.joinpath("green.s2p_results")
    run_volume(
        files,
        save_path=outpath,
        ops={
            "diameter": [4, 4],
            "anatomical_only": 3,
            "cellprob_threshold": -6,
            "flow_threshold": 0.1,
            "do_regmetrics": True,
            "two_step_registration": True,
            # "roidetect": True,
        },
        # keep_raw=False,
        # force_reg=False,
        # force_detect=False,
    )