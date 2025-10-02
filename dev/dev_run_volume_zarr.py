from mbo_utilities import get_files, imread, imwrite
from pathlib import Path
import warnings
from lbm_suite2p_python.run_lsp import run_volume

warnings.simplefilter(action='ignore')

inpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw")
files= [
    x for x in Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\data_roi").iterdir()
    if x.suffix==".zarr"
]

import time
start = time.time()
run_volume(
    files,
    save_path=inpath.joinpath(f"anatomical_3_v5"),
    ops={
        "roidetect": True,
        "two_step_registration": True,
        "anatomical_only": 3,
        "do_regmetrics": True,
        "cellprob_threshold": -6,
        "flow_threshold": -6,
        "diameter": 6,
    },
    force_detect=True,
    force_reg=True,
)

end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")