from mbo_utilities import get_files, imread, imwrite
from pathlib import Path
import warnings
import lbm_suite2p_python as lsp

# # 1. Extract Raw Data
# inpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\green")
# arr = imread(inpath)
# imwrite(arr, inpath.parent.joinpath("raw_data"))
#
# warnings.simplefilter(action='ignore')

inpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\suite2p\z_registered")
aligned_files = get_files(inpath, "aligned", max_depth=3)

for anatomical in [1, 2, 3]:
    lsp.run_volume(
        aligned_files,
        save_path=inpath.joinpath(f"anatomical_{anatomical}"),
        ops={
            "anatomical_only": anatomical,
            "diameter": 6,
        },
        keep_raw=False,
        force_reg=False,
    )