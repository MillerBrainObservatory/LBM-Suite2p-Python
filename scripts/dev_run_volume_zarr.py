from mbo_utilities import get_files, imread, imwrite
from pathlib import Path
import warnings
from lbm_suite2p_python.run_lsp import run_volume

warnings.simplefilter(action='ignore')

# 1. Extract Raw Data
inpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw")
# # arr = imread(inpath, preprocess=True)
# # imwrite(arr, inpath.parent.joinpath("raw_data"), ext=".zarr")
# #
# # inpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\suite2p\z_registered")
# # aligned_files = get_files(inpath, "aligned", max_depth=3)
#
# from lbm_suite2p_python.merging import merge_rois_for_planes, remake_plane_figures, merge_zarr_rois
# # fpath = inpath.parent.joinpath("raw_data")
fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\data_roi")
files= [x for x in fpath.iterdir() if x.suffix==".zarr"]
#
# save_path = inpath.joinpath("anatomical_3_roi")
# rois_dir = save_path.joinpath("merged_rois")
# out_dirs = [x for x in rois_dir.iterdir() if x.is_dir()]
#
# zarr_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\data_roi")
# merge_zarr_rois(zarr_path)
#
# merge_rois_for_planes(save_path, rois_dir)
# for plane_dir in out_dirs:
#     remake_plane_figures(plane_dir)

run_volume(
    files,
    save_path=inpath.joinpath(f"anatomical_1_roi"),
    ops={
        "anatomical_only": 1,
        "cellprob_threshold": -6,
        "diameter": 6,
    },
)
