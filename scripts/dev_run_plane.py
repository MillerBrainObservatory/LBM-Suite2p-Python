from pathlib import Path
import numpy as np
import mbo_utilities as mbo
from lbm_suite2p_python.run_lsp import run_plane

if __name__ == "__main__":

    fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\data_roi")
    files = [x for x in fpath.iterdir() if x.suffix in [".tif", ".tiff", ".zarr"]]
    zarray = mbo.imread(files[0],)
    mean = np.mean(zarray[:100, 0, :, :], axis=0).squeeze()

    # file = fpath  # Use a specific file for testing
    user_ops = {
        "do_regmetrics": True,
        "two_step_registration": True,
        "block_size": (64, 64),
        "reg_tif": True,
        "save_nwb": True,
        "roidetect": False,
        "refImg": mean,
        "force_refImg": True,
    }

    save_path = fpath.parent.joinpath("s2p-fix-reg-force-refimg")
    save_path.mkdir(exist_ok=True)

    _ = run_plane(
        input_path=files[0],
        save_path=save_path,
        ops=user_ops,
        keep_reg=True,
        keep_raw=False,
        force_reg=False,
        force_detect=False,
        save_json=True
    )

