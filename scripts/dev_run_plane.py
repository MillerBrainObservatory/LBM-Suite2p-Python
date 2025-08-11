from pathlib import Path
import lbm_suite2p_python as lsp
import mbo_utilities as mbo

if __name__ == "__main__":

    fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\stitched")
    files = [
        x for x in fpath.glob("*.tif*")
    ]
    file = files[1]
    # file = fpath  # Use a specific file for testing
    user_ops = {
        "anatomical_only": 0,
        "tau": 0.8,
        "reg_tif": True,
        "save_nwb": True,
        # "roi_detect": False,
        # "do_registration": False,
    }

    save_path = fpath.parent.joinpath("nwb_results")
    save_path.mkdir(exist_ok=True)

    _ = lsp.run_plane(
        input_path=file,
        save_path=save_path,
        ops=user_ops,
        keep_reg=True,
        keep_raw=True,
        force_reg=False,
        force_detect=False,
    )

