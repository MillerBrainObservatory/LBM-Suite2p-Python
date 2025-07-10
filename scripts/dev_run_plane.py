from pathlib import Path
import lbm_suite2p_python as lsp
import mbo_utilities as mbo

if __name__ == "__main__":

    fpath = Path().home().joinpath("lbm_data", "fused")
    files = [
        x for x in fpath.glob("*.tif*")
    ]
    file = files[0]

    user_ops = {
        # "two_step_registration": True,
        "anatomical_only": 0,
        "reg_tif": True,
        # "roi_detect": False,
        # "do_registration": False,
    }

    save_path = fpath.joinpath("temp")
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

