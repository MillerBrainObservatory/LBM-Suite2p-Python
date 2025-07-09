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
        "two_step_registration": True,
        "anatomical_only": 0,
        "pretrained_model": "cyto3",
        "cellprob_threshold": -6,
        "flow_threshold": 1.0,
        "diameter": 10,
    }

    save_path = fpath.joinpath("func")
    save_path.mkdir(exist_ok=True)

    _ = lsp.run_plane(
        input_path=file,
        save_path=save_path,
        ops=user_ops,
        keep_reg=True,
        keep_raw=True,
        force_reg=True,
        force_detect=True,
    )

