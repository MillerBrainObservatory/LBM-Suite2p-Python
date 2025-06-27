from pathlib import Path
import lbm_suite2p_python as lsp
import mbo_utilities as mbo

if __name__ == "__main__":

    path = Path(r"D:\demo\test\plane09_roi2.tif")
    save_path = path.parent.joinpath("suite2p")
    save_path.mkdir(exist_ok=True)
    user_ops = {
        "anatomical_only": 3,
        "pretrained_model": "cpsam",
        "cellprob_threshold": -6,
        "flow_threshold": 0.8,
        "diameter": 3
    }

    ops = lsp.run_plane(
        input_path=path,
        save_path=save_path.joinpath("anatomical_cpsam"),
        ops=user_ops,
        keep_reg=True,
        keep_raw=True,
        force_reg=True,
        force_detect=True,
    )

    res = lsp.load_planar_results(ops)
    print(res)