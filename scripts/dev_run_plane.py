from pathlib import Path
import lbm_suite2p_python as lsp
import mbo_utilities as mbo

if __name__ == "__main__":

    files = [
        x for x in Path().home().joinpath("lbm_data", "fused").glob("*.tif*")
    ]
    user_ops = {
        "two_step_registration": True,
        "anatomical_only": 3,
        "pretrained_model": "cyto3",
        "cellprob_threshold": -6,
        "flow_threshold": 1.0,
        "diameter": 10,
    }

    for file in files:
        save_path = Path(r"D:\demo\frame_phase\two_step")
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

        save_path = Path(r"D:\demo\frame_phase\one_step")
        save_path.mkdir(exist_ok=True)
        user_ops["two_step_registration"] = False
        _ = lsp.run_plane(
            input_path=file,
            save_path=save_path,
            ops=user_ops,
            keep_reg=True,
            keep_raw=True,
            force_reg=True,
            force_detect=False,
        )



    # res = lsp.load_planar_results(ops)
