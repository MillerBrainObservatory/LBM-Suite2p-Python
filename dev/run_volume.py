from datetime import datetime
from pathlib import Path
import lbm_suite2p_python as lsp
import mbo_utilities as mbo
# from mbo_utilities.graphics.run_gui import _check_installation

# _check_installation()

current_date_time = datetime.now()

if __name__ == "__main__":
    ## extract data
    raw_scanimage_tiffs = Path(r"D:\W2_DATA\wsynder")
    raw_data = mbo.imread(raw_scanimage_tiffs)
    save_path = Path(r"D:\W2_DATA\wsynder\v2") #+ f"{current_date_time.strftime('%Y%m%d')}")

    mbo.imwrite(
        raw_data,
        outpath=save_path,
        ext=".zarr",
        register_z=True,
        overwrite=False,
        metadata={"notes": "Reproducing will snyder results"},
        roi=None,
    )

    query_str = ".zarr"
    input_files = list(save_path.glob(f"*{query_str}*"))

    new_ops = {
        "do_registration": 1,
        "anatomical_only": 4,
        "cellprob_threshold": -6,
        "flow_threshold": 0,
        "diameter": 4,
        "denoise": 1,
        "spatial_hp_cp": 0.5,
    }

    lsp.run_volume(
        input_files=[input_files[0]],
        save_path=None, # saves next to input files
        ops=new_ops,
        keep_raw=True,
        keep_reg=True,
        force_detect=True
    )

    x = 2
