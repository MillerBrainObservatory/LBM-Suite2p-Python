import numpy as np
import lbm_suite2p_python as lsp

if __name__ == "__main__":

    ops = {
        "diameter": 4,
        "anatomical_only": 3,
        # "spatial_hp_cp": 3,
        "denoise": 1,
        "two_step_registration": 1,
    }

    input_data = r"D:\GCamp_dTomato_tests\corrected"
    save_path = r"D:\GCamp_dTomato_tests\suite2p"

    results = lsp.pipeline(
        input_data=input_data,      # default: path to .zarr, .tiff, or .bin file
        save_path=save_path,        # default: save next to input file
        ops=ops,                    # default: use MBO-optimized parameters
        planes=list(np.arange(1, 14)),                # process single plane (1-indexed)
        num_timepoints=500,
        roi=None,                   # default: stitch multi-ROI data
        keep_reg=True,              # default: keep data.bin (registered binary)
        keep_raw=False,             # default: delete data_raw.bin after processing
        force_reg=False,            # default: skip if already registered
        force_detect=False,         # default: skip if stat.npy exists
        dff_window_size=None,       # default: auto-calculate from tau and framerate
        dff_percentile=20,          # default: 20th percentile for baseline
        dff_smooth_window=None,     # default: auto-calculate from tau and framerate
    )
