import time

from pathlib import Path
from mbo_utilities import get_files, imread, imwrite

if __name__ == "__main__":
    import lbm_suite2p_python as lsp

    base_path = Path(r"D:\demo\multichannel")

    structural = get_files(base_path / "structural")
    functional = get_files(base_path / "functional")

    s_data = imread(structural[0])
    f_data = imread(functional[0])
    min_frames = min(f_data.shape[0], s_data.shape[0])

    outpath = base_path.joinpath("test")
    imwrite(
        s_data,
        outpath,
        ext=".bin",
        structural=True,
        planes=[8, 11],
        num_frames=min_frames,
    ) # names the file data_chan2.bin
    imwrite(
        f_data,
        outpath,
        ext=".bin",
        structural=False,
        planes=[8, 11],
        num_frames=min_frames,
    )

    input_path = outpath.joinpath("plane10_stitched")
    ops = {
        "anatomical_only": 3,
        "diameter": 4,
        "cellprob_threshold": -6,
        "flow_threshold": -6,
    }
    for path in outpath.glob("*plane*"):
        lsp.run_plane(
            path,
            save_path=path,
            ops=ops,
        )
