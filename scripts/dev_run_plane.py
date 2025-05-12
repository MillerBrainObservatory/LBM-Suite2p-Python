from pathlib import Path
import lbm_suite2p_python as lsp

path = Path(r"/home/flynn/lbm_data")
res_out = path.joinpath("results")

ops = lsp.run_plane(
    input_path=path.joinpath("assembled", "plane_08.tif"),
    save_path=res_out,
    keep_reg=True,
    keep_raw=False,
    force_reg=False,
    force_detect=True,
)

# res2 = lsp.load_planar_results(ops2)
# x = 5