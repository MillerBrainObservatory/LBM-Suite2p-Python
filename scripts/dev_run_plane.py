from pathlib import Path
import lbm_suite2p_python as lsp

path = Path(r"D://demo//assembled5")
res_out = path.parent.joinpath("suite2p_results")

ops = lsp.run_plane(
    input_path=path.joinpath("plane_08.tif"),
    save_path=res_out,
    keep_reg=True,
    keep_raw=True,
    force_reg=False,
    force_detect=True,
)

res = lsp.load_planar_results(ops)
print(res)