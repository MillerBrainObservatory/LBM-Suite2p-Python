from pathlib import Path
import lbm_suite2p_python as lsp

path = Path(r"/home/flynn/lbm_data/bin/roi2/plane11")
res_out = path.parent.joinpath("results_testing")

ops = lsp.run_plane(
    input_path=path.joinpath("raw_data.bin"),
    save_path=res_out,
    keep_reg=True,
    keep_raw=True,
    force_reg=False,
    force_detect=True,
    debug=True
)

res = lsp.load_planar_results(ops)
print(res)