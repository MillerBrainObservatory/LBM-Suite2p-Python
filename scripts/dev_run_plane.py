import lbm_suite2p_python as lsp

save_path = "/home/flynn/lbm_data/demo/output"
input_path = "/home/flynn/lbm_data/demo/assembled/plane_08.tif"
lsp.run_plane_any(
    input_path=input_path,
    save_path=save_path,
    keep_raw=True,
    keep_reg=True,
    force_reg=False,
    force_detect=True,
)