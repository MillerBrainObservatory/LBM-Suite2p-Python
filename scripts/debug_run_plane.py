from pathlib import Path
import numpy as np
import suite2p
import mbo_utilities as mbo
import lbm_suite2p_python as lsp

assembled_files = mbo.get_files("D://demo_functional//assembled_ws", 'tif')
metadata = mbo.get_metadata(assembled_files[0])
ops = mbo.params_from_metadata(metadata, suite2p.default_ops())
ops["classifier_path"] = "D://demo//mbo_v1.npy"

# ops_files = mbo.get_files("D://demo_functional//", "ops.npy", 4)
# ops = lsp.load_ops(ops_files[0])

lsp.run_plane(ops, assembled_files[0], save_path="D://demo_functional//results", save_folder="v1")