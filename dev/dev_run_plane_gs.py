from pathlib import Path
import itertools
import lbm_suite2p_python as lsp
import pandas as pd
import numpy as np

def summarize_regdx_metrics(root_dir: Path, subdir: str = "") -> pd.DataFrame:
    rows = []

    for ops_path in root_dir.rglob(f"*{subdir}*/ops.npy"):
        try:
            ops = lsp.load_ops(ops_path)
            regdx = np.array(ops.get("regDX"))
            if regdx.ndim != 2 or regdx.shape[1] != 3:
                continue

            dx = regdx[:, 0]
            dy = regdx[:, 1]
            norm = regdx[:, 2]

            rows.append({
                "config": ops_path.parent.parent.name,
                "mean_rigid": np.mean(np.abs(dx)),
                "mean_avg_nr": np.mean(np.abs(dy)),
                "mean_max_nr": np.mean(np.abs(norm)),
            })
        except Exception as e:
            print(f"Error processing {ops_path}: {e}")
            continue
    return pd.DataFrame(rows)

fpath = Path().home() / "lbm_data" / "fused"
files = list(fpath.glob("*.tif*"))
save_base = fpath / "gridsearch"

df = summarize_regdx_metrics(save_base, subdir="plane10")

two_step_options = [False, True]
block_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
spatial_hp_values = [0, 21, 42]

param_grid = list(itertools.product(two_step_options, block_sizes, spatial_hp_values))

for i, (two_step, block_size, spatial_hp) in enumerate(param_grid):
    user_ops = {
        "two_step_registration": two_step,
        "reg_tif": True,
        "block_size": block_size,
        "spatial_hp_reg": spatial_hp,
        "do_detect": False,
    }

    for file in files:
        label = f"ts{int(two_step)}_bs{block_size[0]}x{block_size[1]}_hp{spatial_hp}"
        save_path = save_base / label / file.stem
        save_path.mkdir(parents=True, exist_ok=True)

        lsp.run_plane(
            input_path=file,
            save_path=save_path,
            ops=user_ops,
            keep_reg=True,
            keep_raw=True,
            force_reg=True,
            force_detect=True,
        )
        

