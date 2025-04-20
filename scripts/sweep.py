import numpy as np
from pathlib import Path
from collections import defaultdict
import re

import lbm_suite2p_python as lsp

SEG_PATTERN = re.compile(
    r"plane_\d+(_ts(?P<ts>[\d.]+))?_shp(?P<shp>\d+)_mo(?P<mo>[\d.]+)_hp(?P<hp>\d+)"
)
DEFAULT_NAME = "plane_07"
DEFAULT_VALUES = {
    "threshold_scaling": 1.0,
    "spatial_hp_detect": 25,
    "max_overlap": 1.0,
    "high_pass": 100
}

def parse_parameters(path):
    match = SEG_PATTERN.search(str(path))
    if match:
        return {
            "threshold_scaling": float(match.group("ts") or 0.0),
            "spatial_hp_detect": int(match.group("shp")),
            "max_overlap": float(match.group("mo")),
            "high_pass": int(match.group("hp"))
        }
    elif str(path.parents[1]) == str(path.parents[1].parent / DEFAULT_NAME):
        return DEFAULT_VALUES.copy()
    else:
        return None

def group_by_param(seg_paths, all_params, param):
    groups = defaultdict(list)
    for p, conf in zip(seg_paths, all_params):
        if conf is None:
            continue
        key = (conf[param],)
        groups[key].append((p, conf))
    return groups

def make_comparative_projections(group, param_name, outdir):
    for path, conf in group:
        ops_path = path.parent.parent / 'plane0' / 'ops.npy'
        if not ops_path.exists():
            continue
        ops = np.load(ops_path, allow_pickle=True).item()
        label = f"{param_name.replace('_', ' ').capitalize()} (default {DEFAULT_VALUES[param_name]}): {conf[param_name]}"
        savefile = outdir / f"{param_name}_{conf[param_name]:.2f}.png"
        lsp.plot_projection(ops, savepath=savefile, fig_label=label, display_masks=True, accepted_only=True)

def main():
    base_dir = Path(r"C:\Users\RBO\repos\nb_sandbox\2025_04\lsp\defaults")
    seg_paths = list(base_dir.rglob("segmentation.png"))
    all_params = [parse_parameters(p) for p in seg_paths]

    outpath = base_dir.parent / "comparative_projections"
    outpath.mkdir(exist_ok=True)

    for param in ["threshold_scaling", "spatial_hp_detect", "max_overlap", "high_pass"]:
        grouped = group_by_param(seg_paths, all_params, param)
        for group_key, group in grouped.items():
            make_comparative_projections(group, param, outpath)

if __name__ == "__main__":
    main()
