from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import suite2p
import mbo_utilities as mbo
from copy import deepcopy

import matplotlib as mpl

mpl.rcParams.update({
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.subplot.wspace': .01,
    'figure.subplot.hspace': .01,
    'figure.figsize': (18, 13),
    'ytick.major.left': True,
})
jet = mpl.cm.get_cmap('jet')
jet.set_bad(color='k')

def slurm():
    ops = suite2p.default_ops()
    ops['sparse_mode'] = 1
    ops['threshold_scaling'] = 3.0
    db = []
    db.append({'data_path': ['D:/DATA/output']})
    db.append({'data_path': ['D:/DATA/output2']})
    for dbi in db:
        opsEnd = suite2p.run_s2p(ops=ops, db=dbi)

def main():

    # path setup
    data_path = Path(r"D:\W2_DATA\kbarber\2025-02-10\mk303\assembled")
    data_path.mkdir(exist_ok=True)
    save_path = Path(r"D:\W2_DATA\kbarber\2025-02-10\mk303\results")
    save_path.mkdir(exist_ok=True, parents=True)

    # metadata setup
    files = mbo.get_files(data_path)
    metadata = mbo.get_metadata(files[0])
    ops = suite2p.default_ops()
    ops = mbo.params_from_metadata(metadata, ops)

    # change params
    # make sure tiff list is a list
    ops["data_path"] = [str(data_path)]
    ops["save_path0"] = "test1"
    ops["tiff_list"] = [
        str(Path(files[1]).name),
    ]
    ops['iplane'] = 'plane_2'

    output_ops = suite2p.run_s2p(ops=ops)

    print(output_ops)

if __name__ == "__main__":
    main()