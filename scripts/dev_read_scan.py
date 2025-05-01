from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import suite2p
import mbo_utilities as mbo
import fastplotlib as fpl
import lbm_suite2p_python as lsp

scan = mbo.read_scan(r"D://demo/raw_data/*")
iw = fpl.ImageWidget(scan)

if __name__ == "__main__":
    fpl.loop.run()