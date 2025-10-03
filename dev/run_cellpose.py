import numpy as np
from pathlib import Path
from cellpose import models
from cellpose.io import imread

# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=False, model_type='cyto')

fpath = Path(r"D:\W2_DATA\kbarber\2025-01-30\mk303\green\assembled")
files = [x for x in fpath.glob("*.tif*")]

output_path = Path(r"D:\W2_DATA\kbarber\2025-01-30\mk303\green\cellpose")
imgs = imread(files)
imgs.shape

masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0], flow_threshold=0.4, do_3D=False)
mask_out = output_path / "mask.npy"
styles_out = output_path / "styles.npy"
flows_out = output_path / "flows.npy"
diams_out = output_path / "diams.npy"
np.save(mask_out, masks)
np.save(flows_out, flows)
np.save(styles_out, styles)
np.save(diams_out, diams)
