import numpy as np
from pathlib import Path
import suite2p

data_path = Path("E://datasets/high_resolution/zplanes/")

files = [x for x in data_path.glob('*.tif*')]
filename = files[0]

print(filename)

save_path0 = data_path / filename / 'results'
save_path0.mkdir(exist_ok=True)

db = {
    'data_path': [filename.parent],
    'save_path0': str(save_path0),
    'tiff_list': [filename],
}

print(db)

np.save(data_path / 'db.npy', db)

