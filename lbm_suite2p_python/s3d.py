import os
import traceback
import numpy as np
from pathlib import Path
import dask.array as da
import fastplotlib as fpl
import suite2p

from suite2p.io import BinaryFile, BinaryFileCombined
import mbo_utilities as mbo

from suite3d.job import Job
from suite3d import ui
from suite3d import io
from suite3d.job import Job
from suite3d import io


def get_params():
    """Chat GPT, just a copy of defaults for development"""

    # basic imaging
    params = {
        "tau": 1.3,                      # GCamp6s parameter (example)
        "voxel_size_um": (17, 2, 2),     # size of a voxel in microns (z, y, x)
        "planes": np.arange(14),         # planes to analyze (0-based indexing)
        "n_ch_tif": 14,                  # number of channels/planes in each TIFF
        "cavity_size": 1,
        "planes": np.arange(14),
        "lbm": True,
    }

    # Filtering Parameters (Cell detection & Neuropil subtraction)
    params.update({
        "cell_filt_type": "gaussian",  # cell detection filter type
        "npil_filt_type": "gaussian",  # neuropil filter type
        "cell_filt_xy_um": 5.0,        # cell detection filter size in xy (microns)
        "npil_filt_xy_um": 3.0,        # neuropil filter size in xy (microns)
        "cell_filt_z_um": 18,          # cell detection filter size in z (microns)
        "npil_filt_z_um": 2.5,            # neuropil filter size in z (microns)
    })

    # Normalization & Thresholding
    params.update({
        "sdnorm_exp": 0.8,          # normalization exponent for correlation map
        "intensity_thresh": 0.7,      # threshold for the normalized, filtered movie
        "extend_thresh": 0.15,
        "detection_timebin": 25,
    })

    # Compute & Batch Parameters
    params.update({
        "t_batch_size": 300,         # number of frames to compute per iteration
        "n_proc_corr": 1,           # number of processors for correlation map calculation
        "mproc_batchsize": 5,        # frames per smaller batch within the larger batch
        "n_init_files": 1,           # number of TIFFs used for initialization
    })

    # Registration & Advanced Parameters
    params.update({
        "fuse_shift_override": None, # override for fusing shifts if desired
        "init_n_frames": None,       # number of frames to use for initialization (None = use defaults)
        "override_crosstalk": None,  # override for crosstalk subtraction
        "gpu_reg_batchsize": 10,     # batch size for GPU registration
        "max_rigid_shift_pix": 250,  # maximum rigid shift (in pixels) allowed during registration
        "max_pix": 2500,             # set this very high and forget about it
        "3d_reg": True,              # perform 3D registration
        "gpu_reg": True,             # use GPU acceleration for registration
    })

    return params

def get_job(job_dir: str | os.PathLike, job_id: str | os.PathLike, tif_list: str | os.PathLike | None=None):
    """
    Given a directory and a job_id, return a Job object or create a new job if one does not exist.

    Parameters
    ----------
    job_dir : str, os.PathLike
        Path to the directory containing the job-id directory.

    job_id : str, int
        str name for the job, to be appended as f"s3d-{job_id}"

    tif_list : list[str] or list[os.PathLike], optional
        List of paths to raw tifs, needed to create a new job.

    Returns
    -------
    Job
        Object containing parameters, directories and function entrypoints to the pipeline.
    """
    job_dir = Path(job_dir)
    job_path = job_dir / f"s3d-{job_id}"

    # If tif_list is passed, force recreation
    if tif_list:
        print(f"Forcing new job creation at {job_path}")
        if job_path.exists():
            import shutil
            shutil.rmtree(job_path)
        return Job(job_dir, job_id, create=True, overwrite=True, verbosity=3, tifs=tif_list, params=get_params())

    # Otherwise load existing job
    if not job_path.exists() or not job_path.joinpath("params.npy").exists():
        raise ValueError(f"{job_path} does not exist and no --tif-dir provided to create it.")

    return Job(job_dir, job_id, create=False, overwrite=False)

def run_job(job, do_init, do_register, do_correlate, do_segment):
    results = {
        "init": None,
        "register": None,
        "correlate": None,
        "segment": None,
        "errors": {},
    }

    def run_stage(stage_name, fn):
        try:
            fn()
            results[stage_name] = True
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            location = f"{tb.filename}:{tb.lineno} in {tb.name}"
            results["errors"][stage_name] = f"{type(e).__name__}: {e} (at {location})"
            results[stage_name] = False
            return False
        return True

    if do_init and not run_stage("init", job.run_init_pass):
        return results
    if do_register and not run_stage("register", job.register):
        return results
    if do_correlate and not run_stage("correlate", job.calculate_corr_map):
        return results
    if do_segment and not run_stage("segment", job.segment_rois):
        return results

    return results

def save_job_to_suite2p(job, save_folder: str | Path, framerate: float = 10.0,
                        edge_crop=False, edge_crop_npix=None):
    """
    Save a Job's registered movie into Suite2p-style folders with BinaryFile writers,
    automatically center-padding all planes to the largest (Ly, Lx).
    """
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # --- Load registered movie ---
    movie = job.get_registered_movie(edge_crop=edge_crop, edge_crop_npix=edge_crop_npix)
    n_planes, n_frames, *_ = movie.shape
    print(f"Saving {n_planes} planes")

    # find largest (Ly, Lx)
    shapes = [movie[z].shape[1:] for z in range(n_planes)]
    max_y = max(s[0] for s in shapes)
    max_x = max(s[1] for s in shapes)
    print(f"Center-padding all planes to ({max_y}, {max_x})")

    for z in range(n_planes):
        plane_data = movie[z]  # (t, y, x)

        plane_data = plane_data.astype(np.float32).compute()
        plane_data = np.clip(plane_data, -32768, 32767).astype(np.int16)

        _, y, x = plane_data.shape

        # --- Center pad ---
        pad_y = (max_y - y)
        pad_x = (max_x - x)

        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left

        plane_data = np.pad(
            plane_data,
            pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0
        )

        assert plane_data.shape == (n_frames, max_y, max_x), f"Padding failed for plane {z}"

        plane_folder = save_folder / f"plane{z}"
        plane_folder.mkdir(parents=True, exist_ok=True)
        bin_path = plane_folder / "data.bin"

        with BinaryFile(Ly=max_y, Lx=max_x, filename=str(bin_path), n_frames=n_frames) as bf:
            bf.file[:] = plane_data

        ops = {
            "nframes": n_frames,
            "Lx": max_x,
            "Ly": max_y,
            "nchannels": 1,
            "functional_chan": 1,
            "fs": framerate,
            "save_path0": str(save_folder),
            "save_folder": save_folder.name,
            "plane": z,
        }
        np.save(plane_folder / "ops.npy", ops)

    print("All planes saved and ops.npy created.")

# def save_dask_movie_to_suite2p(movie: da.Array, save_folder: str | Path, framerate: float = 17.0):
#     """
#     Save a (z, t, y, x) Dask movie into Suite2p-style folders with BinaryFile writers,
#     center-cropping all planes to the smallest (Ly, Lx) found across z-planes.
#     """
#     save_folder = Path(save_folder)
#     save_folder.mkdir(parents=True, exist_ok=True)
#
#     n_planes = movie.shape[0]
#     print(f"Saving {n_planes} planes")
#
#     shapes = [movie[z].shape for z in range(n_planes)]
#     mins = np.min(np.array([s[1:] for s in shapes]), axis=0)  # ignore t
#     min_Ly, min_Lx = mins
#     print(f"Center-cropping all planes to ({min_Ly}, {min_Lx})")
#
#     for z in range(n_planes):
#         plane_data = movie[z]  # (t, y, x)
#         n_frames, Ly, Lx = plane_data.shape
#
#         # center-crop to (min_Ly, min_Lx)
#         crop_y_start = (Ly - min_Ly) // 2
#         crop_x_start = (Lx - min_Lx) // 2
#         plane_data = plane_data[:, crop_y_start:crop_y_start+min_Ly, crop_x_start:crop_x_start+min_Lx]
#
#         plane_data = plane_data.astype(np.float32).compute()
#         plane_data = np.clip(plane_data, -32768, 32767).astype(np.int16)
#
#         plane_folder = save_folder / f"plane{z}"
#         plane_folder.mkdir(parents=True, exist_ok=True)
#         bin_path = plane_folder / "data.bin"
#
#         with BinaryFile(Ly=min_Ly, Lx=min_Lx, filename=str(bin_path), n_frames=n_frames) as bf:
#             bf.file[:] = plane_data
#
#         ops = {
#             "nframes": n_frames,
#             "Lx": min_Lx,
#             "Ly": min_Ly,
#             "nchannels": 1,
#             "functional_chan": 1,
#             "fs": framerate,
#             "save_path0": str(save_folder),
#             "save_folder": save_folder.name,
#             "plane": z,
#         }
#         np.save(plane_folder / "ops.npy", ops)
#
#     print("All planes saved and ops.npy created.")
#

fpath = Path(r"D:\W2_DATA\kbarber\2025_03_01\mk301\results")
metadata = mbo.get_metadata(fpath.parent.joinpath("assembled/plane_07_mk301.tiff"))
job_id = "03_27"
job = get_job(fpath, job_id, tif_list=None)

# files = job.get_registered_files()
# movie = job.get_registered_movie(edge_crop=True, edge_crop_npix=128)

save_folder = Path("../data/")
save_folder.mkdir(exist_ok=True)

sub_folder = save_folder.joinpath("plane0")
sub_folder.mkdir(exist_ok=True)
plane_path = save_folder.joinpath(f"plane{8}.bin")
# move the plane_path to sub_folder
plane_path.rename(sub_folder.joinpath(plane_path.name))
ops = suite2p.default_ops()
ops_base = mbo.params_from_metadata(metadata, ops=ops)
db = {
    "data_path": str(sub_folder),
}

# --- Load registered movie ---
# movie_cropped = job.get_registered_movie(edge_crop=True, edge_crop_npix=10)
data = job.get_registered_movie()
# fpl.ImageWidget(data).show()
# fpl.loop.run()

def save_s3d_movie_to_s2p_binary(mov_reg, save_dir, batch_size=500):
    save_dir = Path(save_dir).resolve().expanduser()  # allows ~ expansion and relative paths

    if not save_dir.is_dir():
        raise NotADirectoryError(f"Save directory {save_dir} does not exist.")

    nplanes, nframes, ny, nx = mov_reg.shape

    # chunks arent uniformly sized so da.concatenate fails internally?
    mov_reg = mov_reg.rechunk((1, batch_size, ny, nx))

    for plane_idx in range(nplanes):
        plane_path = save_dir.joinpath(f"plane{plane_idx}.bin")
        with BinaryFile(Ly=ny, Lx=nx, filename=str(plane_path), n_frames=nframes) as bf:
            for start in range(0, nframes, batch_size):
                end = min(start + batch_size, nframes)
                batch = mov_reg[plane_idx, start:end].compute()  # shape (batch_size, ny, nx)
                batch = np.clip(batch, -32768, 32767).astype(np.int16)
                bf[start:end] = batch

save_s3d_movie_to_s2p_binary(data, save_folder)

n_planes, n_frames, Ly, Lx = data.shape
for z in range(n_planes):
    plane_data = data[z]
    bin_path = save_folder.joinpath(f"plane{z}.bin")
    with BinaryFile(Ly=Ly, Lx=Lx, filename=str(bin_path), n_frames=n_frames) as bf:
        bf[:] = plane_data.astype(np.int16)


# fpl.ImageWidget(movie_cropped, names=["Cropped Movie"], histogram_widget=False).show()
x = 4


# Original Dask array shape: (z_planes, time, height, width)
# Transpose to (time, z_planes, height, width)


#
# # find largest (Ly, Lx)
# shapes = [movie[z].shape[1:] for z in range(n_planes)]
# max_y = max(s[0] for s in shapes)
# max_x = max(s[1] for s in shapes)
# print(f"Center-padding all planes to ({max_y}, {max_x})")
#
# for z in range(n_planes):
#     plane_data = movie[z]  # (t, y, x)
#
#     plane_data = plane_data.astype(np.float32).compute()
#     plane_data = np.clip(plane_data, -32768, 32767).astype(np.int16)
#
#     _, y, x = plane_data.shape
#
#     # --- Center pad ---
#     pad_y = (max_y - y)
#     pad_x = (max_x - x)
#
#     pad_top = pad_y // 2
#     pad_bottom = pad_y - pad_top
#     pad_left = pad_x // 2
#     pad_right = pad_x - pad_left
#
#     plane_data = np.pad(
#         plane_data,
#         pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
#         mode="constant",
#         constant_values=0
#     )
#
# # save_job_to_suite2p(job, save_folder, framerate=17, edge_crop=True, edge_crop_npix=10)