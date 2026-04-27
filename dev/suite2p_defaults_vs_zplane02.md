# Suite2p Defaults vs Christian zplane02

Source: `christian_40k_segmentation/zplane02_tp00001-08440/ops.npy`
Suite2p `default_ops()` keys: 89  |  Present in zplane02: 88  |  Differ from default: 13

| Parameter | Suite2p default | zplane02 value | Differs |
|---|---|---|---|
| `1Preg` | False | False |  |
| `align_by_chan` | 1 | 1 |  |
| `allow_overlap` | False | False |  |
| `anatomical_only` | 0 | 4 |  yes  |
| `aspect` | 1 | 1 |  |
| `baseline` | `maximin` | `maximin` |  |
| `batch_size` | 500 | 500 |  |
| `bidi_corrected` | False | False |  |
| `bidiphase` | 0 | 0 |  |
| `block_size` | [128, 128] | [128, 128] |  |
| `bruker` | False | False |  |
| `bruker_bidirectional` | False | False |  |
| `cellprob_threshold` | 0 | -6 |  yes  |
| `chan2_thres` | 0.65 | 0.1 |  yes  |
| `classifier_path` | `""` | `""` |  |
| `combined` | True | True |  |
| `connected` | True | True |  |
| `delete_bin` | False | False |  |
| `denoise` | False | False |  |
| `diameter` | 0 | 6.07651 |  yes  |
| `do_bidiphase` | False | False |  |
| `do_registration` | True | 1 |  |
| `fast_disk` | [] | [] |  |
| `flow_threshold` | 0.4 | 0 |  yes  |
| `force_refImg` | False | False |  |
| `force_sktiff` | False | False |  |
| `frames_include` | -1 | -1 |  |
| `fs` | 10 | 10 |  |
| `functional_chan` | 1 | 1 |  |
| `h5py` | [] | [] |  |
| `h5py_key` | `data` | `data` |  |
| `high_pass` | 100 | 100 |  |
| `ignore_flyback` | [] | [] |  |
| `inner_neuropil_radius` | 2 | 2 |  |
| `keep_movie_raw` | False | False |  |
| `lam_percentile` | 50 | 0 |  yes  |
| `look_one_level_down` | False | False |  |
| `max_iterations` | 20 | 20 |  |
| `max_overlap` | 0.75 | 1 |  yes  |
| `maxregshift` | 0.1 | 0.1 |  |
| `maxregshiftNR` | 5 | 5 |  |
| `mesoscan` | False | False |  |
| `min_neuropil_pixels` | 350 | 0 |  yes  |
| `move_bin` | False | False |  |
| `multiplane_parallel` | False | False |  |
| `nbinned` | 5000 | 5000 |  |
| `nchannels` | 1 | 1 |  |
| `neucoeff` | 0.7 | 0.7 |  |
| `neuropil_extract` | True | True |  |
| `nimg_init` | 300 | 300 |  |
| `nonrigid` | True | True |  |
| `norm_frames` | True | True |  |
| `nplanes` | 1 | 1 |  |
| `nwb_driver` | `""` | `""` |  |
| `nwb_file` | `""` | `""` |  |
| `nwb_series` | `""` | `""` |  |
| `pad_fft` | False | False |  |
| `prctile_baseline` | 8 | 8 |  |
| `pre_smooth` | 0 | 0 |  |
| `preclassify` | 0 | 0 |  |
| `pretrained_model` | `cpsam` | `cpsam` |  |
| `reg_tif` | False | False |  |
| `reg_tif_chan2` | False | False |  |
| `roidetect` | True | 1 |  |
| `save_NWB` | False | False |  |
| `save_folder` | [] | [] |  |
| `save_mat` | False | False |  |
| `save_path0` | `""` | `""` |  |
| `sig_baseline` | 10 | 10 |  |
| `smooth_sigma` | 1.15 | 1.15 |  |
| `smooth_sigma_time` | 0 | 0 |  |
| `snr_thresh` | 1.2 | 1.2 |  |
| `soma_crop` | True | True |  |
| `sparse_mode` | True | True |  |
| `spatial_hp_cp` | 0 | 3 |  yes  |
| `spatial_hp_detect` | 25 | 25 |  |
| `spatial_hp_reg` | 42 | 42 |  |
| `spatial_scale` | 0 | 1 |  yes  |
| `spatial_taper` | 40 | 40 |  |
| `spikedetect` | True | True |  |
| `subfolders` | [] | [] |  |
| `subpixel` | 10 | 10 |  |
| `suite2p_version` | `1.0.5` | *missing* |  yes  |
| `tau` | 1 | 1.3 |  yes  |
| `th_badframes` | 1 | 1 |  |
| `threshold_scaling` | 1 | 1 |  |
| `two_step_registration` | False | 1 |  yes  |
| `use_builtin_classifier` | False | False |  |
| `win_baseline` | 60 | 60 |  |

## Differences only

| Parameter | Suite2p default | zplane02 value |
|---|---|---|
| `anatomical_only` | 0 | 4 |
| `cellprob_threshold` | 0 | -6 |
| `chan2_thres` | 0.65 | 0.1 |
| `diameter` | 0 | 6.07651 |
| `flow_threshold` | 0.4 | 0 |
| `lam_percentile` | 50 | 0 |
| `max_overlap` | 0.75 | 1 |
| `min_neuropil_pixels` | 350 | 0 |
| `spatial_hp_cp` | 0 | 3 |
| `spatial_scale` | 0 | 1 |
| `suite2p_version` | `1.0.5` | *missing* |
| `tau` | 1 | 1.3 |
| `two_step_registration` | False | 1 |

Worker starting (pid=42480)
2026-04-26 21:18:16,995 - mbo.worker - INFO - Worker started: task=suite2p, pid=42480
Worker started: task=suite2p, pid=42480
2026-04-26 21:18:18,033 - mbo.worker - INFO - Applied custom metadata to ops: ['animal_model', 'dz', 'test']
Applied custom metadata to ops: ['animal_model', 'dz', 'test']

Counting frames:   0%|          | 0/2 [00:00<?, ?it/s]
Counting frames: 100%|##########| 2/2 [00:00<?, ?it/s]
2026-04-26 21:18:18,063 - mbo.worker - INFO - task_suite2p: source fs=14.0, dz=15.0, input_path='D:\\mbo_studio_demo\\raw'
task_suite2p: source fs=14.0, dz=15.0, input_path='D:\\mbo_studio_demo\\raw'
2026-04-26 21:18:18,063 - mbo.worker - INFO - task_suite2p: applied reactive metadata -> fs=7.0, dz=30.0 (t-stride from 787 indices, z-stride from 7 planes)
task_suite2p: applied reactive metadata -> fs=7.0, dz=30.0 (t-stride from 787 indices, z-stride from 7 planes)
2026-04-26 21:18:18,063 - mbo.worker - INFO - Input: D:\mbo_studio_demo\raw
Input: D:\mbo_studio_demo\raw
2026-04-26 21:18:18,063 - mbo.worker - INFO - Output: D:\mbo_studio_demo\raw_to_cellpose
Output: D:\mbo_studio_demo\raw_to_cellpose
2026-04-26 21:18:18,063 - mbo.worker - INFO - Planes: [1, 3, 5, 7, 9, 11, 13]
Planes: [1, 3, 5, 7, 9, 11, 13]
2026-04-26 21:18:18,063 - mbo.worker - INFO - task_suite2p: computing axial plane shifts...
task_suite2p: computing axial plane shifts...
2026-04-26 21:19:14,363 - mbo.worker - INFO - task_suite2p: axial shifts wired into ops (apply_shift=True, 14 planes)
task_suite2p: axial shifts wired into ops (apply_shift=True, 14 planes)
Loading input to determine dimensions...

Counting frames:   0%|          | 0/2 [00:00<?, ?it/s]
Counting frames: 100%|##########| 2/2 [00:00<?, ?it/s]
Delegating to run_volume (volumetric input detected)...
Processing 7 planes in volume (Total planes: 14)
Output: D:\mbo_studio_demo\raw_to_cellpose

--- Volume Step: Plane 1 ---
Importing suite2p packages...
Writing binary to D:\mbo_studio_demo\raw_to_cellpose\zplane01_tp00001-00787...
  Applying axial shift for plane 1: [0 0]
NOTE: running registration-only pass (detection deferred)
WARNING: skipping cell detection (settings['run']['do_detection']=False)
  Clamped valid region: yrange=[102, 545] xrange=[6, 447] (pad=[97, 550],[5, 448]; motion_trim=T5/B5/L1/R1)
  Updated reg_outputs.npy with clamped valid region
  Computing dF/F...
  Computing ROI statistics...
Plotting results for 38 accepted / 2 rejected ROIs
1500 frames needed for registration metrics, found 787. Skipping PC metrics.

--- Volume Step: Plane 3 ---
Importing suite2p packages...
Writing binary to D:\mbo_studio_demo\raw_to_cellpose\zplane03_tp00001-00787...
  Applying axial shift for plane 3: [16 -2]
NOTE: running registration-only pass (detection deferred)
WARNING: skipping cell detection (settings['run']['do_detection']=False)
  Clamped valid region: yrange=[104, 543] xrange=[6, 447] (pad=[97, 550],[5, 448]; motion_trim=T7/B7/L1/R1)
  Updated reg_outputs.npy with clamped valid region
  Computing dF/F...
  Computing ROI statistics...
Plotting results for 66 accepted / 8 rejected ROIs
1500 frames needed for registration metrics, found 787. Skipping PC metrics.

--- Volume Step: Plane 5 ---
Importing suite2p packages...
Writing binary to D:\mbo_studio_demo\raw_to_cellpose\zplane05_tp00001-00787...
  Applying axial shift for plane 5: [32 -4]
NOTE: running registration-only pass (detection deferred)
WARNING: skipping cell detection (settings['run']['do_detection']=False)
  Clamped valid region: yrange=[104, 543] xrange=[6, 447] (pad=[97, 550],[5, 448]; motion_trim=T7/B7/L1/R1)
  Updated reg_outputs.npy with clamped valid region
  Computing dF/F...
  Computing ROI statistics...
Plotting results for 48 accepted / 0 rejected ROIs
  No rejected ROIs - skipping rejected trace plots
1500 frames needed for registration metrics, found 787. Skipping PC metrics.

--- Volume Step: Plane 7 ---
Importing suite2p packages...
Writing binary to D:\mbo_studio_demo\raw_to_cellpose\zplane07_tp00001-00787...
  Applying axial shift for plane 7: [48 -4]
NOTE: running registration-only pass (detection deferred)
WARNING: skipping cell detection (settings['run']['do_detection']=False)
  Clamped valid region: yrange=[102, 545] xrange=[6, 447] (pad=[97, 550],[5, 448]; motion_trim=T5/B5/L1/R1)
  Updated reg_outputs.npy with clamped valid region
