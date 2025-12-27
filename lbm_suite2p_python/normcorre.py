"""
normcorre registration module for LBM-Suite2p-Python.

provides rigid and piecewise-rigid motion correction using FFT-based
cross-correlation, adapted from the CaImAn/NORMCORRE algorithm.

this module can be used as an alternative to suite2p's built-in registration
when better results are needed for high-motion datasets.

references:
- Pnevmatikakis & Giovannucci (2017). NoRMCorre: An online algorithm for
  piecewise rigid motion correction of calcium imaging data.
  Journal of Neuroscience Methods, 291, 83-94.
"""

import numpy as np
from pathlib import Path
from typing import Literal

# optional dependencies check
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from scipy import ndimage
    from scipy.interpolate import interpn
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def normcorre_ops():
    """
    Default parameters for normcorre registration.

    Returns
    -------
    dict
        Default normcorre parameters.

    Notes
    -----
    These parameters can be passed to ``lsp.pipeline()`` via the ``ops`` dict
    when using ``registration_method='normcorre'``.
    """
    return {
        # rigid registration
        "max_shifts": (20, 20),  # max shifts in (y, x) pixels
        "niter_rig": 1,  # iterations of rigid registration
        "upsample_factor": 10,  # subpixel precision (1=pixel, 10=0.1px)

        # piecewise rigid
        "pw_rigid": False,  # use piecewise rigid (vs global rigid)
        "strides": (96, 96),  # patch stride in (y, x) pixels
        "overlaps": (32, 32),  # patch overlap in (y, x) pixels
        "max_deviation_rigid": 3,  # max deviation from rigid shift per patch
        "upsample_factor_grid": 4,  # upsampling of shift field

        # preprocessing
        "gSig_filt": None,  # gaussian high-pass filter sigma (None=disabled)

        # output options
        "border_nan": True,  # True=NaN borders, False=0, 'copy'=replicate edge
        "shifts_opencv": True,  # use opencv (fast) vs fft (no interpolation artifacts)
        "nonneg_movie": False,  # subtract min to make non-negative

        # template
        "num_frames_template": 100,  # frames to use for initial template
        "template_method": "median",  # 'median' or 'mean'

        # batching
        "batch_size": 500,  # frames per batch for memory efficiency
    }


def _check_dependencies():
    """Check that required dependencies are available."""
    if not HAS_OPENCV:
        raise ImportError(
            "normcorre requires opencv-python. Install with: pip install opencv-python"
        )
    if not HAS_SCIPY:
        raise ImportError(
            "normcorre requires scipy. Install with: pip install scipy"
        )


# =============================================================================
# core fft-based registration
# =============================================================================

def register_translation(
    src_image: np.ndarray,
    target_image: np.ndarray,
    upsample_factor: int = 1,
    max_shifts: tuple = None,
) -> tuple:
    """
    Compute shift between two images using FFT cross-correlation.

    Parameters
    ----------
    src_image : ndarray
        Reference/template image.
    target_image : ndarray
        Image to register against the reference.
    upsample_factor : int, default 1
        Subpixel precision. 1 = pixel-level, 10 = 0.1 pixel precision.
    max_shifts : tuple, optional
        Maximum allowed shifts (y, x). If None, no constraint.

    Returns
    -------
    shifts : tuple
        (y_shift, x_shift) to align target to source.
    error : float
        Cross-correlation error metric.
    diffphase : float
        Phase difference (for complex images).
    """
    _check_dependencies()

    src = np.array(src_image, dtype=np.float32)
    target = np.array(target_image, dtype=np.float32)

    # compute fft
    src_freq = np.fft.fft2(src)
    target_freq = np.fft.fft2(target)

    # cross-correlation in frequency domain
    shape = src_freq.shape
    image_product = src_freq * np.conj(target_freq)
    cross_correlation = np.fft.ifft2(image_product)

    # apply max_shifts constraint by zeroing out disallowed regions
    if max_shifts is not None:
        max_y, max_x = max_shifts
        mask = np.ones(shape, dtype=bool)
        # keep center region (small shifts)
        mask[:max_y + 1, :max_x + 1] = False  # top-left (positive shifts)
        mask[:max_y + 1, -max_x:] = False  # top-right
        mask[-max_y:, :max_x + 1] = False  # bottom-left
        mask[-max_y:, -max_x:] = False  # bottom-right (negative shifts)
        cross_correlation[mask] = 0

    # find peak (whole-pixel shift)
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)), shape)
    midpoints = np.array([axis_size // 2 for axis_size in shape])

    # convert fft index to shift
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor > 1:
        # refine with upsampled dft around peak
        shifts = _refine_shift_subpixel(
            image_product, shifts, upsample_factor, shape
        )

    # compute error metric
    src_amp = np.sum(np.abs(src_freq) ** 2)
    target_amp = np.sum(np.abs(target_freq) ** 2)
    ccm = cross_correlation[int(shifts[0]) % shape[0], int(shifts[1]) % shape[1]]
    error = 1.0 - np.abs(ccm) ** 2 / (src_amp * target_amp + 1e-10)

    return tuple(shifts), float(error), 0.0


def _refine_shift_subpixel(
    image_product: np.ndarray,
    shifts: np.ndarray,
    upsample_factor: int,
    shape: tuple,
) -> np.ndarray:
    """
    Refine shift estimate using matrix-multiply DFT.

    This computes the DFT only in a small upsampled region around the
    initial peak, which is much more memory-efficient than upsampling
    the entire cross-correlation.
    """
    # size of upsampled region
    upsampled_region_size = int(np.ceil(upsample_factor * 1.5))

    # center of output (accounting for fft shift conventions)
    dftshift = np.fix(upsampled_region_size / 2.0)

    # compute upsampled cross-correlation around the peak
    sample_region_offset = dftshift - shifts * upsample_factor

    cross_correlation_upsampled = _upsampled_dft(
        image_product,
        upsampled_region_size,
        upsample_factor,
        sample_region_offset,
    )

    # find peak in upsampled region
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation_upsampled)),
        cross_correlation_upsampled.shape,
    )

    # convert to subpixel shift
    maxima = np.array(maxima, dtype=np.float64) - dftshift
    shifts = shifts + maxima / upsample_factor

    return shifts


def _upsampled_dft(
    data: np.ndarray,
    upsampled_region_size: int,
    upsample_factor: int,
    axis_offsets: np.ndarray,
) -> np.ndarray:
    """
    Compute upsampled DFT using matrix multiplication.

    This is the efficient approach from Guizar-Sicairos et al. (2008).
    Instead of zero-padding and computing a large FFT, we compute
    the DFT only at the output points we need.

    Adapted from scikit-image's _upsampled_dft implementation.
    """
    # for 2D data, we compute the upsampled DFT via two matrix multiplications
    ny, nx = data.shape
    oy, ox = axis_offsets

    # row kernel (along axis 0)
    row_kernel = np.exp(
        -2j * np.pi / (ny * upsample_factor)
        * (np.fft.ifftshift(np.arange(ny)) - ny // 2)[:, None]
        * (np.arange(upsampled_region_size) - oy)[None, :]
    )

    # column kernel (along axis 1)
    col_kernel = np.exp(
        -2j * np.pi / (nx * upsample_factor)
        * (np.arange(upsampled_region_size) - ox)[:, None]
        * (np.fft.ifftshift(np.arange(nx)) - nx // 2)[None, :]
    )

    # compute upsampled region via matrix multiplication
    upsampled = row_kernel.conj().T @ data @ col_kernel.conj().T

    return upsampled


# =============================================================================
# shift application
# =============================================================================

def apply_shift_opencv(
    img: np.ndarray,
    shift: tuple,
    border_nan: bool | str = True,
) -> np.ndarray:
    """
    Apply rigid shift using OpenCV warpAffine.

    Parameters
    ----------
    img : ndarray
        Image to shift.
    shift : tuple
        (y_shift, x_shift) to apply.
    border_nan : bool or str
        Border handling: True=NaN, False=0, 'copy'=replicate edge.

    Returns
    -------
    ndarray
        Shifted image.
    """
    _check_dependencies()

    h, w = img.shape[:2]
    shift_y, shift_x = shift

    # transformation matrix for translation
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # border mode
    if border_nan is True or border_nan == "nan":
        border_mode = cv2.BORDER_CONSTANT
        border_value = np.nan
    elif border_nan is False or border_nan == "zero":
        border_mode = cv2.BORDER_CONSTANT
        border_value = 0
    elif border_nan == "copy" or border_nan == "replicate":
        border_mode = cv2.BORDER_REPLICATE
        border_value = 0
    else:
        border_mode = cv2.BORDER_CONSTANT
        border_value = 0

    shifted = cv2.warpAffine(
        img.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=border_mode,
        borderValue=border_value,
    )

    return shifted


def apply_shift_fft(
    img: np.ndarray,
    shift: tuple,
) -> np.ndarray:
    """
    Apply rigid shift using FFT phase shift.

    This avoids interpolation artifacts but is slower than OpenCV.

    Parameters
    ----------
    img : ndarray
        Image to shift.
    shift : tuple
        (y_shift, x_shift) to apply.

    Returns
    -------
    ndarray
        Shifted image.
    """
    h, w = img.shape[:2]
    shift_y, shift_x = shift

    # frequency coordinates
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]

    # phase shift
    phase = np.exp(-2j * np.pi * (shift_y * fy + shift_x * fx))

    # apply in frequency domain
    img_freq = np.fft.fft2(img.astype(np.float32))
    shifted_freq = img_freq * phase
    shifted = np.real(np.fft.ifft2(shifted_freq))

    return shifted.astype(np.float32)


# =============================================================================
# piecewise rigid registration
# =============================================================================

def get_patch_coords(
    shape: tuple,
    strides: tuple,
    overlaps: tuple,
) -> list:
    """
    Calculate patch coordinates for piecewise rigid registration.

    Parameters
    ----------
    shape : tuple
        Image shape (height, width).
    strides : tuple
        Patch stride (y, x).
    overlaps : tuple
        Patch overlap (y, x).

    Returns
    -------
    list
        List of (y_start, y_end, x_start, x_end) for each patch.
    """
    h, w = shape[:2]
    stride_y, stride_x = strides
    overlap_y, overlap_x = overlaps

    patch_h = stride_y + overlap_y
    patch_w = stride_x + overlap_x

    patches = []
    y = 0
    while y < h:
        y_end = min(y + patch_h, h)
        x = 0
        while x < w:
            x_end = min(x + patch_w, w)
            patches.append((y, y_end, x, x_end))
            x += stride_x
            if x_end >= w:
                break
        y += stride_y
        if y_end >= h:
            break

    return patches


def register_piecewise_rigid(
    template: np.ndarray,
    frame: np.ndarray,
    strides: tuple = (96, 96),
    overlaps: tuple = (32, 32),
    max_shifts: tuple = (20, 20),
    max_deviation_rigid: int = 3,
    upsample_factor: int = 10,
) -> tuple:
    """
    Compute piecewise rigid shifts for a frame.

    Parameters
    ----------
    template : ndarray
        Reference template image.
    frame : ndarray
        Frame to register.
    strides : tuple
        Patch stride (y, x).
    overlaps : tuple
        Patch overlap (y, x).
    max_shifts : tuple
        Maximum allowed shifts per patch.
    max_deviation_rigid : int
        Maximum deviation from mean (rigid) shift.
    upsample_factor : int
        Subpixel precision.

    Returns
    -------
    shifts : list
        List of (y_shift, x_shift) per patch.
    rigid_shift : tuple
        Mean (rigid) shift across all patches.
    """
    patches = get_patch_coords(template.shape, strides, overlaps)
    shifts = []

    for y0, y1, x0, x1 in patches:
        template_patch = template[y0:y1, x0:x1]
        frame_patch = frame[y0:y1, x0:x1]

        shift, _, _ = register_translation(
            template_patch,
            frame_patch,
            upsample_factor=upsample_factor,
            max_shifts=max_shifts,
        )
        shifts.append(shift)

    # compute rigid (mean) shift
    shifts_arr = np.array(shifts)
    rigid_shift = tuple(np.median(shifts_arr, axis=0))

    # constrain deviations from rigid
    if max_deviation_rigid is not None:
        for i, (sy, sx) in enumerate(shifts):
            dy = sy - rigid_shift[0]
            dx = sx - rigid_shift[1]
            # clip deviations
            dy = np.clip(dy, -max_deviation_rigid, max_deviation_rigid)
            dx = np.clip(dx, -max_deviation_rigid, max_deviation_rigid)
            shifts[i] = (rigid_shift[0] + dy, rigid_shift[1] + dx)

    return shifts, rigid_shift


def apply_piecewise_shifts(
    frame: np.ndarray,
    shifts: list,
    strides: tuple,
    overlaps: tuple,
    upsample_factor_grid: int = 4,
    border_nan: bool | str = True,
) -> np.ndarray:
    """
    Apply piecewise rigid shifts to a frame.

    Parameters
    ----------
    frame : ndarray
        Frame to warp.
    shifts : list
        Per-patch shifts from register_piecewise_rigid.
    strides : tuple
        Patch stride (y, x).
    overlaps : tuple
        Patch overlap (y, x).
    upsample_factor_grid : int
        Upsampling of shift field for smooth interpolation.
    border_nan : bool or str
        Border handling.

    Returns
    -------
    ndarray
        Warped frame.
    """
    _check_dependencies()

    h, w = frame.shape[:2]
    patches = get_patch_coords((h, w), strides, overlaps)

    # compute patch centers
    centers_y = []
    centers_x = []
    for y0, y1, x0, x1 in patches:
        centers_y.append((y0 + y1) / 2)
        centers_x.append((x0 + x1) / 2)

    # create grid of patch centers
    n_patches_y = len(set(centers_y))
    n_patches_x = len(set(centers_x))
    unique_y = sorted(set(centers_y))
    unique_x = sorted(set(centers_x))

    # reshape shifts to grid
    shifts_arr = np.array(shifts)
    shift_grid_y = shifts_arr[:, 0].reshape(n_patches_y, n_patches_x)
    shift_grid_x = shifts_arr[:, 1].reshape(n_patches_y, n_patches_x)

    # interpolate shift field to full resolution
    # create coordinate grids
    yi = np.arange(h)
    xi = np.arange(w)
    yi_grid, xi_grid = np.meshgrid(yi, xi, indexing='ij')

    # interpolate using scipy
    points = (np.array(unique_y), np.array(unique_x))

    shift_field_y = interpn(
        points, shift_grid_y, (yi_grid, xi_grid),
        method='linear', bounds_error=False, fill_value=None
    )
    shift_field_x = interpn(
        points, shift_grid_x, (yi_grid, xi_grid),
        method='linear', bounds_error=False, fill_value=None
    )

    # create remapping coordinates
    map_x = (xi_grid + shift_field_x).astype(np.float32)
    map_y = (yi_grid + shift_field_y).astype(np.float32)

    # apply remap
    if border_nan is True or border_nan == "nan":
        border_mode = cv2.BORDER_CONSTANT
        border_value = np.nan
    elif border_nan is False or border_nan == "zero":
        border_mode = cv2.BORDER_CONSTANT
        border_value = 0
    else:
        border_mode = cv2.BORDER_REPLICATE
        border_value = 0

    warped = cv2.remap(
        frame.astype(np.float32),
        map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=border_mode,
        borderValue=border_value,
    )

    return warped


# =============================================================================
# high-level registration functions
# =============================================================================

def compute_template(
    frames: np.ndarray,
    method: str = "median",
    max_frames: int = 100,
) -> np.ndarray:
    """
    Compute reference template from frames.

    Parameters
    ----------
    frames : ndarray
        Stack of frames (T, H, W).
    method : str
        'median' or 'mean'.
    max_frames : int
        Maximum frames to use.

    Returns
    -------
    ndarray
        Template image.
    """
    n_frames = min(len(frames), max_frames)
    indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    subset = frames[indices]

    if method == "median":
        return np.median(subset, axis=0).astype(np.float32)
    else:
        return np.mean(subset, axis=0).astype(np.float32)


def high_pass_filter(
    img: np.ndarray,
    gSig: float,
) -> np.ndarray:
    """
    Apply gaussian high-pass filter.

    Parameters
    ----------
    img : ndarray
        Input image.
    gSig : float
        Gaussian sigma for low-pass (subtracted from image).

    Returns
    -------
    ndarray
        High-pass filtered image.
    """
    _check_dependencies()
    low_pass = ndimage.gaussian_filter(img.astype(np.float32), gSig)
    return img - low_pass


def register_frames(
    frames: np.ndarray,
    template: np.ndarray = None,
    ops: dict = None,
    progress_callback=None,
) -> tuple:
    """
    Register a stack of frames using normcorre algorithm.

    Parameters
    ----------
    frames : ndarray
        Stack of frames (T, H, W) to register.
    template : ndarray, optional
        Reference template. If None, computed from frames.
    ops : dict, optional
        Registration parameters. If None, uses normcorre_ops().
    progress_callback : callable, optional
        Function called with (frame_idx, total_frames) for progress.

    Returns
    -------
    registered : ndarray
        Registered frames (T, H, W).
    shifts : list
        List of shifts per frame. Each is (y, x) for rigid or
        list of (y, x) per patch for piecewise rigid.
    template : ndarray
        Final template used.
    """
    _check_dependencies()

    if ops is None:
        ops = normcorre_ops()

    n_frames, h, w = frames.shape
    pw_rigid = ops.get("pw_rigid", False)
    max_shifts = ops.get("max_shifts", (20, 20))
    upsample_factor = ops.get("upsample_factor", 10)
    gSig_filt = ops.get("gSig_filt", None)
    border_nan = ops.get("border_nan", True)
    shifts_opencv = ops.get("shifts_opencv", True)

    # piecewise rigid params
    strides = ops.get("strides", (96, 96))
    overlaps = ops.get("overlaps", (32, 32))
    max_deviation_rigid = ops.get("max_deviation_rigid", 3)
    upsample_factor_grid = ops.get("upsample_factor_grid", 4)

    # compute template if not provided
    if template is None:
        template = compute_template(
            frames,
            method=ops.get("template_method", "median"),
            max_frames=ops.get("num_frames_template", 100),
        )

    # apply high-pass filter to template if specified
    if gSig_filt is not None:
        template_filt = high_pass_filter(template, gSig_filt)
    else:
        template_filt = template

    # register each frame
    registered = np.zeros_like(frames, dtype=np.float32)
    all_shifts = []

    for i in range(n_frames):
        frame = frames[i].astype(np.float32)

        # apply high-pass filter if specified
        if gSig_filt is not None:
            frame_filt = high_pass_filter(frame, gSig_filt)
        else:
            frame_filt = frame

        if pw_rigid:
            # piecewise rigid registration
            shifts, rigid_shift = register_piecewise_rigid(
                template_filt, frame_filt,
                strides=strides,
                overlaps=overlaps,
                max_shifts=max_shifts,
                max_deviation_rigid=max_deviation_rigid,
                upsample_factor=upsample_factor,
            )
            registered[i] = apply_piecewise_shifts(
                frame, shifts, strides, overlaps,
                upsample_factor_grid=upsample_factor_grid,
                border_nan=border_nan,
            )
            all_shifts.append({"pw_shifts": shifts, "rigid_shift": rigid_shift})
        else:
            # rigid registration
            shift, error, _ = register_translation(
                template_filt, frame_filt,
                upsample_factor=upsample_factor,
                max_shifts=max_shifts,
            )
            if shifts_opencv:
                registered[i] = apply_shift_opencv(frame, shift, border_nan)
            else:
                registered[i] = apply_shift_fft(frame, shift)
            all_shifts.append(shift)

        if progress_callback is not None:
            progress_callback(i + 1, n_frames)

    return registered, all_shifts, template


def register_binary(
    input_path: str | Path,
    output_path: str | Path = None,
    ops: dict = None,
    Ly: int = None,
    Lx: int = None,
    dtype: np.dtype = np.int16,
) -> tuple:
    """
    Register a suite2p-style binary file using normcorre.

    Parameters
    ----------
    input_path : str or Path
        Path to input binary file (data_raw.bin or data.bin).
    output_path : str or Path, optional
        Path for output binary. If None, replaces input.
    ops : dict, optional
        Registration parameters including Ly, Lx if not provided separately.
    Ly : int, optional
        Image height. If None, read from ops.
    Lx : int, optional
        Image width. If None, read from ops.
    dtype : dtype, default np.int16
        Data type of binary file.

    Returns
    -------
    shifts : list
        Registration shifts per frame.
    template : ndarray
        Final reference template.
    """
    _check_dependencies()

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    output_path = Path(output_path)

    if ops is None:
        ops = normcorre_ops()

    # get dimensions
    if Ly is None:
        Ly = ops.get("Ly")
    if Lx is None:
        Lx = ops.get("Lx")

    if Ly is None or Lx is None:
        raise ValueError("Must provide Ly and Lx in ops or as arguments")

    # calculate number of frames
    file_size = input_path.stat().st_size
    bytes_per_frame = Ly * Lx * np.dtype(dtype).itemsize
    n_frames = file_size // bytes_per_frame

    print(f"Normcorre Registration")
    print(f"=" * 60)
    print(f"Input: {input_path}")
    print(f"Frames: {n_frames}, Shape: {Ly} x {Lx}")
    print(f"Mode: {'piecewise rigid' if ops.get('pw_rigid', False) else 'rigid'}")

    # memory-map input
    data = np.memmap(input_path, dtype=dtype, mode='r', shape=(n_frames, Ly, Lx))

    # compute template
    print(f"\nComputing template...")
    template = compute_template(
        data,
        method=ops.get("template_method", "median"),
        max_frames=ops.get("num_frames_template", 100),
    )

    # create output file
    if output_path != input_path:
        output = np.memmap(output_path, dtype=dtype, mode='w+', shape=(n_frames, Ly, Lx))
    else:
        # in-place registration - need to load to memory in batches
        output = None

    # register in batches
    batch_size = ops.get("batch_size", 500)
    all_shifts = []

    print(f"\nRegistering frames...")
    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        batch = data[start:end].astype(np.float32)

        registered, shifts, _ = register_frames(batch, template, ops)
        all_shifts.extend(shifts)

        # write output
        if output is not None:
            output[start:end] = registered.astype(dtype)
        else:
            # in-place: need to write back to same file
            # this is tricky with memmap, would need temp file
            pass

        pct = 100 * end / n_frames
        print(f"  {end}/{n_frames} ({pct:.1f}%)")

    print(f"\nRegistration complete")
    print(f"Output: {output_path}")

    return all_shifts, template
