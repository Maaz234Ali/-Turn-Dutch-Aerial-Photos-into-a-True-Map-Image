"""
projection.py
Vectorized collinearity projection and sub-pixel image sampling.
Maps Dutch RD New ground coordinates (X, Y, Z_DSM) into UltraCam aerial photo space.

GPU dispatch:
  - project_ground_to_camera() will use the CUDA collinearity kernel when a
    CUDA-capable GPU is present, falling back to NumPy otherwise.
  - sample_image_bilinear() remains on the CPU (rasterio I/O + OpenCV remap)
    because aerial image I/O is disk-bound, not compute-bound.
"""

import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from camera_parser import Camera

# ---------------------------------------------------------------------------
# Optional GPU dependency detection
# ---------------------------------------------------------------------------
try:
    from numba import cuda as numba_cuda
    HAS_CUDA = numba_cuda.is_available()
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from gpu_kernels import _project_collinearity_cuda

_CUDA_THREADS_1D = 256   # threads per block for the 1-D projection kernel


# ---------------------------------------------------------------------------
# Public collinearity projection
# ---------------------------------------------------------------------------

def project_ground_to_camera(
    camera: Camera,
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    Z_grid: np.ndarray,
    use_gpu: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects a 2D or 1D ground coordinate grid (X, Y, Z) in EPSG:28992
    into camera image coordinates using the collinearity equations.

    GPU path:  launches _project_collinearity_cuda; flattens the 2-D grid,
               runs one CUDA thread per pixel, and reshapes results.
    CPU path:  vectorised NumPy matrix multiply (original implementation).

    Returns:
        cols:       Sub-pixel column indices in the aerial photo  (shape = Z_grid.shape)
        rows:       Sub-pixel row indices in the aerial photo     (shape = Z_grid.shape)
        view_angles: Off-nadir zenith angles (degrees)           (shape = Z_grid.shape)
        in_bounds:  Boolean mask where projection is inside sensor margins
    """
    shape = Z_grid.shape
    N = Z_grid.size

    if use_gpu and HAS_CUDA:
        # ---------------------------------------------------------------
        # GPU path – one thread per pixel
        # ---------------------------------------------------------------
        X_flat = np.ascontiguousarray(X_grid.ravel(), dtype=np.float64)
        Y_flat = np.ascontiguousarray(Y_grid.ravel(), dtype=np.float64)
        Z_flat = np.ascontiguousarray(Z_grid.ravel(), dtype=np.float32)
        R_flat = np.ascontiguousarray(camera.R.ravel(), dtype=np.float64)

        # Allocate device arrays
        d_X       = numba_cuda.to_device(X_flat)
        d_Y       = numba_cuda.to_device(Y_flat)
        d_Z       = numba_cuda.to_device(Z_flat)
        d_R       = numba_cuda.to_device(R_flat)
        d_cols    = numba_cuda.device_array(N, dtype=np.float32)
        d_rows    = numba_cuda.device_array(N, dtype=np.float32)
        d_angles  = numba_cuda.device_array(N, dtype=np.float32)
        d_valid   = numba_cuda.device_array(N, dtype=np.bool_)

        blocks = (N + _CUDA_THREADS_1D - 1) // _CUDA_THREADS_1D

        _project_collinearity_cuda[blocks, _CUDA_THREADS_1D](
            d_X, d_Y, d_Z, d_R,
            np.float64(camera.X0),
            np.float64(camera.Y0),
            np.float64(camera.Z0),
            np.float64(camera.f_m),
            np.float64(camera.pixel_size_m),
            np.float64(camera.x_ppa_m),
            np.float64(camera.y_ppa_m),
            np.float64(camera.c_mid),
            np.float64(camera.r_mid),
            np.int32(camera.width),
            np.int32(camera.height),
            np.float64(5.0),   # margin pixels
            d_cols, d_rows, d_angles, d_valid
        )
        numba_cuda.synchronize()

        cols      = d_cols.copy_to_host().reshape(shape)
        rows      = d_rows.copy_to_host().reshape(shape)
        view_angles = d_angles.copy_to_host().reshape(shape)
        in_bounds = d_valid.copy_to_host().reshape(shape)

        # Free device memory immediately
        del d_X, d_Y, d_Z, d_R, d_cols, d_rows, d_angles, d_valid
        numba_cuda.current_context().memory_manager.deallocations.clear()

    else:
        # ---------------------------------------------------------------
        # CPU path – original vectorised NumPy implementation
        # ---------------------------------------------------------------
        dX = (X_grid - camera.X0).ravel()
        dY = (Y_grid - camera.Y0).ravel()
        dZ = (Z_grid - camera.Z0).ravel()

        # Ground distance and off-nadir angle
        horiz_dist = np.hypot(dX, dY)
        vert_dist  = np.maximum(-dZ, 1.0)
        view_angles = np.degrees(np.arctan2(horiz_dist, vert_dist)).reshape(shape)

        # Transform to camera coordinate system: cam_coords = R @ [dX, dY, dZ]
        pts = np.stack([dX, dY, dZ], axis=0)  # (3, N)
        cam_coords = camera.R @ pts            # (3, N)

        Xc = cam_coords[0]
        Yc = cam_coords[1]
        Zc = cam_coords[2]

        # Guard against points behind camera (Zc >= 0)
        valid_z = Zc < -1.0
        safe_Zc = np.where(valid_z, Zc, -1.0)

        # Perspective projection onto focal plane (metres)
        x_m = -camera.f_m * (Xc / safe_Zc)
        y_m = -camera.f_m * (Yc / safe_Zc)

        u = y_m - camera.y_ppa_m
        v = x_m - camera.x_ppa_m

        cols = (camera.c_mid + u / camera.pixel_size_m).reshape(shape).astype(np.float32)
        rows = (camera.r_mid + v / camera.pixel_size_m).reshape(shape).astype(np.float32)

        margin = 5.0
        in_bounds = (
            (cols >= margin) &
            (cols < camera.width  - margin) &
            (rows >= margin) &
            (rows < camera.height - margin) &
            valid_z.reshape(shape)
        )

    return cols, rows, view_angles.astype(np.float32), in_bounds


# ---------------------------------------------------------------------------
# Sub-pixel image sampling (CPU – disk I/O bound, not compute bound)
# ---------------------------------------------------------------------------

def sample_image_bilinear(
    camera: Camera,
    cols: np.ndarray,
    rows: np.ndarray,
    mask: np.ndarray = None
) -> np.ndarray:
    """
    Samples RGB color values from the aerial GeoTIFF using bilinear interpolation.
    Efficiently crops a sub-window of the large aerial photo to minimise RAM overhead.

    Returns:
        rgb_sampled: (H, W, 3) uint8 RGB array
    """
    H, W = cols.shape
    rgb_sampled = np.zeros((H, W, 3), dtype=np.uint8)

    if mask is not None:
        if not np.any(mask):
            return rgb_sampled
        valid_cols = cols[mask]
        valid_rows = rows[mask]
    else:
        valid_cols = cols.ravel()
        valid_rows = rows.ravel()

    c_min = max(0, int(np.floor(np.min(valid_cols))) - 4)
    c_max = min(camera.width,  int(np.ceil(np.max(valid_cols))) + 4)
    r_min = max(0, int(np.floor(np.min(valid_rows))) - 4)
    r_max = min(camera.height, int(np.ceil(np.max(valid_rows))) + 4)

    win_w = c_max - c_min
    win_h = r_max - r_min

    if win_w <= 0 or win_h <= 0:
        return rgb_sampled

    # Read window from disk
    with rasterio.open(camera.image_path) as src:
        win  = Window(c_min, r_min, win_w, win_h)
        data = src.read([1, 2, 3], window=win)  # (3, win_h, win_w)

    # OpenCV remap requires coordinates relative to cropped window
    map_x = cols - c_min
    map_y = rows - r_min

    # Remap each band with bilinear interpolation
    for b in range(3):
        sampled_band = cv2.remap(
            data[b],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        rgb_sampled[:, :, b] = sampled_band

    if mask is not None:
        rgb_sampled[~mask] = 0

    return rgb_sampled
