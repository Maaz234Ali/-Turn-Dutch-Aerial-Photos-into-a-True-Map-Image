"""
visibility.py
Line-of-Sight (LOS) raycasting across the Digital Surface Model (DSM)
to detect building occlusions and compute camera visibility masks.

GPU dispatch hierarchy:
  1. Numba CUDA (@cuda.jit) – fastest; used when CUDA GPU is available.
  2. Numba CPU  (@njit parallel) – fallback when GPU is absent.
  3. Pure NumPy – last resort when Numba is not installed at all.

The public API is unchanged: call compute_visibility_mask() exactly as
before.  The GPU path accepts a pre-allocated device array for the visible
output so that chunked callers (mosaic.py) can avoid redundant H2D/D2H
transfers.
"""

import numpy as np
from camera_parser import Camera

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    from numba import cuda as numba_cuda
    HAS_CUDA = numba_cuda.is_available()
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from gpu_kernels import _compute_los_visibility_cuda

# CUDA launch config tuning constants
_CUDA_BLOCK = (16, 16)   # 256 threads per block (warps of 32 × 8 columns)


# ---------------------------------------------------------------------------
# CPU fallbacks
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _los_raycast_numba(
        Z_surface: np.ndarray,
        Z_obstacle: np.ndarray,
        is_elevated: np.ndarray,
        minx: float,
        maxy: float,
        gsd: float,
        cam_X: float,
        cam_Y: float,
        cam_Z: float,
        max_dist_m: float,
        step_m: float,
        height_tol_m: float
    ) -> np.ndarray:
        """
        Numba parallel JIT raycasting across the surface.
        For ground pixels, traces rays towards the camera.
        If the ray intersects an elevated obstacle higher than the ray, the pixel is occluded.
        """
        H, W = Z_surface.shape
        visible = np.ones((H, W), dtype=np.bool_)
        max_steps = int(max_dist_m / step_m)

        for i in prange(H):
            yp = maxy - (i + 0.5) * gsd
            for j in range(W):
                # Elevated roof pixels have clear line-of-sight to the overhead camera
                if is_elevated[i, j]:
                    continue

                zp = Z_surface[i, j]
                xp = minx + (j + 0.5) * gsd

                dx = cam_X - xp
                dy = cam_Y - yp
                dz = cam_Z - zp

                horiz_dist = np.sqrt(dx * dx + dy * dy)
                if horiz_dist < 1.0:
                    continue

                ux = dx / horiz_dist
                uy = dy / horiz_dist
                slope = dz / horiz_dist

                step_col = (ux * step_m) / gsd
                step_row = (-uy * step_m) / gsd
                step_z = slope * step_m

                cur_col = j + step_col
                cur_row = i + step_row
                cur_z = zp + step_z

                for step_idx in range(1, max_steps + 1):
                    c_idx = int(cur_col + 0.5)
                    r_idx = int(cur_row + 0.5)

                    if c_idx < 0 or c_idx >= W or r_idx < 0 or r_idx >= H:
                        break

                    # Check collision against physical DSM unconditionally
                    obs_h = Z_obstacle[r_idx, c_idx]
                    if obs_h > (cur_z + height_tol_m):
                        visible[i, j] = False
                        break

                    cur_col += step_col
                    cur_row += step_row
                    cur_z += step_z

        return visible


# ---------------------------------------------------------------------------
# GPU dispatch wrapper
# ---------------------------------------------------------------------------

def _run_cuda_los(
    Z_surface: np.ndarray,
    Z_obstacle: np.ndarray,
    is_elevated: np.ndarray,
    minx: float,
    maxy: float,
    gsd: float,
    cam_X: float,
    cam_Y: float,
    cam_Z: float,
    max_dist_m: float,
    step_m: float,
    height_tol_m: float
) -> np.ndarray:
    H, W = Z_surface.shape
    N = H * W
    
    # Generate ground coordinates
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    X_grid = minx + (cols + 0.5) * gsd
    Y_grid = maxy - (rows + 0.5) * gsd
    
    X_flat = np.ascontiguousarray(X_grid.ravel(), dtype=np.float64)
    Y_flat = np.ascontiguousarray(Y_grid.ravel(), dtype=np.float64)
    Z_flat = np.ascontiguousarray(Z_surface.ravel(), dtype=np.float32)
    Z_surf_pad = np.ascontiguousarray(Z_obstacle, dtype=np.float32) # using DSM for obstacle

    d_X = numba_cuda.to_device(X_flat)
    d_Y = numba_cuda.to_device(Y_flat)
    d_Z = numba_cuda.to_device(Z_flat)
    d_Z_surf_pad = numba_cuda.to_device(Z_surf_pad)
    d_out_vis = numba_cuda.device_array(N, dtype=np.bool_)

    blocks = (N + 256 - 1) // 256
    
    _compute_los_visibility_cuda[blocks, 256](
        d_X, d_Y, d_Z,
        d_Z_surf_pad, float(minx), float(maxy), float(gsd),
        float(cam_X), float(cam_Y), float(cam_Z),
        np.int32(W), np.int32(H), d_out_vis
    )

    numba_cuda.synchronize()
    visible = d_out_vis.copy_to_host().reshape((H, W))

    del d_X, d_Y, d_Z, d_Z_surf_pad, d_out_vis
    numba_cuda.current_context().memory_manager.deallocations.clear()

    return visible


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_visibility_mask(
    camera: Camera,
    Z_surface: np.ndarray,
    Z_obstacle: np.ndarray,
    is_elevated: np.ndarray,
    minx: float,
    maxy: float,
    gsd: float,
    in_bounds_mask: np.ndarray = None,
    max_dist_m: float = 100.0,
    step_m: float = 0.20,
    height_tol_m: float = 0.40,
    use_gpu: bool = True,
    use_numba: bool = True
) -> np.ndarray:
    """
    Computes a boolean visibility mask for a camera across the ground & roof surface.

    GPU dispatch order:
      CUDA (if available and use_gpu=True)  →  Numba CPU  →  Pure NumPy

    Args:
        camera:        Camera object (provides X0, Y0, Z0).
        Z_surface:     (H, W) float32 bare-earth / DTM elevation.
        Z_obstacle:    (H, W) float32 full DSM (roofs + terrain).
        is_elevated:   (H, W) bool mask of building roof pixels.
        minx:          West edge of this chunk in EPSG:28992 (metres).
        maxy:          North edge of this chunk in EPSG:28992 (metres).
        gsd:           Ground sample distance (metres/pixel).
        in_bounds_mask: Optional pre-computed in-bounds mask (ANDed with result).
        max_dist_m:    Maximum raycast distance in metres (default 100 m).
        step_m:        Ray march step in metres (default 0.20 m).
        height_tol_m:  Height clearance tolerance (default 0.40 m).
        use_gpu:       Set False to force CPU execution (testing / fallback).
        use_numba:     Set False to force pure-NumPy fallback.

    Returns:
        visible: (H, W) bool ndarray; True = unoccluded, False = blocked.
    """
    Z_surface  = Z_surface.astype(np.float32)
    Z_obstacle = Z_obstacle.astype(np.float32)
    is_elevated = is_elevated.astype(np.bool_)

    if use_gpu and HAS_CUDA:
        print("[Visibility] GPU (CUDA) LOS raycasting")
        vis = _run_cuda_los(
            Z_surface, Z_obstacle, is_elevated,
            minx, maxy, gsd,
            float(camera.X0), float(camera.Y0), float(camera.Z0),
            max_dist_m, step_m, height_tol_m
        )
    elif HAS_NUMBA and use_numba:
        print("[Visibility] CPU (Numba parallel) LOS raycasting")
        vis = _los_raycast_numba(
            Z_surface, Z_obstacle, is_elevated,
            float(minx), float(maxy), float(gsd),
            float(camera.X0), float(camera.Y0), float(camera.Z0),
            float(max_dist_m), float(step_m), float(height_tol_m)
        )
    else:
        print("[Visibility] CPU (NumPy) LOS raycasting")
        H, W = Z_surface.shape
        vis = np.ones((H, W), dtype=bool)
        cols, rows = np.meshgrid(np.arange(W), np.arange(H))
        gx = minx + (cols + 0.5) * gsd
        gy = maxy - (rows + 0.5) * gsd

        dx = camera.X0 - gx
        dy = camera.Y0 - gy
        dz = camera.Z0 - Z_surface
        dist = np.maximum(np.hypot(dx, dy), 1.0)
        ux = dx / dist
        uy = dy / dist
        slope = dz / dist

        num_steps = int(max_dist_m / step_m)
        for s in range(1, num_steps + 1):
            r_step = s * step_m
            tx = gx + r_step * ux
            ty = gy + r_step * uy
            tz = Z_surface + r_step * slope

            tc = np.round((tx - minx) / gsd - 0.5).astype(np.int32)
            tr = np.round((maxy - ty) / gsd - 0.5).astype(np.int32)

            valid = (tc >= 0) & (tc < W) & (tr >= 0) & (tr < H)
            safe_c = np.clip(tc, 0, W - 1)
            safe_r = np.clip(tr, 0, H - 1)

            hit = valid & (Z_obstacle[safe_r, safe_c] > (tz + height_tol_m))
            vis[~is_elevated & hit] = False

    if in_bounds_mask is not None:
        vis &= in_bounds_mask

    return vis
