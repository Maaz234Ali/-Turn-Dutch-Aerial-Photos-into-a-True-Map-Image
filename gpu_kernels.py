"""
gpu_kernels.py
Raw CUDA kernels for the True Orthophoto pipeline.

Two primary compute workloads are hosted here:

1. _los_raycast_cuda  – per-pixel Line-of-Sight (LOS) raycasting kernel.
   Replaces the Numba @njit loop in visibility.py.
   Each CUDA thread handles one output pixel; no atomics are required.

2. _project_collinearity_cuda – per-pixel collinearity back-projection kernel.
   Replaces the NumPy matrix multiply in projection.py.
   Accepts the flattened rotation matrix as a constant array for fast access.

Both kernels are designed for chunked / tiled execution so they never exceed
the 8 GB VRAM budget.  The caller is responsible for allocating device arrays
and transferring data; see visibility.py and projection.py for the host-side
wrappers.
"""

import math

try:
    from numba import cuda
    import numpy as np
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False


# ---------------------------------------------------------------------------
# 1. LINE-OF-SIGHT RAYCASTING KERNEL
# ---------------------------------------------------------------------------

if HAS_CUDA:
    @cuda.jit
    def _compute_los_visibility_cuda(
        X_flat, Y_flat, Z_flat, 
        Z_surf_pad, pad_minx, pad_maxy, gsd, 
        cam_X0, cam_Y0, cam_Z0, 
        pad_cols, pad_rows, out_vis
    ):
        idx = cuda.grid(1)
        if idx >= X_flat.shape[0]: return
            
        target_X = X_flat[idx]
        target_Y = Y_flat[idx]
        target_Z = Z_flat[idx]
        
        # Vector from ground target to camera
        dX = cam_X0 - target_X
        dY = cam_Y0 - target_Y
        dZ = cam_Z0 - target_Z
        
        dist_2d = math.sqrt(dX**2 + dY**2)
        steps = int(math.ceil(dist_2d / (gsd * 0.5))) # Half-pixel step increments
        
        if steps < 2:
            out_vis[idx] = True
            return
            
        step_X = dX / steps
        step_Y = dY / steps
        step_Z = dZ / steps
        
        is_visible = True
        
        # We only need to skip a tiny distance (e.g., 2 steps = 0.25m) to avoid self-collision.
        # Skipping 1.5m was too aggressive and caused rays to jump completely over walls!
        start_step = 2 
        
        for i in range(start_step, steps):
            curr_X = target_X + i * step_X
            curr_Y = target_Y + i * step_Y
            curr_Z = target_Z + i * step_Z
            
            # Convert ray to padded grid coordinates
            grid_c = int((curr_X - pad_minx) / gsd)
            grid_r = int((pad_maxy - curr_Y) / gsd)
            
            if 0 <= grid_c < pad_cols and 0 <= grid_r < pad_rows:
                surface_elev = Z_surf_pad[grid_r, grid_c]
                # Increase tolerance to 30cm to handle slight DSM slopes
                if surface_elev > (curr_Z + 0.30): 
                    is_visible = False
                    break
                    
        out_vis[idx] = is_visible

    @cuda.jit
    def _los_raycast_cuda(
        Z_surface,        # float32 (H, W) – bare-earth / DTM elevation per pixel
        Z_obstacle,       # float32 (H, W) – DSM (full surface including roofs)
        is_elevated,      # bool    (H, W) – True where pixel is a roof/building
        visible,          # bool    (H, W) – OUTPUT: occlusion result
        minx,             # float64 scalar – west edge of chunk (RD m)
        maxy,             # float64 scalar – north edge of chunk (RD m)
        gsd,              # float64 scalar – ground sample distance (m/pixel)
        cam_X,            # float64 scalar – camera easting
        cam_Y,            # float64 scalar – camera northing
        cam_Z,            # float64 scalar – camera elevation (m NAP)
        max_steps,        # int32   scalar – maximum ray steps
        step_m,           # float64 scalar – ray step length in metres
        height_tol_m      # float64 scalar – height tolerance (m)
    ):
        """
        Each thread processes one output pixel (row i, col j).

        Ray geometry:
          - Origin:    ground point (xp, yp, zp)
          - Direction: towards camera (dx, dy, dz) normalised on the horizontal plane
          - Steps:     march step_m metres per iteration toward camera
          - Occlusion: if any intermediate DSM cell is ABOVE the ray height, pixel is hidden

        Roof pixels (is_elevated == True) always have clear LOS and are skipped.
        """
        j, i = cuda.grid(2)   # (col, row) – match array indexing
        H, W = Z_surface.shape

        if i >= H or j >= W:
            return

        # Roof pixels are always visible from overhead camera
        if is_elevated[i, j]:
            visible[i, j] = True
            return

        # Pixel centre in RD metres
        xp = minx + (j + 0.5) * gsd
        yp = maxy - (i + 0.5) * gsd
        zp = Z_surface[i, j]

        dx = cam_X - xp
        dy = cam_Y - yp
        dz = cam_Z - zp

        horiz_dist = math.sqrt(dx * dx + dy * dy)
        if horiz_dist < 1.0:
            visible[i, j] = True
            return

        # Normalise horizontal direction and compute slope
        inv_h = 1.0 / horiz_dist
        ux = dx * inv_h
        uy = dy * inv_h
        slope = dz * inv_h

        # Pre-compute pixel-space increments for the ray march
        step_col = (ux * step_m) / gsd
        step_row = (-uy * step_m) / gsd
        step_z   = slope * step_m

        cur_col = j + step_col
        cur_row = i + step_row
        cur_z   = zp + step_z

        occluded = False
        for _ in range(max_steps):
            c_idx = int(cur_col + 0.5)
            r_idx = int(cur_row + 0.5)

            # Ray has left the chunk – no more occlusion possible
            if c_idx < 0 or c_idx >= W or r_idx < 0 or r_idx >= H:
                break

            # Check against full DSM (buildings block rays from below)
            if Z_obstacle[r_idx, c_idx] > (cur_z + height_tol_m):
                occluded = True
                break

            cur_col += step_col
            cur_row += step_row
            cur_z   += step_z

        visible[i, j] = not occluded


# ---------------------------------------------------------------------------
# 2. COLLINEARITY BACK-PROJECTION KERNEL
# ---------------------------------------------------------------------------

if HAS_CUDA:
    @cuda.jit
    def _project_collinearity_cuda(
        X_flat,       # float64 (N,) – easting of each pixel
        Y_flat,       # float64 (N,) – northing of each pixel
        Z_flat,       # float32 (N,) – elevation of each pixel
        R_flat,       # float64 (9,) – row-major flattened 3x3 rotation matrix
        cam_X0,       # float64 – camera easting
        cam_Y0,       # float64 – camera northing
        cam_Z0,       # float64 – camera elevation
        f_m,          # float64 – focal length in metres
        pixel_size_m, # float64 – detector pixel pitch in metres
        x_ppa_m,      # float64 – principal point x offset (metres)
        y_ppa_m,      # float64 – principal point y offset (metres)
        c_mid,        # float64 – column centre (pixels)
        r_mid,        # float64 – row centre (pixels)
        cam_W,        # int32   – sensor width (pixels)
        cam_H,        # int32   – sensor height (pixels)
        margin,       # float64 – in-bounds margin (pixels)
        out_cols,     # float32 (N,) OUTPUT sub-pixel column indices
        out_rows,     # float32 (N,) OUTPUT sub-pixel row indices
        out_angles,   # float32 (N,) OUTPUT off-nadir view angle (degrees)
        out_valid,    # bool    (N,) OUTPUT in-bounds mask
    ):
        """
        Vectorised collinearity equations executed one thread per ground pixel.

        Rotation matrix convention:  cam_coords = R @ [dX, dY, dZ]^T
        where R_flat[0..8] = [R00, R01, R02, R10, R11, R12, R20, R21, R22].
        """
        idx = cuda.grid(1)
        if idx >= X_flat.shape[0]:
            return

        dX = X_flat[idx] - cam_X0
        dY = Y_flat[idx] - cam_Y0
        dZ = Z_flat[idx] - cam_Z0

        # Rotate into camera frame
        Xc = R_flat[0] * dX + R_flat[1] * dY + R_flat[2] * dZ
        Yc = R_flat[3] * dX + R_flat[4] * dY + R_flat[5] * dZ
        Zc = R_flat[6] * dX + R_flat[7] * dY + R_flat[8] * dZ

        # Off-nadir angle
        horiz = math.sqrt(dX * dX + dY * dY)
        neg_dZ = -dZ
        vert  = neg_dZ if neg_dZ > 1.0 else 1.0   # max() not valid in cuda.jit
        # math.degrees() not available in device code; multiply by 180/pi manually
        out_angles[idx] = math.atan2(horiz, vert) * 57.29577951308232

        # Points behind the camera are invalid
        if Zc >= -1.0:
            out_cols[idx]  = 0.0
            out_rows[idx]  = 0.0
            out_valid[idx] = False
            return

        inv_Zc = -f_m / Zc   # equals f / |Zc|

        # Perspective projection (metres on focal plane)
        x_m = Xc * inv_Zc
        y_m = Yc * inv_Zc

        # Map focal-plane metres to sensor pixel coordinates
        u = y_m - y_ppa_m
        v = x_m - x_ppa_m

        col = c_mid + u / pixel_size_m
        row = r_mid + v / pixel_size_m

        out_cols[idx]  = col
        out_rows[idx]  = row

        in_bounds = (
            col >= margin and
            col <  cam_W - margin and
            row >= margin and
            row <  cam_H - margin
        )
        out_valid[idx] = in_bounds
