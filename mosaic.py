"""
mosaic.py
Multi-view True Orthophoto synthesis, building-aware seamline routing,
and occlusion patching.

GPU / VRAM-Safe Tiled Architecture
====================================
The output raster is divided into spatial tiles (default 2048 x 2048 output
pixels, ~512 m x 512 m at 0.25 m GSD).  For each tile the pipeline:

  1. Slices only the DSM/DTM/mask sub-arrays for the tile **plus a buffer
     margin** of `dsm_buffer_m` metres (default 100 m).  The buffer ensures
     the LOS raycaster can see tall buildings just outside the tile edge.

  2. Determines which cameras intersect the padded tile bounding box and
     projects only those cameras.

  3. Runs the CUDA LOS kernel on the padded DSM chunk (fits in VRAM because
     a 2048+2*400 px chunk at float32 is ~50 MB, well within 8 GB).

  4. Crops the padded visibility mask back to the bare tile extent.

  5. Samples aerial image pixels for each camera on the tile.

  6. Writes the finished tile into the output GeoTIFF via rasterio windowed
     writes (no full-raster array ever needs to sit in RAM simultaneously).

  7. Explicitly frees all intermediate arrays to keep host RAM stable.

BAG footprint + nDSM building detection is still performed on the full raster
(it is read-only metadata, not the heavy compute).
"""

import os
import gc
import json
import math
import urllib.request
import numpy as np
import cv2
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.transform import from_origin
from scipy.ndimage import label
from shapely.geometry import shape
from shapely.affinity import translate
try:
    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False

from camera_parser import Camera
from projection import project_ground_to_camera, sample_image_bilinear
from visibility import compute_visibility_mask
from radiometry import match_radiometry

# ---------------------------------------------------------------------------
# Per-building lean correction calibration constants
#
# NOTE: Set to ZERO because cameras fly at ~1950m altitude (EPSG:28992).
# At this height the off-nadir angle is only ~1-2°, so tan(θ) ≈ 0.02.
# For a 10m building that yields < 0.2m lean displacement — sub-pixel at
# 0.25m GSD.  Applying the old 45° calibration constants (derived from a
# low-altitude dataset) massively overcorrects and shifts BAG footprints
# in the wrong direction.
#
# To re-enable lean correction: measure dX/dY from known GCP pairs in
# YOUR imagery and set the two coefficients accordingly.
# ---------------------------------------------------------------------------
_LEAN_TAN_THETA    = 0.0   # disabled — negligible at high-altitude (1950 m)
_LEAN_AVG_DX_PER_M = 0.0   # dX per metre of building height
_LEAN_AVG_DY_PER_M = 0.0   # dY per metre of building height

# ---------------------------------------------------------------------------
# VRAM budget constants  (tune per GPU)
# ---------------------------------------------------------------------------
TILE_PIXELS     = 1000      # output tile edge length (pixels)
DSM_BUFFER_M    = 60.0      # extra DSM margin matching max_dist_m in visibility.py

# ---------------------------------------------------------------------------
# BAG footprint fetching
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Local GPKG path: official BAG footprints downloaded via download_bag.py
# ---------------------------------------------------------------------------
_LOCAL_GPKGS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bag_pand.gpkg"),
]


def fetch_bag_footprints(minx: float, miny: float, maxx: float, maxy: float) -> list[dict]:
    """Fetches BAG building footprint polygons from local GPKG (data/bag_pand.gpkg). 100% offline."""
    if not _HAS_GPD:
        raise ImportError("geopandas is required to load BAG footprints.")

    for gpkg_path in _LOCAL_GPKGS:
        if os.path.isfile(gpkg_path):
            gdf = gpd.read_file(gpkg_path, bbox=(minx, miny, maxx, maxy))
            features = [
                {
                    'id': str(row.get('identificatie', row.get('id', idx))),
                    'geometry': geom.__geo_interface__
                }
                for idx, (_, row) in enumerate(gdf.iterrows())
                for geom in [row.geometry]
                if geom is not None and not geom.is_empty
            ]
            print(f"[Mosaic] Loaded {len(features)} BAG footprints from {gpkg_path}.")
            return features

    raise FileNotFoundError(
        f"[Mosaic] Local BAG GeoPackage not found at {_LOCAL_GPKGS}. "
        "Run download_bag.py to fetch footprints offline."
    )


# ---------------------------------------------------------------------------
# Global building / surface model (once per full raster)
# ---------------------------------------------------------------------------

def _sample_median_in_polygon(
    poly_geom,
    raster: np.ndarray,
    trans: rasterio.Affine,
    H: int,
    W: int
) -> float:
    """
    Rasterises a single polygon and returns the median raster value inside it.
    Returns NaN if the polygon has no valid pixels in the raster.
    """
    mask = rasterize(
        [(poly_geom, 1)],
        out_shape=(H, W),
        transform=trans,
        fill=0,
        dtype=np.uint8
    )
    vals = raster[mask == 1]
    return float(np.median(vals)) if vals.size > 0 else float('nan')


def _apply_per_building_lean_correction(
    bag_features: list[dict],
    Z_dsm: np.ndarray,
    Z_dtm: np.ndarray,
    trans: rasterio.Affine,
    H: int,
    W: int
) -> list:
    """
    For every BAG feature polygon, derives the building's nDSM height from
    the DSM/DTM rasters and shifts the polygon geometry by the height-dependent
    lean correction vector.

    Correction formula (calibrated from measured GCP pairs):
        dX = height * _LEAN_AVG_DX_PER_M
        dY = height * _LEAN_AVG_DY_PER_M

    These coefficients encode tan(theta)=1.0 (45° camera angle) AND the
    azimuth direction (~221° SW) simultaneously, derived from:
        Point 1:  measured dX=-2.268 m, dY=-2.816 m,  height=3.616 m
        Point 2:  measured dX=-2.269 m, dY=-2.347 m,  height=3.264 m

    Returns:
        List of (shifted_shapely_geometry, 1) tuples ready for rasterize().
    """
    shifted_shapes = []
    n_corrected   = 0
    n_fallback    = 0
    height_stats  = []

    for feat in bag_features:
        geom = shape(feat['geometry'])

        # Sample median DSM and DTM inside this footprint
        h_dsm = _sample_median_in_polygon(geom, Z_dsm, trans, H, W)
        h_dtm = _sample_median_in_polygon(geom, Z_dtm, trans, H, W)

        if math.isnan(h_dsm) or math.isnan(h_dtm):
            # Polygon is entirely outside the raster extent – no shift
            shifted_shapes.append((geom, 1))
            n_fallback += 1
            continue

        building_height = max(0.0, h_dsm - h_dtm)   # nDSM height (metres)
        height_stats.append(building_height)

        if building_height < 0.5:
            # Essentially flat ground – no correction needed
            shifted_shapes.append((geom, 1))
            n_fallback += 1
            continue

        # Per-building shift derived from calibrated lean coefficients
        dx = building_height * _LEAN_AVG_DX_PER_M
        dy = building_height * _LEAN_AVG_DY_PER_M

        shifted_geom = translate(geom, xoff=dx, yoff=dy)
        shifted_shapes.append((shifted_geom, 1))
        n_corrected += 1

    if height_stats:
        print(
            f"[Mosaic] Per-building lean correction applied to {n_corrected}/{len(bag_features)} polygons "
            f"(height range: {min(height_stats):.1f}–{max(height_stats):.1f} m, "
            f"median: {float(np.median(height_stats)):.1f} m). "
            f"Flat/out-of-bounds: {n_fallback}."
        )
    return shifted_shapes


def build_building_and_ground_models(
    Z_dsm: np.ndarray,
    minx: float,
    maxy: float,
    gsd: float,
    buffer_halo_m: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs a footprint-regularised surface model and bare-earth DTM:
      1. Inside each footprint: flat roof height (90th percentile of DSM inside).
      2. Outside each footprint: local ground height (DTM), with ramps and halos
         within 2m of walls removed, making walls vertical exactly on the footprint line.
      3. Returns (Z_surface_regularized, Z_dtm, building_mask_exact).
    """
    H, W = Z_dsm.shape
    maxx = minx + W * gsd
    miny = maxy - H * gsd

    # 1. Bare-earth DTM via morphological opening (80m filter for terrain)
    k_sz = max(5, int(80.0 / gsd))
    if k_sz % 2 == 0:
        k_sz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sz, k_sz))
    Z_dtm = cv2.morphologyEx(Z_dsm.astype(np.float32), cv2.MORPH_OPEN, kernel)

    # 2. Query official BAG building footprints
    trans = rasterio.transform.from_origin(minx, maxy, gsd, gsd)
    bag_features = fetch_bag_footprints(minx, miny, maxx, maxy)

    Z_surface = Z_dtm.copy()
    building_mask = np.zeros((H, W), dtype=bool)

    if bag_features:
        print(f"[Mosaic] Regularising {len(bag_features)} building footprints to vertical walls...")
        for feat in bag_features:
            geom = shape(feat['geometry'])
            if geom is None or geom.is_empty:
                continue

            # Rasterize exact footprint geometry (undilated)
            poly_mask = rasterize(
                [(geom, 1)],
                out_shape=(H, W),
                transform=trans,
                fill=0,
                default_value=1,
                dtype=np.uint8
            ) == 1

            if not np.any(poly_mask):
                continue

            dsm_vals = Z_dsm[poly_mask]
            dtm_vals = Z_dtm[poly_mask]

            valid_dsm = dsm_vals[np.isfinite(dsm_vals) & (dsm_vals < 500.0) & (dsm_vals > -50.0)]
            if valid_dsm.size == 0:
                continue

            # Flat roof deck height: 90th percentile inside footprint
            h_roof = float(np.percentile(valid_dsm, 90))
            valid_dtm = dtm_vals[np.isfinite(dtm_vals)]
            h_dtm_med = float(np.median(valid_dtm)) if valid_dtm.size > 0 else 0.0

            # Only elevate if building is higher than ground by at least 1.8m
            if (h_roof - h_dtm_med) >= 1.8:
                Z_surface[poly_mask] = h_roof
                building_mask[poly_mask] = True
            else:
                Z_surface[poly_mask] = np.maximum(dsm_vals, dtm_vals)

        # 3. Clear any ramp or halo within ~2m outside walls
        halo_px = int(np.ceil(buffer_halo_m / gsd))
        k_halo = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * halo_px + 1, 2 * halo_px + 1))
        dilated_bldg = cv2.dilate(building_mask.astype(np.uint8), k_halo) == 1
        halo_zone = dilated_bldg & (~building_mask)
        Z_surface[halo_zone] = Z_dtm[halo_zone]

        print(f"[Mosaic] Footprint-regularised DSM: {np.sum(building_mask)} roof pixels set to flat roofs, "
              f"{np.sum(halo_zone)} halo/ramp pixels within {buffer_halo_m}m cleared to ground DTM.")
    else:
        print("[Mosaic] Warning: No BAG footprints found, using nDSM fallback.")
        ndsm = Z_dsm - Z_dtm
        building_mask = (ndsm > 2.8)
        Z_surface = np.where(building_mask, Z_dsm, Z_dtm)

    return Z_surface, Z_dtm, building_mask


# ---------------------------------------------------------------------------
# Per-tile radiometry: collect overlap statistics for the full image pair
# ---------------------------------------------------------------------------

def _global_radiometry_reference(cam_data_list: list[dict]) -> int:
    """Return the index of the camera with the lowest mean view angle (closest to nadir)."""
    mean_angles = [
        np.mean(c['view_angles'][c['vis_mask']]) if np.any(c['vis_mask']) else 999.0
        for c in cam_data_list
    ]
    return int(np.argmin(mean_angles))


# ---------------------------------------------------------------------------
# Helper: extract a padded DSM sub-chunk with buffer margin
# ---------------------------------------------------------------------------

def _extract_padded_chunk(
    arr: np.ndarray,
    row0: int, row1: int,
    col0: int, col1: int,
    buf_px: int,
    full_H: int,
    full_W: int
) -> tuple[np.ndarray, int, int, int, int]:
    """
    Extracts a padded sub-array from `arr` (full-raster grid).

    Args:
        arr:    Full 2-D raster array (H_full, W_full).
        row0/row1, col0/col1: Tile pixel range (exclusive end).
        buf_px: Buffer in pixels to add around tile.
        full_H, full_W: Dimensions of the full raster.

    Returns:
        chunk, padded_row0, padded_col0, padded_row1, padded_col1
        where padded_* are the actual indices into arr used for slicing.
    """
    pr0 = max(0, row0 - buf_px)
    pr1 = min(full_H, row1 + buf_px)
    pc0 = max(0, col0 - buf_px)
    pc1 = min(full_W, col1 + buf_px)
    return arr[pr0:pr1, pc0:pc1], pr0, pc0, pr1, pc1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_true_orthophoto(
    cameras: list[Camera],
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    Z_dsm: np.ndarray,
    transform: rasterio.Affine,
    output_path: str,
    gsd: float,
    crs: str = "EPSG:28992",
    traditional: bool = False,
    tile_pixels: int = TILE_PIXELS,
    dsm_buffer_m: float = DSM_BUFFER_M
) -> dict:
    """
    Executes the multi-view True Orthophoto synthesis in VRAM-safe spatial tiles.

    Pipeline per tile
    -----------------
    1. Extract padded DSM/DTM/mask sub-arrays (tile + buffer_margin).
    2. Run CUDA LOS raycasting on the padded chunk per camera.
    3. Crop the padded result back to the bare tile extent.
    4. Sample aerial image pixels for the tile.
    5. Blend and write the tile into the output GeoTIFF using windowed write.
    6. Free all intermediate GPU and CPU buffers.

    Args:
        cameras:       List of Camera objects covering the AOI.
        X_grid:        1-D Easting coordinate array (full raster).
        Y_grid:        1-D Northing coordinate array (full raster, top→bottom).
        Z_dsm:         (H, W) float32 DSM raster.
        transform:     Rasterio affine transform for the full raster.
        output_path:   Path to write the output GeoTIFF.
        gsd:           Ground sample distance in metres.
        crs:           Coordinate reference system string.
        traditional:   If True, skip occlusion detection.
        tile_pixels:   Tile edge size in pixels (default 2048).
        dsm_buffer_m:  Buffer margin for LOS raycasting in metres (default 100 m).

    Returns:
        dict with 'grid_shape', 'num_cams_used', 'num_tiles'.
    """
    H, W = Z_dsm.shape
    num_cams = len(cameras)
    minx = X_grid[0]  - gsd / 2.0
    maxy = Y_grid[0]  + gsd / 2.0

    # ------------------------------------------------------------------
    # Step 1: Full-raster surface models (cheap metadata pass)
    # ------------------------------------------------------------------
    print(f"[Mosaic] Building surface models for {W}x{H} grid at {gsd:.2f}m GSD...")
    Z_surface, Z_dtm, building_mask = build_building_and_ground_models(Z_dsm, minx, maxy, gsd)

    # Buffer in pixels for DSM margin around each tile
    buf_px = int(np.ceil(dsm_buffer_m / gsd))

    # ------------------------------------------------------------------
    # Step 2: Initialise output GeoTIFF (streaming write to temp file)
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Export the regularised DSM as aligned_dsm.tif
    aligned_dsm_path = os.path.join(out_dir, "aligned_dsm.tif")
    print(f"[Mosaic] Exporting regularised DSM to: {aligned_dsm_path}")
    with rasterio.open(
        aligned_dsm_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=1, dtype="float32",
        crs=crs, transform=transform,
        compress="deflate", nodata=-9999.0
    ) as dsm_dst:
        dsm_dst.write(Z_surface, 1)

    temp_output_path = output_path + ".tmp.tif"
    print(f"[Mosaic] Opening temporary GeoTIFF for streamed tile writes: {temp_output_path}")
    dst_file = rasterio.open(
        temp_output_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=3, dtype="uint8",
        crs=crs, transform=transform,
        compress="deflate",
        photometric="RGB",
        tiled=True,
        blockxsize=256, blockysize=256,
    )

    # ------------------------------------------------------------------
    # Step 3: Tile iteration
    # ------------------------------------------------------------------
    tile_rows = list(range(0, H, tile_pixels))
    tile_cols = list(range(0, W, tile_pixels))
    num_tiles = len(tile_rows) * len(tile_cols)
    tile_idx  = 0

    # Aggregate stats returned to caller
    all_cam_data_summary = []

    for tr0 in tile_rows:
        tr1 = min(tr0 + tile_pixels, H)
        tile_H = tr1 - tr0

        for tc0 in tile_cols:
            tc1 = min(tc0 + tile_pixels, W)
            tile_W = tc1 - tc0
            tile_idx += 1

            # Tile bounding box in RD metres
            t_minx = minx + tc0 * gsd
            t_maxx = minx + tc1 * gsd
            t_miny = maxy - tr1 * gsd
            t_maxy = maxy - tr0 * gsd

            print(f"\n[Mosaic] === Tile {tile_idx}/{num_tiles}: "
                  f"pixels ({tc0},{tr0})->({tc1},{tr1}), "
                  f"RD X=[{t_minx:.0f},{t_maxx:.0f}] Y=[{t_miny:.0f},{t_maxy:.0f}] ===")

            # ---- 3a. Extract padded DSM arrays for this tile ----
            # (padded so the LOS ray can see buildings just outside tile edge)
            Z_dsm_pad,    pr0, pc0, pr1, pc1 = _extract_padded_chunk(
                Z_dsm, tr0, tr1, tc0, tc1, buf_px, H, W)
            Z_dtm_pad,    *_ = _extract_padded_chunk(
                Z_dtm, tr0, tr1, tc0, tc1, buf_px, H, W)
            Z_surf_pad,   *_ = _extract_padded_chunk(
                Z_surface, tr0, tr1, tc0, tc1, buf_px, H, W)
            bldg_pad,     *_ = _extract_padded_chunk(
                building_mask.astype(np.bool_), tr0, tr1, tc0, tc1, buf_px, H, W)

            pad_H, pad_W = Z_dsm_pad.shape

            # Padded tile origin in RD metres
            pad_minx = minx + pc0 * gsd
            pad_maxy = maxy - pr0 * gsd

            # Coordinate grids for the padded tile
            pad_X = X_grid[pc0:pc1]
            pad_Y = Y_grid[pr0:pr1]
            pad_gx, pad_gy = np.meshgrid(pad_X, pad_Y)

            # Pixel offsets of the bare tile within the padded tile
            inner_r0 = tr0 - pr0
            inner_c0 = tc0 - pc0
            inner_r1 = inner_r0 + tile_H
            inner_c1 = inner_c0 + tile_W

            # ---- 3b. Filter cameras that overlap the padded tile ----
            tile_cameras = [
                cam for cam in cameras
                if cam.intersects_bbox(
                    pad_minx, pad_maxy - pad_H * gsd,
                    pad_minx + pad_W * gsd, pad_maxy
                )
            ]
            if not tile_cameras:
                print(f"  [Tile {tile_idx}] No cameras overlap – writing black.")
                dst_file.write(
                    np.zeros((3, tile_H, tile_W), dtype=np.uint8),
                    window=Window(tc0, tr0, tile_W, tile_H)
                )
                continue

            # ---- 3c. Per-camera projection + LOS on padded chunk ----
            tile_cam_data = []
            for cam_i, cam in enumerate(tile_cameras):
                # Project ground pixels onto this camera (GPU or CPU)
                cols_pad, rows_pad, angles_pad, inbounds_pad = project_ground_to_camera(
                    cam, pad_gx, pad_gy, Z_surf_pad
                )

                if not np.any(inbounds_pad):
                    continue

                if traditional:
                    vis_pad = inbounds_pad.copy()
                else:
                    vis_pad = compute_visibility_mask(
                        camera=cam,
                        Z_surface=Z_surf_pad,
                        Z_obstacle=Z_surf_pad,
                        is_elevated=bldg_pad,
                        minx=float(pad_minx),
                        maxy=float(pad_maxy),
                        gsd=float(gsd),
                        in_bounds_mask=inbounds_pad,
                        max_dist_m=60.0,
                        step_m=0.25,
                        height_tol_m=0.40,
                        use_gpu=False,
                        use_numba=True
                    )

                # Sample aerial image pixels for the padded tile
                sampled_rgb_pad = sample_image_bilinear(cam, cols_pad, rows_pad, mask=inbounds_pad)

                # ---- Crop padded results back to bare tile ----
                cols_tile    = cols_pad   [inner_r0:inner_r1, inner_c0:inner_c1]
                rows_tile    = rows_pad   [inner_r0:inner_r1, inner_c0:inner_c1]
                angles_tile  = angles_pad [inner_r0:inner_r1, inner_c0:inner_c1]
                inbounds_tile = inbounds_pad[inner_r0:inner_r1, inner_c0:inner_c1]
                vis_tile     = vis_pad    [inner_r0:inner_r1, inner_c0:inner_c1]
                rgb_tile     = sampled_rgb_pad[inner_r0:inner_r1, inner_c0:inner_c1]

                occluded_px = int(np.sum(inbounds_tile & ~vis_tile))
                print(
                    f"  [{cam_i+1}/{len(tile_cameras)}] {cam.image_name}: "
                    f"coverage {np.sum(inbounds_tile)/(tile_H*tile_W)*100:.1f}%, "
                    f"occluded ground {occluded_px} px"
                )

                tile_cam_data.append({
                    'cam':        cam,
                    'cols':       cols_tile,
                    'rows':       rows_tile,
                    'view_angles': angles_tile,
                    'in_bounds':  inbounds_tile,
                    'vis_mask':   vis_tile,
                    'rgb':        rgb_tile
                })

                # Explicitly free padded intermediate arrays
                del cols_pad, rows_pad, angles_pad, inbounds_pad
                del vis_pad, sampled_rgb_pad
                gc.collect()

            if not tile_cam_data:
                print(f"  [Tile {tile_idx}] No valid camera projections – writing black.")
                dst_file.write(
                    np.zeros((3, tile_H, tile_W), dtype=np.uint8),
                    window=Window(tc0, tr0, tile_W, tile_H)
                )
                continue

            # ---- 3d. Radiometric matching for this tile ----
            mean_tile_angles = [
                np.mean(c['view_angles'][c['vis_mask']]) if np.any(c['vis_mask']) else 999.0
                for c in tile_cam_data
            ]
            ref_t_idx = int(np.argmin(mean_tile_angles))
            ref_rgb_t  = tile_cam_data[ref_t_idx]['rgb']
            ref_vis_t  = tile_cam_data[ref_t_idx]['vis_mask']

            bldg_tile  = building_mask[tr0:tr1, tc0:tc1]
            ref_ground = ref_vis_t & ~bldg_tile

            for idx, c in enumerate(tile_cam_data):
                if idx == ref_t_idx:
                    continue
                overlap = c['vis_mask'] & ref_ground
                c['rgb'] = match_radiometry(c['rgb'], ref_rgb_t, overlap)

            # ---- 3e. Multi-View Synthesis for this tile ----
            best_cam_idx   = np.full((tile_H, tile_W), -1, dtype=np.int32)
            min_angle_grid = np.full((tile_H, tile_W), 9999.0, dtype=np.float32)

            # Rule C: Building-aware seamline avoidance
            # For each distinct building polygon, select the single best nadir camera covering it
            num_bldg_labels, bldg_labels = cv2.connectedComponents(bldg_tile.astype(np.uint8))
            for b_id in range(1, num_bldg_labels):
                b_mask = (bldg_labels == b_id)
                best_c = -1
                best_ang = 9999.0
                for c_idx, c in enumerate(tile_cam_data):
                    if np.all(c['in_bounds'][b_mask]):
                        mean_ang = float(np.mean(c['view_angles'][b_mask]))
                        if mean_ang < best_ang:
                            best_ang = mean_ang
                            best_c = c_idx
                if best_c == -1:
                    for c_idx, c in enumerate(tile_cam_data):
                        cov = float(np.mean(c['in_bounds'][b_mask]))
                        if cov >= 0.5:
                            mean_ang = float(np.mean(c['view_angles'][b_mask]))
                            if mean_ang < best_ang:
                                best_ang = mean_ang
                                best_c = c_idx
                if best_c != -1:
                    best_cam_idx[b_mask] = best_c
                else:
                    for c_idx, c in enumerate(tile_cam_data):
                        roof_valid = b_mask & c['in_bounds'] & (c['view_angles'] < min_angle_grid)
                        best_cam_idx[roof_valid] = c_idx
                        min_angle_grid[roof_valid] = c['view_angles'][roof_valid]

            # Ground: Must have clear LOS (c['vis_mask'] == True). Never paint walls over ground!
            min_angle_ground = np.full((tile_H, tile_W), 9999.0, dtype=np.float32)
            for c_idx, c in enumerate(tile_cam_data):
                ground_valid = (~bldg_tile) & c['vis_mask'] & (c['view_angles'] < min_angle_ground)
                best_cam_idx[ground_valid] = c_idx
                min_angle_ground[ground_valid] = c['view_angles'][ground_valid]

            # Fallback for hidden spots (pixels occluded in all cameras)
            unassigned = (best_cam_idx == -1)
            if np.any(unassigned):
                fallback_angles = np.full((tile_H, tile_W), 9999.0, dtype=np.float32)
                for c_idx, c in enumerate(tile_cam_data):
                    cand = unassigned & c['in_bounds'] & (c['view_angles'] < fallback_angles)
                    best_cam_idx[cand] = c_idx
                    fallback_angles[cand] = c['view_angles'][cand]

            # Compose tile RGB
            tile_rgb = np.zeros((tile_H, tile_W, 3), dtype=np.uint8)
            for c_idx, c in enumerate(tile_cam_data):
                pix = (best_cam_idx == c_idx)
                if np.any(pix):
                    tile_rgb[pix] = c['rgb'][pix]

            # ---- 3f. Write tile to output GeoTIFF ----
            dst_file.write(
                np.moveaxis(tile_rgb, -1, 0),   # (3, tile_H, tile_W)
                window=Window(tc0, tr0, tile_W, tile_H)
            )

            # ---- 3g. Free all tile-level buffers ----
            del tile_cam_data, tile_rgb, best_cam_idx, min_angle_grid
            del Z_dsm_pad, Z_dtm_pad, Z_surf_pad, bldg_pad
            del pad_gx, pad_gy
            gc.collect()
            # Attempt GPU cache flush if CUDA is available
            try:
                from numba import cuda as _cuda
                if _cuda.is_available():
                    _cuda.current_context().memory_manager.deallocations.clear()
            except Exception:
                pass

    dst_file.close()

    # Safely rename temporary file to final output path
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_output_path, output_path)
    print(f"\n[Mosaic] Safely completed writing and renamed to: {output_path}")

    # Read EVERY block of the final file and report unreadable block count
    unreadable_blocks = 0
    total_blocks = 0
    with rasterio.open(output_path) as src:
        for ji, window in src.block_windows(1):
            total_blocks += 1
            try:
                _ = src.read(window=window)
            except Exception as e:
                unreadable_blocks += 1
                print(f"[Mosaic] ERROR: unreadable block {window}: {e}")

    print(f"[Mosaic] Block verification: {unreadable_blocks} unreadable blocks out of {total_blocks} total blocks.")
    if unreadable_blocks > 0:
        raise IOError(f"GeoTIFF verification failed: {unreadable_blocks} unreadable blocks.")

    print(f"[Mosaic] True Orthophoto successfully saved to: {output_path}")
    print(f"[Mosaic] Total tiles processed: {tile_idx}")

    return {
        'num_cams_used': num_cams,
        'grid_shape':    (H, W),
        'num_tiles':     tile_idx,
        'unreadable_blocks': unreadable_blocks,
        'total_blocks': total_blocks
    }
