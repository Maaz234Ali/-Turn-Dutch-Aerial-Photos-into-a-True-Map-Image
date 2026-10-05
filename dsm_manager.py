"""
dsm_manager.py
Manages loading, fetching, and resampling of the Digital Surface Model (DSM) in EPSG:28992.
Supports automated retrieval of AHN 0.5m DSM from official Dutch PDOK WCS.
"""

import os
import urllib.request
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import cv2
from scipy.ndimage import distance_transform_edt

PDOK_AHN_WCS_URL = "https://service.pdok.nl/rws/ahn/wcs/v1_0"

class DSMManager:
    """
    Manages DSM loading, interpolation, and spatial queries.
    """
    def __init__(self, dsm_path: str = None, cache_dir: str = "data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.dsm_path = dsm_path
        self._src = None
        self._arr = None
        self._transform = None
        self._nodata = None

    def fetch_ahn_dsm_wcs(self, minx: float, miny: float, maxx: float, maxy: float, output_path: str) -> str:
        """
        Downloads the 0.5m AHN DSM GeoTIFF from PDOK WCS for the given bounding box.
        Chunks requests if area is too large to prevent HTTP 400.
        """
        import tempfile
        from rasterio.merge import merge

        bx0 = np.floor(minx - 10.0)
        by0 = np.floor(miny - 10.0)
        bx1 = np.ceil(maxx + 10.0)
        by1 = np.ceil(maxy + 10.0)

        chunk_size = 1000.0
        x_starts = np.arange(bx0, bx1, chunk_size)
        y_starts = np.arange(by0, by1, chunk_size)

        temp_dir = tempfile.mkdtemp()
        chunk_files = []

        print(f"[DSM] Downloading AHN DSM from PDOK WCS: BBOX=({bx0}, {by0}, {bx1}, {by1})...")
        
        for xs in x_starts:
            for ys in y_starts:
                xe = min(xs + chunk_size, bx1)
                ye = min(ys + chunk_size, by1)
                url = (
                    f"{PDOK_AHN_WCS_URL}?"
                    f"service=WCS&version=2.0.1&request=GetCoverage&coverageId=dsm_05m&"
                    f"subset=x({xs},{xe})&subset=y({ys},{ye})&format=image/tiff"
                )
                chunk_path = os.path.join(temp_dir, f"chunk_{xs}_{ys}.tif")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = resp.read()
                        
                    # Write the raw TIFF
                    raw_path = chunk_path + ".raw.tif"
                    with open(raw_path, "wb") as f:
                        f.write(data)
                        
                    # Re-open and add georeferencing
                    import rasterio
                    from rasterio.transform import from_bounds
                    with rasterio.open(raw_path, "r") as src:
                        meta = src.meta.copy()
                        transform = from_bounds(xs, ys, xe, ye, src.width, src.height)
                        meta.update({
                            "crs": "EPSG:28992",
                            "transform": transform,
                            "nodata": 3.4028235e+38
                        })
                        data_arr = src.read(1)
                        
                    with rasterio.open(chunk_path, "w", **meta) as dst:
                        dst.write(data_arr, 1)
                        
                    os.remove(raw_path)
                    chunk_files.append(chunk_path)
                except Exception as e:
                    print(f"[DSM] Failed to download chunk {xs},{ys}: {e}")

        if not chunk_files:
            raise RuntimeError("Failed to download any DSM chunks")

        if len(chunk_files) == 1:
            import shutil
            shutil.move(chunk_files[0], output_path)
            print(f"[DSM] Saved AHN DSM to {output_path}")
        else:
            print(f"[DSM] Merging {len(chunk_files)} DSM chunks manually...")
            # Calculate total bounds
            min_x, min_y, max_x, max_y = float('inf'), float('inf'), -float('inf'), -float('inf')
            srcs = []
            for fp in chunk_files:
                src = rasterio.open(fp)
                srcs.append(src)
                min_x = min(min_x, src.bounds.left)
                min_y = min(min_y, src.bounds.bottom)
                max_x = max(max_x, src.bounds.right)
                max_y = max(max_y, src.bounds.top)
            
            # Create empty array
            gsd = 0.5
            width = int(round((max_x - min_x) / gsd))
            height = int(round((max_y - min_y) / gsd))
            mosaic_arr = np.full((1, height, width), 3.4028235e+38, dtype=np.float32)
            
            from rasterio.transform import from_bounds
            out_trans = from_bounds(min_x, min_y, max_x, max_y, width, height)
            
            for src in srcs:
                data = src.read()
                # Calculate pixel offsets
                col_off = int(round((src.bounds.left - min_x) / gsd))
                row_off = int(round((max_y - src.bounds.top) / gsd))
                h, w = data.shape[1], data.shape[2]
                mosaic_arr[:, row_off:row_off+h, col_off:col_off+w] = data
                src.close()

            out_meta = srcs[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": height,
                "width": width,
                "transform": out_trans,
                "nodata": 3.4028235e+38
            })

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(mosaic_arr)
            
            print(f"[DSM] Saved merged AHN DSM to {output_path}")
            
        for fp in chunk_files:
            if os.path.exists(fp):
                os.remove(fp)
        os.rmdir(temp_dir)

        return output_path

    def ensure_dsm(self, minx: float, miny: float, maxx: float, maxy: float) -> str:
        """
        Ensures a DSM covering the bounding box is available, downloading if needed.
        """
        if self.dsm_path and os.path.exists(self.dsm_path):
            return self.dsm_path

        default_path = os.path.join(self.cache_dir, f"ahn_dsm_{int(minx)}_{int(miny)}_{int(maxx)}_{int(maxy)}.tif")
        if os.path.exists(default_path):
            self.dsm_path = default_path
            return self.dsm_path

        # Check merged AHN5 DSM
        merged_path = os.path.join(self.cache_dir, "dsm_ahn5_merged.tif")
        if os.path.exists(merged_path):
            with rasterio.open(merged_path) as ds:
                b = ds.bounds
                if b.left <= minx and b.bottom <= miny and b.right >= maxx and b.top >= maxy:
                    self.dsm_path = merged_path
                    return self.dsm_path

        # Check generic cache
        generic_path = os.path.join(self.cache_dir, "ahn_dsm.tif")
        if os.path.exists(generic_path):
            # Verify bounds
            with rasterio.open(generic_path) as ds:
                b = ds.bounds
                if b.left <= minx and b.bottom <= miny and b.right >= maxx and b.top >= maxy:
                    self.dsm_path = generic_path
                    return self.dsm_path

        # Fetch via WCS
        self.dsm_path = self.fetch_ahn_dsm_wcs(minx, miny, maxx, maxy, default_path)
        return self.dsm_path

    @staticmethod
    def _sharpen_dsm_edges(z: np.ndarray, gsd: float) -> np.ndarray:
        """
        Sharpens DSM building edges by snapping intermediate wall-height
        pixels to either roof or ground level.

        When a 0.5m AHN DSM is resampled to 0.25m, the building wall
        transition zone spans 1-2 pixels with intermediate heights
        (e.g. 3m on a 10m roof). These intermediate values cause the
        collinearity equations to map edge pixels to the building facade
        in the aerial image, creating a 1-2m positional offset in the
        true orthophoto.

        This function detects the edge zone around each building and
        snaps pixels on the ground-side to DTM height, eliminating
        the blurry ramp without affecting roof interiors or open ground.
        """
        z = z.astype(np.float32)

        # Step 1: Estimate bare earth (DTM) via large morphological opening.
        # A 15m kernel spans most residential buildings without collapsing.
        k_sz = max(3, int(15.0 / gsd))
        if k_sz % 2 == 0:
            k_sz += 1
        kernel_dtm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
        z_dtm = cv2.morphologyEx(z, cv2.MORPH_OPEN, kernel_dtm)

        # Step 2: Normalized DSM (height above ground)
        ndsm = np.maximum(z - z_dtm, 0.0)

        # Step 3: Detect building interiors — nDSM > 2m, cleaned with
        # morphological open (remove noise) + close (fill roof gaps).
        k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        roof_raw   = (ndsm > 2.0).astype(np.uint8)
        roof_open  = cv2.morphologyEx(roof_raw,  cv2.MORPH_OPEN,  k5)
        roof_close = cv2.morphologyEx(roof_open, cv2.MORPH_CLOSE, k5)
        roof_interior = roof_close.astype(bool)

        # Step 4: Edge zone = 1-pixel ring outside building interior
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        roof_dilated = cv2.dilate(roof_close, k3).astype(bool)
        edge_zone    = roof_dilated & ~roof_interior

        # Step 5: Snap edge pixels
        #   • Ground-side edge (nDSM <= 1.0m) → set to DTM (remove ramp)
        #   • Roof-side  edge (nDSM  > 1.0m) → keep z as-is
        z_sharp = z.copy()
        ground_side_edge = edge_zone & (ndsm <= 1.0)
        z_sharp[ground_side_edge] = z_dtm[ground_side_edge]

        print(f"[DSM] Edge sharpening: snapped {int(ground_side_edge.sum())} "
              f"ground-side wall pixels to DTM level.")
        return z_sharp

    def get_grid(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        target_gsd: float = 0.25
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.Affine]:
        """
        Generates regular (X, Y) coordinate grids and samples the DSM at the target GSD.
        Returns:
            X_coords: 1D array of X coordinates (Easting)
            Y_coords: 1D array of Y coordinates (Northing, top to bottom)
            Z_grid: 2D array of DSM heights (shape: [height, width])
            transform: Affine transform for the target grid
        """
        self.ensure_dsm(minx, miny, maxx, maxy)

        # Coordinate arrays
        X_coords = np.arange(minx + target_gsd / 2.0, maxx, target_gsd)
        Y_coords = np.arange(maxy - target_gsd / 2.0, miny, -target_gsd)

        grid_w = len(X_coords)
        grid_h = len(Y_coords)
        transform = from_origin(minx, maxy, target_gsd, target_gsd)

        with rasterio.open(self.dsm_path) as src:
            nodata_val = src.nodata
            # Resample DSM directly to the target window and resolution
            window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            z_sampled = src.read(
                1,
                window=window,
                out_shape=(grid_h, grid_w),
                resampling=Resampling.bilinear
            )

        # Sharpen building edges to remove blurry wall gradients
        z_sampled = self._sharpen_dsm_edges(z_sampled, target_gsd)

        # Handle nodata or invalid values
        invalid_mask = np.isnan(z_sampled)
        if nodata_val is not None:
            invalid_mask |= (z_sampled == nodata_val)
            invalid_mask |= (z_sampled > 1000.0) | (z_sampled < -50.0)
        else:
            invalid_mask |= (z_sampled > 1000.0) | (z_sampled < -50.0)

        if np.any(invalid_mask):
            valid_mask = ~invalid_mask
            if np.any(valid_mask):
                # Nearest neighbor fill using distance transform
                indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
                z_sampled = z_sampled[tuple(indices)]
            else:
                z_sampled = np.full((grid_h, grid_w), 15.0, dtype=np.float32)

        return X_coords, Y_coords, z_sampled.astype(np.float32), transform

    def export_aligned_dsm(
        self,
        output_path: str,
        Z_grid: np.ndarray,
        transform: rasterio.Affine,
        crs: str = "EPSG:28992"
    ):
        """Exports the aligned DSM grid as a georeferenced GeoTIFF."""
        h, w = Z_grid.shape
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            compress="deflate",
            nodata=-9999.0
        ) as dst:
            dst.write(Z_grid, 1)
        print(f"[DSM] Exported aligned DSM to {output_path}")
