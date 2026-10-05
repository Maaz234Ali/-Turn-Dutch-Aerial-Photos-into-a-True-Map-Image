"""
camera_parser.py
Parses Interior and Exterior Orientation for aerial photogrammetric imagery.
Designed for Vexcel UltraCam Eagle M3 imagery and Dutch projection center tables.
"""

import os
import glob
import numpy as np
import rasterio

class Camera:
    """
    Represents an aerial camera station with Interior and Exterior Orientation.
    """
    def __init__(
        self,
        image_name: str,
        image_path: str,
        X0: float,
        Y0: float,
        Z0: float,
        omega: float,
        phi: float,
        kappa: float,
        focal_length_mm: float = 100.5,
        pixel_size_mm: float = 0.004,
        x_ppa_mm: float = -0.120,
        y_ppa_mm: float = 0.0,
        width_px: int = 26460,
        height_px: int = 17004,
    ):
        self.image_name = image_name
        self.image_path = image_path
        self.X0 = float(X0)
        self.Y0 = float(Y0)
        self.Z0 = float(Z0)
        self.omega_deg = float(omega)
        self.phi_deg = float(phi)
        self.kappa_deg = float(kappa)

        # Interior orientation
        self.f_m = focal_length_mm / 1000.0
        self.pixel_size_m = pixel_size_mm / 1000.0
        self.x_ppa_m = x_ppa_mm / 1000.0
        self.y_ppa_m = y_ppa_mm / 1000.0
        self.width = width_px
        self.height = height_px

        self.c_mid = (self.width - 1) / 2.0
        self.r_mid = (self.height - 1) / 2.0

        # Construct rotation matrix R = (Rx * Ry * Rz)^T
        om = np.radians(self.omega_deg)
        ph = np.radians(self.phi_deg)
        kp = np.radians(self.kappa_deg)

        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(om), -np.sin(om)],
            [0.0, np.sin(om), np.cos(om)]
        ], dtype=np.float64)

        Ry = np.array([
            [np.cos(ph), 0.0, np.sin(ph)],
            [0.0, 1.0, 0.0],
            [-np.sin(ph), 0.0, np.cos(ph)]
        ], dtype=np.float64)

        Rz = np.array([
            [np.cos(kp), -np.sin(kp), 0.0],
            [np.sin(kp), np.cos(kp), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Direction cosine matrix mapping ground vectors to camera axes
        self.R = (Rx @ Ry @ Rz).T.astype(np.float64)

        # Approximate ground footprint bounding box
        # Footprint at mean ground elevation ~15m
        mean_ground_z = 15.0
        H_agl = max(self.Z0 - mean_ground_z, 100.0)
        footprint_x = (self.width * self.pixel_size_m / self.f_m) * H_agl
        footprint_y = (self.height * self.pixel_size_m / self.f_m) * H_agl
        radius = np.hypot(footprint_x, footprint_y) * 0.65
        self.bbox = (self.X0 - radius, self.Y0 - radius, self.X0 + radius, self.Y0 + radius)

    def intersects_bbox(self, minx: float, miny: float, maxx: float, maxy: float) -> bool:
        """Check if camera footprint roughly overlaps with the target ground bounding box."""
        bx0, by0, bx1, by1 = self.bbox
        return not (bx1 < minx or bx0 > maxx or by1 < miny or by0 > maxy)

    def __repr__(self) -> str:
        return f"<Camera {self.image_name} (X={self.X0:.1f}, Y={self.Y0:.1f}, Z={self.Z0:.1f})>"


def parse_inner_orientation_from_tiff(tiff_path: str) -> dict:
    """Extract interior orientation parameters from UltraCam TIFF header tags if present."""
    default_params = {
        'focal_length_mm': 100.5,
        'pixel_size_mm': 0.004,
        'x_ppa_mm': -0.120,
        'y_ppa_mm': 0.0,
        'width_px': 26460,
        'height_px': 17004,
    }
    try:
        with rasterio.open(tiff_path) as src:
            default_params['width_px'] = src.width
            default_params['height_px'] = src.height
            tags = src.tags()
            desc = tags.get('TIFFTAG_IMAGEDESCRIPTION', '')
            for line in desc.splitlines():
                line = line.strip()
                if 'PRINCIPAL_DISTANCE:' in line:
                    val = line.split(':')[1].split('[')[0].strip()
                    default_params['focal_length_mm'] = float(val)
                elif 'PIXEL_SIZE_WIDTH:' in line:
                    val = line.split(':')[1].split('[')[0].strip()
                    default_params['pixel_size_mm'] = float(val) / 1000.0  # micron to mm
                elif 'PRINCIPAL_POINT_X:' in line:
                    val = line.split(':')[1].split('[')[0].strip()
                    default_params['x_ppa_mm'] = float(val)
                elif 'PRINCIPAL_POINT_Y:' in line:
                    val = line.split(':')[1].split('[')[0].strip()
                    default_params['y_ppa_mm'] = float(val)
    except Exception as e:
        pass
    return default_params


def load_all_cameras(workspace_root: str) -> dict[str, Camera]:
    """
    Scans the workspace for aerial photos and exterior orientation files,
    returning a dictionary of {image_name: Camera}.
    """
    # 1. Discover all .tif images
    image_paths = glob.glob(os.path.join(workspace_root, '**', 'beelden_RGB_8bit', '*.tif'), recursive=True)
    images_by_name = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}

    if not images_by_name:
        # Fallback search for any .tif
        all_tifs = glob.glob(os.path.join(workspace_root, '**', '*.tif'), recursive=True)
        images_by_name = {os.path.splitext(os.path.basename(p))[0]: p for p in all_tifs if 'hrl' in os.path.basename(p).lower()}

    # 2. Discover exterior orientation txt files
    txt_files = glob.glob(os.path.join(workspace_root, '**', 'metadata', '*projectiecentra*.txt'), recursive=True)
    if not txt_files:
        txt_files = glob.glob(os.path.join(workspace_root, '**', '*projectiecentra*.txt'), recursive=True)

    # Cache interior orientation from first image
    inner_params = None
    first_img = next(iter(images_by_name.values())) if images_by_name else None
    if first_img:
        inner_params = parse_inner_orientation_from_tiff(first_img)
    else:
        inner_params = {
            'focal_length_mm': 100.5,
            'pixel_size_mm': 0.004,
            'x_ppa_mm': -0.120,
            'y_ppa_mm': 0.0,
            'width_px': 26460,
            'height_px': 17004,
        }

    cameras = {}
    for txt_path in txt_files:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    name = parts[0]
                    if name in images_by_name:
                        X0 = float(parts[1])
                        Y0 = float(parts[2])
                        Z0 = float(parts[3])
                        omega = float(parts[4])
                        phi = float(parts[5])
                        kappa = float(parts[6])

                        cam = Camera(
                            image_name=name,
                            image_path=images_by_name[name],
                            X0=X0,
                            Y0=Y0,
                            Z0=Z0,
                            omega=omega,
                            phi=phi,
                            kappa=kappa,
                            **inner_params
                        )
                        cameras[name] = cam

    return cameras


if __name__ == '__main__':
    cams = load_all_cameras('.')
    print(f"Loaded {len(cams)} cameras.")
    if cams:
        first = next(iter(cams.values()))
        print("Sample Camera:", first)
        print("Rotation Matrix:\n", first.R)
