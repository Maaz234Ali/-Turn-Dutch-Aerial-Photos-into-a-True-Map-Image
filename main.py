"""
main.py
End-to-End True Orthophoto Generation Pipeline.
Entry point for processing Dutch aerial survey imagery and AHN elevation data.
"""

import os
import sys
import argparse
import time

from camera_parser import load_all_cameras
from dsm_manager import DSMManager
from mosaic import generate_true_orthophoto


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End True Orthophoto Generation Pipeline (EPSG:28992)")
    parser.add_argument(
        "--workspace",
        type=str,
        default=".",
        help="Workspace directory containing aerial photos and metadata"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=[220800.0, 454400.0, 221800.0, 455400.0],
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Target bounding box in Dutch RD New (EPSG:28992) coordinates"
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=0.25,
        help="Output Ground Sampling Distance (GSD) in meters (default: 0.25m)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output",
        help="Output directory for generated GeoTIFFs (default: output)"
    )
    parser.add_argument(
        "--dsm-file",
        type=str,
        default=None,
        help="Optional local path to AHN DSM raster (if omitted, auto-fetches via PDOK WCS)"
    )
    parser.add_argument(
        "--traditional",
        action="store_true",
        help="Generate a traditional orthophoto mosaic without removing building facades"
    )
    return parser.parse_args()


def run_pipeline(
    workspace: str = ".",
    bbox: list[float] = None,
    gsd: float = 0.25,
    out_dir: str = "output",
    dsm_file: str = None,
    traditional: bool = False
):
    start_time = time.time()
    if bbox is None:
        bbox = [220800.0, 454400.0, 221800.0, 455400.0]

    minx, miny, maxx, maxy = bbox
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" TRUE ORTHOPHOTO GENERATION PIPELINE ")
    print(" Coordinate System: RD New / Amersfoort (EPSG:28992)")
    print(f" Target Bounding Box: X=[{minx:.1f}, {maxx:.1f}], Y=[{miny:.1f}, {maxy:.1f}]")
    print(f" Target Resolution: {gsd:.2f} m/pixel (Extent: {(maxx-minx):.0f}m x {(maxy-miny):.0f}m)")
    print("=" * 70)

    # 1. Discover and parse camera orientation metadata
    print("\n[Step 1] Loading camera metadata and imagery...")
    all_cameras = load_all_cameras(workspace)
    print(f"  Total aerial photos with orientation: {len(all_cameras)}")

    # Filter cameras that intersect target bbox
    intersecting_cams = [
        cam for cam in all_cameras.values()
        if cam.intersects_bbox(minx, miny, maxx, maxy)
    ]
    print(f"  Photos intersecting target area: {len(intersecting_cams)}")
    for c in intersecting_cams:
        print(f"   * {c.image_name} (X={c.X0:.1f}, Y={c.Y0:.1f}, Z={c.Z0:.1f}, Yaw={c.kappa_deg:.1f}°)")

    if not intersecting_cams:
        print("ERROR: No photos intersect the designated bounding box.")
        sys.exit(1)

    # 2. Prepare AHN DSM grid
    print("\n[Step 2] Preparing Digital Surface Model (DSM)...")
    dsm_mgr = DSMManager(dsm_path=dsm_file)
    X_grid, Y_grid, Z_dsm, transform = dsm_mgr.get_grid(minx, miny, maxx, maxy, target_gsd=gsd)
    print(f"  DSM Grid Shape: {Z_dsm.shape[1]} cols x {Z_dsm.shape[0]} rows")
    print(f"  Elevation range: {Z_dsm.min():.2f}m to {Z_dsm.max():.2f}m (NAP)")

    # Export aligned DSM GeoTIFF
    aligned_dsm_path = os.path.join(out_dir, "aligned_dsm.tif")
    dsm_mgr.export_aligned_dsm(aligned_dsm_path, Z_dsm, transform)

    # 3. Synthesize True Orthophoto
    print("\n[Step 3] Multi-View True Orthophoto Synthesis...")
    true_ortho_path = os.path.join(out_dir, "true_orthophoto.tif")
    mosaic_result = generate_true_orthophoto(
        cameras=intersecting_cams,
        X_grid=X_grid,
        Y_grid=Y_grid,
        Z_dsm=Z_dsm,
        transform=transform,
        output_path=true_ortho_path,
        gsd=gsd,
        traditional=traditional
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f" Pipeline finished successfully in {elapsed:.1f} seconds!")
    print(f" 1. True Orthophoto: {os.path.abspath(true_ortho_path)}")
    print(f" 2. Aligned DSM:     {os.path.abspath(aligned_dsm_path)}")
    print("=" * 70)

    return mosaic_result


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        workspace=args.workspace,
        bbox=args.bbox,
        gsd=args.gsd,
        out_dir=args.out_dir,
        dsm_file=args.dsm_file,
        traditional=args.traditional
    )
