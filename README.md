# True Orthophoto Generation Pipeline (Delft, Netherlands - EPSG:28992)

A modular, high-performance, and reproducible Python photogrammetry pipeline for generating **True Orthophotos** from Dutch aerial survey imagery (`bm-2026-stereo`), official Dutch Kadaster BAG cadastral footprints (`data/bag_pand.gpkg`), and the Dutch National Elevation Model (AHN5 0.5m DSM).

---

## 1. Overview & Acceptance Criteria

Conventional orthophotos project aerial imagery onto a bare-earth Digital Terrain Model (DTM). For tall structures, this causes **radial displacement ("lean")**, exposing wall facades and occluding the ground behind them. In contrast, this pipeline produces a **True Orthophoto** where:

1. **Buildings must not lean**: Building roofs sit strictly within their cadastral BAG footprints; zero visible wall facades.
2. **Hidden spots filled with real ground**: Blind spots behind buildings are detected via Line-of-Sight (LOS) raycasting across the regularised DSM and seamlessly patched from overlapping photos that can see the ground.
3. **Seamlines navigate around buildings**: Seamlines are routed around building boundaries (never cutting across a roof), with radiometric balancing across photos.
4. **Sharpness & Geometric Precision**: Crisp textures with sub-pixel alignment in Dutch RD New (`EPSG:28992`).

---

## 2. Data Sources & Manual Portal Steps

### 1. Aerial Survey Photos (`bm-2026-stereo`)
- **Portal**: [Beeldmateriaal Nederland / PDOK](https://www.beeldmateriaal.nl/)
- **Dataset**: 2026 Spring High-Resolution Stereo Flight (Delft AOI).
- **Files**:
  - Image TIFFs: `bm-2026-stereo/1/1/beelden_RGB_8bit/*.tif` (Vexcel UltraCam Eagle M3, 8-bit RGB, 26460 x 17004 px).
  - Projection Centers: `bm-2026-stereo/1/1/metadata/2026_projectiecentra_01_01_hrl.txt` (Exterior orientation: $X_0, Y_0, Z_0$, $\omega, \varphi, \kappa$).
  - Camera Calibration: `bm-2026-stereo/1/1/metadata/2026_431S61680X916102-f100_Rev09.00_V01.pdf` (Focal length $100.5$ mm, pixel size $4.0\ \mu\text{m}$, principal point $x_{ppa} = -0.120$ mm, $y_{ppa} = 0.0$ mm).

### 2. AHN5 Elevation Model (0.5m DSM)
- **Portal**: [PDOK AHN5](https://www.pdok.nl/introductie/-/article/actueel-hoogtebestand-nederland-ahn-)
- **Coverage**: Merged 0.5m DSM raster covering Delft AOI at `data/dsm_ahn5_merged.tif` (10000 x 10000 pixels, $82000 \le X \le 87000$, $443000 \le Y \le 448000$).

### 3. BAG Cadastral Building Footprints (`data/bag_pand.gpkg`)
- **Why BAG is used**: The Basisregistratie Adressen en Gebouwen (BAG) from Kadaster is the authoritative legal register of all building polygons in the Netherlands. It serves two vital roles:
  1. **DSM Regularisation Source**: Buildings are regularised to flat roof heights (90th percentile of DSM) with vertical walls placed precisely on the cadastral line, clearing 2m wall ramps/halos.
  2. **Independent Verification Reference**: Eliminates circular validation by testing orthophoto roof alignment against official municipal boundaries rather than DSM-derived contours.
- **Download Command** (one-time, then runs 100% offline):
  ```bash
  python download_bag.py --bbox 83500 445300 84500 446300 --out data/bag_pand.gpkg
  ```
  *(Filters out invalid statuses: `gesloopt`, `niet gerealiseerd`, `ten onrechte`, `bouwvergunning`)*.

---

## 3. Installation & CPU Environment

This pipeline is optimized for pure CPU execution with zero GPU requirements.

### System Requirements
- Python 3.10, 3.11, or 3.12 (64-bit)
- Windows, Linux, or macOS

### Installation Steps
```bash
# 1. Clone repository & enter workspace
cd "d:\Aleem bhai"

# 2. Install dependencies
pip install -r requirements.txt
```

### Core Libraries
- `numba`: Just-In-Time parallel CPU compiler for multi-threaded Line-of-Sight raycasting.
- `rasterio` & `pyproj`: Cloud-optimized GeoTIFF I/O and Dutch RD New (`EPSG:28992`) projection.
- `geopandas` & `shapely`: Offline cadastral vector processing.
- `opencv-python-headless`: Bilinear image sampling, morphological DTM filtering, and edge detection.

---

## 4. Execution Workflow

### Step 1: Download Official BAG Footprints (Offline Prep)
```bash
python download_bag.py --bbox 83500 445300 84500 446300 --out data/bag_pand.gpkg
```

### Step 2: Generate True Orthophoto (Safe Streaming Write)
```bash
python main.py --bbox 83500 445300 84500 446300 --out-dir output_bag
```
*Features:*
- Safe write: writes to `true_orthophoto.tmp.tif`, renames to `true_orthophoto.tif` upon completion.
- Full integrity check: verifies every raster block (asserts 0 unreadable blocks).
- Building-aware seamline avoidance: routes seamlines around buildings.

### Step 3: Run Honest Measurement & Quality Verification
```bash
python verify_bag_ortho.py
```
*Generates:*
- Quantitative measurement table across 30 buildings of mixed height ($\ge 10$ tall).
- 10 tall building outline-only crops (**thin red line, NO fill**) comparing before vs after.
- Full verification report at `output_bag/verification/REPORT.md`.

---

## 5. Pipeline Deliverables & Outputs

All deliverables for the current run are saved in `output_bag/`:

| Output File | Description |
|---|---|
| `output_bag/true_orthophoto.tif` | 0.25m True Orthophoto GeoTIFF in EPSG:28992 (4000 x 4000 px). |
| `output_bag/aligned_dsm.tif` | Footprint-regularised Digital Surface Model in EPSG:28992. |
| `output_bag/verification/REPORT.md` | Verification report with Check 1-4 numbers and findings. |
| `output_bag/verification/crops_before_after/` | High-resolution outline-only crops of 10 tall buildings. |

---

## 6. Known Limitations

1. **Temporal Discrepancy**: AHN5 LiDAR acquisition predates the April 2026 aerial survey photos. Structures constructed, renovated, or demolished between AHN5 scanning and 2026 lack accurate LiDAR heights.
2. **Flight Strip Geometry**: The test area is covered by 2 primary stereo flight strips. In narrow urban street canyons oriented perpendicular to the flight path, viewing angles are limited to 2 opposing perspectives, requiring fallback filling from the best nadir photo.
