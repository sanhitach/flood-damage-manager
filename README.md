# Mapping Flood Damage from Space (Multimodal AI Hackathon 2026 - Track B)

Give the system **an area + a flood date**; it uses Sentinel-1 radar (and optionally Sentinel-2 optical) to
1. **map flooded / debris-covered areas** (before vs after, same orbit track),
2. **estimate damaged buildings, roads and bridges** (pre-event OpenStreetMap),
3. **find settlements cut off** from the nearest town / hospital (road-network connectivity),
4. *(bonus)* **trace the flood path** downstream from an upstream point using the Copernicus DEM,

and produces a **map dashboard** and a **one-page situation report (English + Nepali)**.
Case study: the August 2026 Trishuli / Bhote Koshi flood (compare with EMSR927 - checking only).

> Educational prototype - **not an operational tool**. All outputs are satellite-based estimates that need ground verification.

## Pipeline

```
area + date -> STAC search (same-track S1 pair) -> openEO: calibrate + terrain-correct -> speckle filter, dB
            -> change detection (log-ratio baseline, or U-Net + debris heuristic) + DEM slope mask
            -> flood/debris polygons -> x OSM (ohsome, snapshot BEFORE event): buildings, roads, bridges
            -> road graph minus flooded edges -> settlements cut off (confirmed + conservative scenarios)
            -> (bonus) DEM flow path -> settlements along it
            -> summary.json, GeoJSON layers, dashboard, report.pdf
```

## Setup

```bash
git clone https://github.com/<you>/trishuli-flood-mapper && cd trishuli-flood-mapper
conda env create -f environment.yml && conda activate floodmap      # recommended (GDAL/rasterio are painful with pip)
# or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
Register (free) at <https://dataspace.copernicus.eu>. The first real-data run opens a login prompt (openEO OIDC); tokens are cached.
PDF export uses WeasyPrint (needs Pango). If unavailable, `report.html` is still written - print it to PDF from a browser.
For Nepali text install the *Noto Sans Devanagari* font.

## Run

```bash
# 0. Offline self-test (synthetic valley, no accounts needed) - takes ~2 s
python run_pipeline.py --demo
pytest -q

# 1. Real data, any area/date (this is what the judges will do)
python run_pipeline.py --aoi 85.15 28.00 85.40 28.30 --date 2026-08-26 \
       --upstream 85.30 28.25 --destination 85.18 27.95 --use-s2
#    --aoi    MINLON MINLAT MAXLON MAXLAT (keep it < ~30x30 km while testing; 20 m pixels)
#    --upstream      bonus flood-path start point
#    --destination   extra destination (e.g. district hospital) if OSM has none in the AOI
#    --osm-date      defaults to min(event-1 day, 2026-07-27); post-event OSM is refused by the code

# 2. Your own rasters instead of downloading (projected CRS, 2 bands VV,VH linear power, SAME orbit track)
python run_pipeline.py --before b.tif --after a.tif --dem dem.tif --date 2026-08-26 --osm-dir my_osm/

# 3. Dashboard
streamlit run dashboard/app.py

# 4. Compare with EMSR927 (CHECKING ONLY - never as input)
python -m src.validation.compare_emsr927 --mask outputs/<run>/flood_mask.tif --ref data/EMSR927_flood.geojson \
       --out outputs/<run>/emsr927_comparison.json
```
Outputs land in `outputs/<run>/`: `flood_mask.tif` (1 water, 2 debris), `flood_polygons.geojson`, `buildings_affected.geojson`,
`roads.geojson` (status per road), `settlements.geojson` (status), `flowpath*.geojson`, `summary.json`, `report.pdf/html`, `preview.png`.

## AI component: flood segmentation model (U-Net)

```bash
# Convert Kuro Siwo (and/or Sen1Floods11) to tiles - adapt iter_samples() in src/models/export_npz.py first
python -m src.models.export_npz --src /path/to/KuroSiwo --dst data/tiles --holdout-substr nepal bangladesh
python -m src.models.train --train data/tiles/train --val data/tiles/val --epochs 25 --out models/unet_flood.pt   # GPU / Colab
python -m src.models.evaluate --weights models/unet_flood.pt --data data/tiles/test_holdout --out outputs/eval_unet.json
python run_pipeline.py --method unet ...      # 'auto' uses the U-Net when models/unet_flood.pt exists, else the baseline
```
The evaluation script reports IoU / precision / recall / F1 for the U-Net **and** the log-ratio baseline on held-out scenes.
The U-Net predicts open water; debris is added by a transparent heuristic (backscatter rise next to detected water).

## Results (fill in before submitting)

| Test | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| U-Net, held-out Himalayan/South-Asian tiles | | | | |
| Log-ratio baseline, same tiles | | | | |
| Trishuli case study vs EMSR927 | | | | |

## Data rules (compliance)
Inputs: Sentinel-1, Sentinel-2, Copernicus DEM, OSM **as of a pre-event snapshot**, listed training datasets.
Copernicus EMS / UNOSAT products and post-event OSM edits are **never** read by the pipeline; EMSR927 is used only by
`src/validation/compare_emsr927.py`. `osm_ohsome.get_layers` raises an error if the snapshot is not before the event.

## Limitations (summary - full text in docs/report)
- Sentinel-1 revisits every few days: no early warning, "after" imagery can be days old; the flood may have receded.
- Same-track pairs only; layover/shadow and steep slopes are masked, so flooding there is **not assessed**.
- 10-20 m pixels: narrow flows, single buildings and footpaths are below the resolution. "Hit" = overlap, not damage.
- Cut-off analysis depends on the completeness of pre-event OSM roads, ignores foot/helicopter access, landslide blocks and repairs.
- Debris detection is heuristic; Kuro Siwo is mostly water floods, not debris flows.
- Sentinel-2 is cloud-limited in monsoon season; Sentinel-1 radar is the primary source.
- Nepali text is template-based (numbers inserted by code) and should be reviewed by a native speaker.

## Attributions & citations
- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- (c) OpenStreetMap contributors (ODbL).
- Kuro Siwo: Bountos et al., 2024 (NeurIPS Datasets & Benchmarks). Sen1Floods11: Bonafilia et al., CVPR Workshops 2020 (if used).
- EMSR927 (checking only): European Union, Copernicus Emergency Management Service data.

## Status / honesty notes
The pipeline, damage overlay, cut-off network, flow path, report and dashboard are tested end-to-end on a synthetic valley
(`pytest`). The Copernicus (STAC/openEO) and ohsome calls and the U-Net training code could not be run against live services when
this was written - see `docs/` and test them first.

License: MIT (code). Data licenses as listed above.
