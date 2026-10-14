"""Compare YOUR flood mask with the official Copernicus EMS reference (EMSR927). CHECKING ONLY.

The EMS vectors must NEVER be used as input, training data or for threshold tuning (rule: disqualification).
Download the delineation/monitoring product from https://mapping.emergency.copernicus.eu (GeoJSON / shapefile / GPKG),
then:
   python -m src.validation.compare_emsr927 --mask outputs/<run>/flood_mask.tif --ref data/EMSR927_flood.geojson \
          --out outputs/<run>/emsr927_comparison.json [--aoi-file my_aoi.geojson]

Evaluation region = the AOI file if given, else the bounding box of the reference polygons (EMS maps only cover
the areas it chose to map; pixels outside that are NOT scored, otherwise recall/precision are meaningless).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import box


def compare(mask_path, ref_path, aoi_file=None, classes=(1, 2)) -> dict:
    with rasterio.open(mask_path) as ds:
        mask = np.isin(ds.read(1), classes)
        transform, crs, shape = ds.transform, ds.crs, ds.shape
    ref = gpd.read_file(ref_path).to_crs(crs)
    ref = ref[~ref.geometry.is_empty & ref.geometry.notna()]
    region_geom = gpd.read_file(aoi_file).to_crs(crs).union_all() if aoi_file else box(*ref.total_bounds)
    region = features.rasterize([(region_geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    truth = features.rasterize([(g, 1) for g in ref.geometry], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    p, t = mask & region, truth & region
    tp, fp, fn = float((p & t).sum()), float((p & ~t).sum()), float((~p & t).sum())
    px_km2 = abs(transform.a * transform.e) / 1e6
    iou = tp / max(tp + fp + fn, 1)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"iou": round(iou, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "area_pred_km2": round(float(p.sum()) * px_km2, 2), "area_ref_km2": round(float(t.sum()) * px_km2, 2),
            "region_km2": round(float(region.sum()) * px_km2, 2),
            "note": "Reference = EMSR927 (checking only). Pixels outside the evaluation region are not scored."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--aoi-file")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = compare(a.mask, a.ref, a.aoi_file)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
