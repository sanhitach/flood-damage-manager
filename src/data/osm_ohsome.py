"""Pre-event OpenStreetMap via the ohsome API (history snapshot).

DATA RULE: only snapshots dated BEFORE the event are allowed. `get_layers` refuses otherwise.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import requests

from src.common import log

OHSOME_URL = "https://api.ohsome.org/v1/elements/geometry"

ROAD_TYPES = "motorway,trunk,primary,secondary,tertiary,unclassified,residential,service,track," \
             "motorway_link,trunk_link,primary_link,secondary_link,tertiary_link,living_street"

FILTERS = {
    "buildings": "building=* and geometry:polygon",
    "roads": f"highway in ({ROAD_TYPES}) and geometry:line",
    "hospitals": "(amenity=hospital or healthcare=hospital)",
    "settlements": "place in (city,town,village,hamlet)",
}


def _empty() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")


def _query(bbox, snapshot: str, flt: str, cache_dir: Path) -> gpd.GeoDataFrame:
    key = hashlib.md5(f"{bbox}|{snapshot}|{flt}".encode()).hexdigest()[:12]
    cache = cache_dir / f"ohsome_{key}.geojson"
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        log.info("ohsome: %s @ %s", flt, snapshot)
        r = requests.post(
            OHSOME_URL,
            data={"bboxes": ",".join(f"{v:.6f}" for v in bbox), "time": snapshot,
                  "filter": flt, "properties": "tags"},
            timeout=600,
        )
        if r.status_code != 200:
            raise RuntimeError(f"ohsome error {r.status_code}: {r.text[:400]}")
        data = r.json()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    feats = data.get("features", [])
    if not feats:
        return _empty()
    return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")


def get_layers(bbox, snapshot: str, event_date: date, cache_dir="data/osm_cache", osm_dir=None) -> dict:
    """Return dict of GeoDataFrames (EPSG:4326): buildings, roads, hospitals, settlements.

    If `osm_dir` holds <layer>.geojson files they are used instead of the API (offline/demo mode)."""
    if osm_dir is not None:
        out = {}
        for name in FILTERS:
            p = Path(osm_dir) / f"{name}.geojson"
            out[name] = gpd.read_file(p) if p.exists() else _empty()
        return out

    if date.fromisoformat(snapshot) >= event_date:
        raise ValueError(f"OSM snapshot {snapshot} is not before the event ({event_date}). "
                         "Post-event OSM edits are NOT allowed as inputs (disqualification risk).")
    cache_dir = Path(cache_dir)
    return {name: _query(bbox, snapshot, flt, cache_dir) for name, flt in FILTERS.items()}


def col(gdf: gpd.GeoDataFrame, name: str, default=None):
    """Safe column access: ohsome only returns tag columns that exist in the data."""
    if name in gdf.columns:
        return gdf[name]
    import pandas as pd
    return pd.Series([default] * len(gdf), index=gdf.index)
