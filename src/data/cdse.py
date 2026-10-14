"""Copernicus Data Space Ecosystem access: STAC search for a SAME-TRACK Sentinel-1 pair,
openEO processing (radiometric calibration + terrain correction on the server), Copernicus DEM, Sentinel-2 NDWI.

First run opens a login prompt (free account: https://dataspace.copernicus.eu).
NOTE: property names (sat:relative_orbit, sat:orbit_state) follow the STAC SAR/SAT extensions; if the CDSE
catalogue returns them differently, adjust `find_s1_pair`. This module could not be exercised against the live
service when it was written - test it on day 1.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from shapely.geometry import box, shape

from src.common import log, utm_epsg


# ----------------------------------------------------------------------------- Sentinel-1 pair selection
def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def find_s1_pair(bbox, event: date, cfg: dict) -> dict:
    """Pick a before/after Sentinel-1 IW GRD pair on the SAME relative orbit and orbit direction.

    before = latest acquisition before the event date; after = earliest acquisition at least 1 day after it,
    (so a same-day acquisition of unknown time is never used). Closest-to-event pair wins."""
    from pystac_client import Client

    days = cfg["max_pair_search_days"]
    start = datetime.combine(event - timedelta(days=days), datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(event + timedelta(days=days), datetime.max.time(), tzinfo=timezone.utc)
    client = Client.open(cfg["stac_url"])
    search = client.search(collections=["sentinel-1-grd"], bbox=list(bbox),
                           datetime=f"{start.isoformat()}/{end.isoformat()}", max_items=1000)
    aoi = box(*bbox)
    tracks: dict = {}
    for it in search.items():
        p = it.properties
        if p.get("sar:instrument_mode") not in (None, "IW"):
            continue
        cover = shape(it.geometry).intersection(aoi).area / aoi.area
        if cover < cfg["min_aoi_coverage"]:
            continue
        dt = _parse_dt(p["datetime"])
        rel = p.get("sat:relative_orbit")
        state = p.get("sat:orbit_state")
        key = (rel, state) if rel is not None else (None, state, dt.hour * 6 + dt.minute // 10)
        tracks.setdefault(key, []).append((dt, it.id, cover))
    if not tracks:
        raise RuntimeError("No Sentinel-1 IW GRD scenes cover the AOI in the search window.")

    ev0 = datetime.combine(event, datetime.min.time(), tzinfo=timezone.utc)
    ev1 = ev0 + timedelta(days=1)
    best = None
    for key, rows in tracks.items():
        befores = [r for r in rows if r[0] < ev0]
        afters = [r for r in rows if r[0] >= ev1]
        if not befores or not afters:
            continue
        b = max(befores, key=lambda r: r[0])
        a = min(afters, key=lambda r: r[0])
        score = (ev0 - b[0]).total_seconds() + (a[0] - ev1).total_seconds()
        if best is None or score < best[0]:
            best = (score, key, b, a)
    if best is None:
        raise RuntimeError("No orbit track has BOTH a before and an after acquisition. Widen max_pair_search_days.")
    _, key, b, a = best
    gap = (a[0] - b[0]).days
    log.info("Same-track pair: track=%s before=%s after=%s (gap %d d)", key, b[0].date(), a[0].date(), gap)
    return {"track": [str(k) for k in key], "before_datetime": b[0].isoformat(), "after_datetime": a[0].isoformat(),
            "before_id": b[1], "after_id": a[1], "gap_days": gap,
            "aoi_coverage": min(b[2], a[2])}


# ----------------------------------------------------------------------------- openEO
def _connect(cfg):
    import openeo
    return openeo.connect(cfg["openeo_url"]).authenticate_oidc()


def _extent(bbox):
    w, s, e, n = bbox
    return {"west": w, "south": s, "east": e, "north": n, "crs": "EPSG:4326"}


def fetch_s1(bbox, acquisition_iso: str, out_path, cfg: dict, conn=None):
    """Download ONE Sentinel-1 acquisition (calibrated + terrain-corrected) as 2-band GeoTIFF: VV, VH in LINEAR power."""
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    conn = conn or _connect(cfg)
    t = _parse_dt(acquisition_iso)
    window = [(t - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
              (t + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")]
    epsg = utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    cube = conn.load_collection("SENTINEL1_GRD", spatial_extent=_extent(bbox), temporal_extent=window,
                                bands=["VV", "VH"])
    cube = cube.sar_backscatter(coefficient=cfg["backscatter_coefficient"])
    cube = cube.resample_spatial(projection=epsg, resolution=cfg["s1_resolution_m"])
    cube = cube.reduce_dimension(dimension="t", reducer="mean")
    log.info("openEO batch job: Sentinel-1 %s (can take several minutes)", acquisition_iso)
    cube.execute_batch(outputfile=str(out_path), out_format="GTiff")
    return out_path


def fetch_dem(bbox, out_path, cfg: dict, conn=None):
    """Copernicus DEM GLO-30 as GeoTIFF (metres)."""
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    conn = conn or _connect(cfg)
    epsg = utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    cube = conn.load_collection("COPERNICUS_30", spatial_extent=_extent(bbox)).max_time()
    cube = cube.resample_spatial(projection=epsg, resolution=cfg["s1_resolution_m"])
    cube.execute_batch(outputfile=str(out_path), out_format="GTiff")
    return out_path


def fetch_s2_ndwi(bbox, event: date, out_path, cfg: dict, s2cfg: dict, conn=None):
    """Cloud-masked median NDWI (green-NIR) for the days AFTER the event. Cloudy pixels -> nodata."""
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    conn = conn or _connect(cfg)
    start = (event + timedelta(days=1)).isoformat()
    end = (event + timedelta(days=1 + s2cfg["search_days"])).isoformat()
    epsg = utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    s2 = conn.load_collection("SENTINEL2_L2A", spatial_extent=_extent(bbox), temporal_extent=[start, end],
                              bands=["B03", "B08", "SCL"], max_cloud_cover=70)
    scl = s2.band("SCL")
    cloud = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 1)
    g, n = s2.band("B03"), s2.band("B08")
    ndwi = ((g - n) / (g + n)).mask(cloud)
    ndwi = ndwi.reduce_dimension(dimension="t", reducer="median")
    ndwi = ndwi.resample_spatial(projection=epsg, resolution=cfg["s1_resolution_m"])
    ndwi.execute_batch(outputfile=str(out_path), out_format="GTiff")
    return out_path
