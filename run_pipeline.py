#!/usr/bin/env python
"""End-to-end flood damage pipeline.

  python run_pipeline.py --aoi 85.15 28.00 85.40 28.30 --date 2026-08-26          # real data via CDSE + ohsome
  python run_pipeline.py --demo                                                   # offline synthetic valley
  python run_pipeline.py --before b.tif --after a.tif --dem dem.tif --date ...    # your own rasters (projected CRS, linear VV,VH)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from src.baseline import change_detection
from src.common import (ATTRIBUTION, DISCLAIMER, Grid, dump_json, load_config, log, read_aligned, save_geojson,
                        write_raster)
from src.damage import overlay
from src.data import osm_ohsome
from src.network import cutoff
from src.preprocess import sar


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", nargs=4, type=float, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    p.add_argument("--date", help="flood date YYYY-MM-DD")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default=None, help="output root (default from config)")
    p.add_argument("--method", choices=["auto", "baseline", "unet"], default="auto")
    p.add_argument("--weights", default="models/unet_flood.pt")
    p.add_argument("--osm-date", default=None, help="OSM snapshot (must be BEFORE the event)")
    p.add_argument("--osm-dir", default=None, help="folder with buildings/roads/hospitals/settlements .geojson (offline)")
    p.add_argument("--before"); p.add_argument("--after"); p.add_argument("--dem")
    p.add_argument("--use-s2", action="store_true", help="add Sentinel-2 NDWI confirmation (cloud-masked)")
    p.add_argument("--upstream", nargs=2, type=float, metavar=("LON", "LAT"), help="bonus: trace flood path from here")
    p.add_argument("--destination", nargs=2, type=float, action="append", metavar=("LON", "LAT"),
                   help="extra destination (e.g. district hospital); repeatable")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--no-report", action="store_true")
    return p.parse_args(argv)


def _count(df, status):
    return int((df["status"] == status).sum()) if len(df) else 0


def run(a) -> Path:
    t0 = time.time()
    cfg = load_config(a.config)
    warnings: list[str] = []
    meta = {}
    if a.demo:
        from src.demo.synthetic import make_demo
        meta = make_demo(Path(cfg["paths"]["data_dir"]) / "demo")
        a.before, a.after, a.dem = meta["before"], meta["after"], meta["dem"]
        a.osm_dir = meta["osm_dir"]
        a.aoi = a.aoi or meta["aoi"]
        a.date = a.date or meta["event_date"]
        a.upstream = a.upstream or meta["upstream_lonlat"]
        warnings.append("SYNTHETIC DEMO DATA - not a real event.")
    if not a.date:
        raise SystemExit("--date is required")
    event = date.fromisoformat(a.date)
    local = bool(a.before and a.after)
    if not local and not a.aoi:
        raise SystemExit("--aoi is required unless --before/--after are given")

    latest = date.fromisoformat(cfg["osm"]["latest_available"])
    osm_date = a.osm_date or min(event - timedelta(days=1), latest).isoformat()

    run_id = "demo" if a.demo else f"{a.date}_{hashlib.md5(str(a.aoi or a.before).encode()).hexdigest()[:6]}"
    out = Path(a.out or cfg["paths"]["output_dir"]) / run_id
    out.mkdir(parents=True, exist_ok=True)
    work = Path(cfg["paths"]["data_dir"]) / run_id
    work.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. data
    s1_pair = None
    if not local:
        from src.data import cdse
        s1_pair = cdse.find_s1_pair(a.aoi, event, cfg["cdse"])
        conn = cdse._connect(cfg["cdse"])
        a.before = str(cdse.fetch_s1(a.aoi, s1_pair["before_datetime"], work / "s1_before.tif", cfg["cdse"], conn))
        a.after = str(cdse.fetch_s1(a.aoi, s1_pair["after_datetime"], work / "s1_after.tif", cfg["cdse"], conn))
        a.dem = a.dem or str(cdse.fetch_dem(a.aoi, work / "dem.tif", cfg["cdse"], conn))
    grid = Grid.from_raster(a.before)
    if not grid.crs.is_projected:
        raise SystemExit("Rasters must be in a projected CRS (e.g. UTM). Reproject with gdalwarp first.")
    if a.aoi is None:
        a.aoi = list(grid.bounds_wgs84())
    log.info("Grid %dx%d @ %.0f m (%s)", grid.height, grid.width, grid.pixel_size, grid.crs)

    before = read_aligned(a.before, grid)
    after = read_aligned(a.after, grid)
    if a.dem:
        dem = read_aligned(a.dem, grid)[0]
        slope = sar.slope_deg(dem, grid.pixel_size)
    else:
        dem, slope = None, np.zeros(grid.shape, "float32")
        warnings.append("No DEM: terrain (layover/shadow) masking and flow-path trace skipped.")

    # ---------------------------------------------------------------- 2. flood / debris detection
    dcfg = cfg["detection"]
    method = a.method
    if method == "auto":
        method = "unet" if Path(a.weights).exists() else "baseline"
    base_mask, change = change_detection.detect(before, after, slope, grid.pixel_size, dcfg)
    if method == "unet":
        from src.models import infer
        k = int(dcfg["speckle_window"])
        b_db, a_db = sar.to_db(sar.boxcar(before, k)), sar.to_db(sar.boxcar(after, k))
        x = sar.normalize_db(np.concatenate([b_db, a_db]))
        prob = infer.predict(a.weights, x, dcfg["unet_tile"], dcfg["unet_overlap"])
        valid = np.isfinite(b_db).all(0) & np.isfinite(a_db).all(0)
        permanent = b_db[0] < dcfg["water_db"]
        water = (prob > dcfg["unet_threshold"]) & ~permanent
        mask = change_detection.finalize_mask(water, base_mask == 2, valid, slope < dcfg["max_slope_deg"],
                                              grid.pixel_size, dcfg)
        write_raster(out / "unet_probability.tif", prob.astype("float32"), grid, nodata=np.nan)
    else:
        mask = base_mask
    write_raster(out / "flood_mask.tif", mask, grid, nodata=0, dtype="uint8")
    write_raster(out / "change_db.tif", change, grid, nodata=np.nan, dtype="float32")
    write_raster(out / "baseline_mask.tif", base_mask, grid, nodata=0, dtype="uint8")
    px_km2 = grid.pixel_size**2 / 1e6
    water_km2, debris_km2 = float((mask == 1).sum() * px_km2), float((mask == 2).sum() * px_km2)

    s2_info = None
    if a.use_s2 and not local:
        try:
            from src.data import cdse
            ndwi_p = cdse.fetch_s2_ndwi(a.aoi, event, work / "s2_ndwi.tif", cfg["cdse"], cfg["s2"])
            ndwi = read_aligned(ndwi_p, grid)[0]
            clear = np.isfinite(ndwi) & (np.abs(ndwi) <= 1)
            fl = (mask == 1) & clear
            s2_info = {"clear_fraction": float(clear.mean()), "water_pixels_in_clear_area": int(fl.sum()),
                       "confirmed_fraction": float((ndwi[fl] > cfg["s2"]["ndwi_water"]).mean()) if fl.any() else None}
        except Exception as e:
            warnings.append(f"Sentinel-2 step failed ({e}); radar-only result.")

    flood_gdf = overlay.mask_to_gdf(mask, grid, dcfg["min_area_m2"])
    save_geojson(flood_gdf, out / "flood_polygons.geojson")

    # ---------------------------------------------------------------- 3. damage overlay
    layers = osm_ohsome.get_layers(a.aoi, osm_date, event, work / "osm_cache", a.osm_dir)
    L = {k: v.to_crs(grid.crs) if len(v) else v for k, v in layers.items()}
    dmg = overlay.analyse_damage(L, flood_gdf, cfg["damage"]["near_buffer_m"])
    buildings, roads = dmg["buildings"], dmg["roads"]
    br = roads[roads["is_bridge"]] if len(roads) else roads
    save_geojson(buildings[buildings.status != "unaffected"] if len(buildings) else buildings,
                 out / "buildings_affected.geojson")
    save_geojson(roads, out / "roads.geojson")

    # ---------------------------------------------------------------- 4. cut-off analysis
    sett_all = L["settlements"]
    place = osm_ohsome.col(sett_all, "place", "").astype(str) if len(sett_all) else None
    def pts(g):
        g = g.copy()
        if len(g):
            g["geometry"] = g.geometry.representative_point()
        return g
    villages = pts(sett_all[place.isin(["village", "hamlet"])]) if len(sett_all) else sett_all
    towns = pts(sett_all[place.isin(["town", "city"])]) if len(sett_all) else sett_all
    hospitals = pts(L["hospitals"])
    for g in (villages, towns, hospitals):
        if len(g):
            g["name"] = osm_ohsome.col(g, "name", "").fillna("").replace("", "unnamed")
    extra = None
    if a.destination:
        from pyproj import Transformer
        tr = Transformer.from_crs(4326, grid.crs, always_xy=True)
        extra = [tr.transform(lon, lat) for lon, lat in a.destination]
    if len(roads):
        settle, ninfo = cutoff.analyse(roads, dmg["union"], dmg["buffer"], villages, hospitals, towns,
                                       cfg["network"]["snap_m"], extra)
        warnings += ninfo["warnings"]
    else:
        settle, ninfo = villages.assign(status="no_roads", status_conservative="no_roads"), {}
        warnings.append("No roads found in OSM for this AOI; cut-off analysis not possible.")
    save_geojson(settle, out / "settlements.geojson")
    save_geojson(hospitals, out / "hospitals.geojson")
    save_geojson(towns, out / "towns.geojson")

    # ---------------------------------------------------------------- 5. bonus flow path
    flow = None
    if a.upstream and dem is not None:
        from src.hydrology import flowpath
        path_gdf, along = flowpath.trace(dem, grid.transform, grid.crs, a.upstream, pts(sett_all) if len(sett_all) else sett_all,
                                         cfg["flowpath"]["buffer_m"], cfg["flowpath"]["max_dem_pixels"])
        save_geojson(path_gdf, out / "flowpath.geojson")
        save_geojson(along, out / "flowpath_settlements.geojson")
        flow = {"length_km": round(float(path_gdf.length_km.iloc[0]), 2), "buffer_m": cfg["flowpath"]["buffer_m"],
                "settlements": [str(n) for n in (along["name"] if len(along) and "name" in along else [])]}
    elif a.upstream:
        warnings.append("Flow path needs a DEM.")

    # ---------------------------------------------------------------- 6. summary
    vs = settle["status"] if len(settle) else []
    vc = settle["status_conservative"] if len(settle) else []
    summary = {
        "run_id": run_id, "event_date": a.date, "aoi": [round(float(v), 4) for v in a.aoi], "method": method,
        "s1_pair": s1_pair, "osm_snapshot": osm_date, "synthetic": bool(a.demo),
        "pixel_size_m": grid.pixel_size,
        "flood_area_km2": round(water_km2 + debris_km2, 2), "water_km2": round(water_km2, 2),
        "debris_km2": round(debris_km2, 2),
        "buildings": {"total": int(len(buildings)), "hit": _count(buildings, "likely_hit"),
                      "possible": _count(buildings, "possibly_hit")},
        "roads": {"total_km": round(float(roads.length.sum() / 1000), 1) if len(roads) else 0,
                  "hit_km": round(float(roads["flooded_m"].sum() / 1000), 2) if len(roads) else 0,
                  "hit_ways": _count(roads, "likely_hit")},
        "bridges": {"total": int(len(br)), "hit": _count(br, "likely_hit"), "possible": _count(br, "possibly_hit")},
        "settlements": {"total": int(len(settle)),
                        "cut_off": int((np.asarray(vs) == "cut_off").sum()),
                        "cut_off_conservative": int((np.asarray(vc) == "cut_off").sum()),
                        "connected": int((np.asarray(vs) == "connected").sum()),
                        "no_access_before": int((np.asarray(vs) == "no_road_access_before").sum()),
                        "unmapped": int((np.asarray(vs) == "unmapped_no_road_nearby").sum())},
        "network": ninfo, "s2": s2_info, "flowpath": flow, "warnings": warnings,
        "attribution": ATTRIBUTION, "disclaimer": DISCLAIMER, "runtime_s": round(time.time() - t0, 1),
    }
    dump_json(summary, out / "summary.json")

    # ---------------------------------------------------------------- 7. report
    if not a.no_report:
        from src.report import make_report
        bounds = tuple(gpd.GeoSeries([__import__("shapely.geometry", fromlist=["box"]).box(
            grid.transform.c, grid.transform.f + grid.transform.e * grid.height,
            grid.transform.c + grid.transform.a * grid.width, grid.transform.f)]).total_bounds)
        png = make_report.render_map_png(flood_gdf, roads, settle, bounds, hospitals)
        (out / "preview.png").write_bytes(png)
        make_report.make_report(summary, settle, png, out)
    log.info("Done in %.1fs -> %s", time.time() - t0, out)
    return out


def main(argv=None) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
