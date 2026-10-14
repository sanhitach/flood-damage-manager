"""Synthetic valley used for offline demos and tests (NOT real data - always label it as synthetic).

Creates before/after SAR (linear VV,VH), a DEM and fake 'OSM' layers (buildings, roads with a bridge, hospital,
villages, town). A debris-flood corridor crosses the main road, so some villages become cut off by construction.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS
from scipy import ndimage
from shapely.geometry import LineString, Point, box

from src.common import Grid, dump_json, write_raster


def make_demo(out_dir, seed: int = 7) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    osm = out / "osm"
    osm.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    H, W, pix = 320, 420, 20.0
    crs = CRS.from_epsg(32645)
    x0, y0 = Transformer.from_crs(4326, crs, always_xy=True).transform(85.28, 28.20)
    transform = Affine(pix, 0, x0, 0, -pix, y0)
    grid = Grid(transform, crs, H, W)
    yy, xx = np.mgrid[0:H, 0:W].astype("float64")
    river_y = 160 + 18 * np.sin(xx / 45.0)
    dist = np.abs(yy - river_y)

    dem = 3800 - 1.5 * xx + 4.0 * np.minimum(dist, 60) + ndimage.gaussian_filter(rng.normal(size=(H, W)), 8) * 25
    corridor = (xx >= 130) & (dist <= 5)

    land = -10 + ndimage.gaussian_filter(rng.normal(size=(H, W)), 2) * 1.5
    before_db = np.where(dist < 1.2, -22.0, land)
    after_db = before_db.copy()
    pond = ndimage.gaussian_filter(rng.normal(size=(H, W)), 6) > 0.0
    pond |= (xx >= 262) & (xx <= 300)   # ponded water where the road/bridge crosses the river
    after_db = np.where(corridor & pond, -22.0, after_db)
    after_db = np.where(corridor & ~pond, before_db + 5.0, after_db)

    def to_lin(db_vv):
        vv = 10 ** (db_vv / 10) * rng.gamma(4, 1 / 4, size=(H, W))
        vh = 10 ** ((db_vv - 7) / 10) * rng.gamma(4, 1 / 4, size=(H, W))
        return np.stack([vv, vh]).astype("float32")

    before, after = to_lin(before_db), to_lin(after_db)
    write_raster(out / "before.tif", before, grid, nodata=np.nan)
    write_raster(out / "after.tif", after, grid, nodata=np.nan)
    write_raster(out / "dem.tif", dem.astype("float32"), grid, nodata=np.nan)

    def px2xy(col, row):
        return x0 + (col + 0.5) * pix, y0 - (row + 0.5) * pix

    def road_y(x):
        return 160 + 18 * np.sin(x / 45.0) + 60 * np.sin(x / 90.0)

    def way(xs):
        return LineString([px2xy(x, road_y(x)) for x in xs])

    xs = np.arange(5, 411, 3.0)
    west, bridge, east = xs[xs <= 275], xs[(xs >= 275) & (xs <= 293)], xs[xs >= 293]
    spur = LineString([px2xy(149, road_y(149)), px2xy(149, road_y(149) - 35)])
    spur_c = LineString([px2xy(380, road_y(380)), px2xy(380, road_y(380) + 25)])
    roads = gpd.GeoDataFrame(
        {"highway": ["secondary", "secondary", "secondary", "track", "track"],
         "bridge": [None, "yes", None, None, None],
         "name": ["Highway W", "Bhote Koshi Bridge", "Highway E", "Spur E1", "Spur C"]},
        geometry=[way(west), way(bridge), way(east), spur, spur_c], crs=crs)

    def pt(col, rowoff, xcol=None):
        x = col if xcol is None else xcol
        return Point(px2xy(x, road_y(col) + rowoff))

    settlements = gpd.GeoDataFrame(
        {"name": ["Village A", "Village D", "Village E", "Village B", "Village C"],
         "place": ["village"] * 5},
        geometry=[pt(70, 8), pt(200, -6), Point(px2xy(149, road_y(149) - 35)), pt(330, 8), Point(px2xy(380, road_y(380) + 25))],
        crs=crs)
    town = gpd.GeoDataFrame({"name": ["Demo Town"], "place": ["town"]}, geometry=[pt(14, 5)], crs=crs)
    settlements_all = gpd.GeoDataFrame(
        gpd.pd.concat([settlements, town], ignore_index=True), geometry="geometry", crs=crs)
    hospitals = gpd.GeoDataFrame({"name": ["District Hospital"], "amenity": ["hospital"]},
                                 geometry=[pt(8, -6)], crs=crs)

    bx = rng.uniform(5, 410, 300)
    by = river_y[0, (bx.astype(int))] + rng.normal(0, 22, 300)
    bgeoms = [box(*(px2xy(x, y)[0] - 7, px2xy(x, y)[1] - 7, px2xy(x, y)[0] + 7, px2xy(x, y)[1] + 7))
              for x, y in zip(bx, np.clip(by, 5, H - 5))]
    buildings = gpd.GeoDataFrame({"building": ["yes"] * len(bgeoms)}, geometry=bgeoms, crs=crs)

    for name, gdf in [("roads", roads), ("settlements", settlements_all), ("hospitals", hospitals),
                      ("buildings", buildings)]:
        gdf.to_crs(4326).to_file(osm / f"{name}.geojson", driver="GeoJSON")

    sx, sy = px2xy(45, 160 + 18 * np.sin(45 / 45.0) - 2)
    up = Transformer.from_crs(crs, 4326, always_xy=True).transform(sx, sy)
    meta = {"before": str(out / "before.tif"), "after": str(out / "after.tif"), "dem": str(out / "dem.tif"),
            "osm_dir": str(osm), "upstream_lonlat": [up[0], up[1]], "event_date": "2026-08-26",
            "aoi": list(grid.bounds_wgs84()), "synthetic": True}
    dump_json(meta, out / "meta.json")
    return meta
