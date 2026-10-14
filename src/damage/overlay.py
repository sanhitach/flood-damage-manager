"""Flood raster -> polygons, then intersect with pre-event OSM buildings / roads / bridges."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely
from rasterio import features
from shapely.geometry import shape

CLASS_NAMES = {1: "water", 2: "debris"}


def mask_to_gdf(mask: np.ndarray, grid, min_area_m2: float = 0.0) -> gpd.GeoDataFrame:
    geoms, cls = [], []
    for geom, val in features.shapes(mask.astype("uint8"), mask=mask > 0, transform=grid.transform):
        geoms.append(shape(geom))
        cls.append(int(val))
    gdf = gpd.GeoDataFrame({"cls": cls}, geometry=geoms, crs=grid.crs)
    if len(gdf):
        gdf = gdf[gdf.area >= min_area_m2].copy()
    gdf["class"] = gdf["cls"].map(CLASS_NAMES)
    gdf["area_km2"] = gdf.area / 1e6
    return gdf.reset_index(drop=True)


def flood_geometries(flood_gdf: gpd.GeoDataFrame, buffer_m: float):
    """Prepared union of the flood extent and of its buffered version (None if no flood)."""
    if flood_gdf is None or len(flood_gdf) == 0:
        return None, None
    union = shapely.union_all(flood_gdf.geometry.values)
    buf = union.buffer(buffer_m)
    shapely.prepare(union)
    shapely.prepare(buf)
    return union, buf


def classify(gdf: gpd.GeoDataFrame, union, buf) -> gpd.GeoDataFrame:
    """Adds `status` = likely_hit (intersects flood) | possibly_hit (within buffer) | unaffected."""
    gdf = gdf.copy()
    if len(gdf) == 0:
        gdf["status"] = []
        return gdf
    if union is None:
        gdf["status"] = "unaffected"
        return gdf
    g = gdf.geometry.values
    hit = shapely.intersects(g, union)
    near = shapely.intersects(g, buf) & ~hit
    gdf["status"] = np.where(hit, "likely_hit", np.where(near, "possibly_hit", "unaffected"))
    return gdf


def is_bridge(roads: gpd.GeoDataFrame) -> np.ndarray:
    if "bridge" not in roads.columns:
        return np.zeros(len(roads), dtype=bool)
    return roads["bridge"].fillna("no").astype(str).str.lower().isin(["yes", "viaduct", "aqueduct", "movable", "cantilever", "true"]).values


def road_flooded_length(roads: gpd.GeoDataFrame, union) -> np.ndarray:
    if union is None or len(roads) == 0:
        return np.zeros(len(roads))
    g = roads.geometry.values
    return shapely.length(shapely.intersection(g, union))


def analyse_damage(layers_utm: dict, flood_gdf, buffer_m: float) -> dict:
    union, buf = flood_geometries(flood_gdf, buffer_m)
    buildings = classify(layers_utm["buildings"], union, buf)
    roads = classify(layers_utm["roads"], union, buf)
    roads["is_bridge"] = is_bridge(roads)
    roads["flooded_m"] = road_flooded_length(roads, union)
    return {"buildings": buildings, "roads": roads, "union": union, "buffer": buf}
