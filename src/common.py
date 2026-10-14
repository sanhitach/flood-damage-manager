"""Shared helpers: config, raster grid I/O, GeoJSON export, attribution text."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import yaml
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform_bounds

log = logging.getLogger("floodmap")

ATTRIBUTION = [
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 \u00a9 DLR e.V. 2010\u20132014 and \u00a9 Airbus Defence and Space "
    "GmbH 2014\u20132018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "\u00a9 OpenStreetMap contributors.",
    "Training data: Bountos et al., 2024 (Kuro Siwo, NeurIPS Datasets & Benchmarks); "
    "Bonafilia et al., 2020 (Sen1Floods11, CVPR Workshops) - cite only those actually used.",
]

DISCLAIMER = "Educational prototype - not an operational tool. Estimates from satellite data need ground verification."


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class Grid:
    transform: Affine
    crs: CRS
    height: int
    width: int

    @property
    def pixel_size(self) -> float:
        return abs(self.transform.a)

    @property
    def shape(self):
        return (self.height, self.width)

    def bounds_wgs84(self):
        l, t = self.transform.c, self.transform.f
        r = l + self.transform.a * self.width
        b = t + self.transform.e * self.height
        return transform_bounds(self.crs, "EPSG:4326", l, b, r, t)

    @classmethod
    def from_raster(cls, path) -> "Grid":
        with rasterio.open(path) as ds:
            return cls(ds.transform, ds.crs, ds.height, ds.width)


def read_aligned(path, grid: Grid, resampling=Resampling.bilinear) -> np.ndarray:
    """Read every band of `path`, resampled onto `grid`. Missing data -> NaN. Shape (bands, H, W)."""
    with rasterio.open(path) as ds:
        out = np.full((ds.count, grid.height, grid.width), np.nan, dtype="float32")
        for i in range(ds.count):
            reproject(
                source=rasterio.band(ds, i + 1),
                destination=out[i],
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=resampling,
                src_nodata=ds.nodata,
                dst_nodata=np.nan,
            )
    return out


def write_raster(path, arr: np.ndarray, grid: Grid, nodata=None, dtype=None):
    arr = arr if arr.ndim == 3 else arr[None]
    dtype = dtype or arr.dtype
    with rasterio.open(
        path, "w", driver="GTiff", height=grid.height, width=grid.width, count=arr.shape[0],
        dtype=dtype, crs=grid.crs, transform=grid.transform, nodata=nodata, compress="deflate",
    ) as ds:
        ds.write(arr.astype(dtype))


def save_geojson(gdf, path):
    """Write a GeoDataFrame as GeoJSON in WGS84 (also handles empty frames)."""
    path = Path(path)
    if gdf is None or len(gdf) == 0:
        path.write_text('{"type":"FeatureCollection","features":[]}')
        return
    path.write_text(gdf.to_crs(4326).to_json())


def to_py(o):
    """json.dump default= hook for numpy types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def dump_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=to_py, ensure_ascii=False), encoding="utf-8")
