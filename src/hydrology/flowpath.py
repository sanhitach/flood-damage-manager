"""BONUS: trace the flood path downstream from an upstream point using the DEM, list settlements along it.

Method: priority-flood depression filling (with epsilon gradient so flats drain), then steepest-descent (D8) walk.
Pure numpy/heapq - no extra GIS dependency. For big DEMs the DEM is block-averaged first (max_pixels)."""
from __future__ import annotations

import heapq

import geopandas as gpd
import numpy as np
from affine import Affine
from pyproj import Transformer
from shapely.geometry import LineString, Point

_NBR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_depressions(dem: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    H, W = dem.shape
    d = dem.astype("float64").ravel().tolist()
    f = list(d)
    closed = bytearray(H * W)
    heap = []
    for i in range(H):
        for j in (0, W - 1):
            k = i * W + j
            if not closed[k]:
                closed[k] = 1
                heap.append((d[k], i, j))
    for j in range(W):
        for i in (0, H - 1):
            k = i * W + j
            if not closed[k]:
                closed[k] = 1
                heap.append((d[k], i, j))
    heapq.heapify(heap)
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        z, i, j = pop(heap)
        for di, dj in _NBR:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                k = ni * W + nj
                if not closed[k]:
                    closed[k] = 1
                    nz = d[k]
                    if nz <= z:
                        nz = z + eps
                    f[k] = nz
                    push(heap, (nz, ni, nj))
    return np.array(f).reshape(H, W)


def _downsample(dem, transform, max_pixels):
    H, W = dem.shape
    if H * W <= max_pixels:
        return dem, transform
    k = int(np.ceil(np.sqrt(H * W / max_pixels)))
    h2, w2 = H // k, W // k
    small = dem[: h2 * k, : w2 * k].reshape(h2, k, w2, k).mean(axis=(1, 3))
    return small, transform * Affine.scale(k, k)


def trace(dem, transform, crs, lonlat, settlements_utm, buffer_m=500.0, max_pixels=3_000_000):
    """Returns (path GeoDataFrame [1 row], settlements-along-path GeoDataFrame)."""
    dem = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    dem, transform = _downsample(dem, transform, max_pixels)
    H, W = dem.shape
    filled = fill_depressions(dem)

    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(*lonlat)
    col, row = ~transform * (x, y)
    r, c = int(row), int(col)
    if not (0 <= r < H and 0 <= c < W):
        raise ValueError("Upstream point is outside the DEM/AOI.")
    r0, r1, c0, c1 = max(r - 3, 0), min(r + 4, H), max(c - 3, 0), min(c + 4, W)
    win = dem[r0:r1, c0:c1]
    dr, dc = np.unravel_index(np.argmin(win), win.shape)
    cur = (r0 + dr, c0 + dc)

    path = [cur]
    while True:
        i, j = cur
        best, bz = None, filled[i, j]
        for di, dj in _NBR:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and filled[ni, nj] < bz:
                bz, best = filled[ni, nj], (ni, nj)
        if best is None:
            break
        cur = best
        path.append(cur)
        if cur[0] in (0, H - 1) or cur[1] in (0, W - 1):
            break

    xy = np.array([transform * (cj + 0.5, ci + 0.5) for ci, cj in path])
    elev = np.array([dem[ci, cj] for ci, cj in path])
    line = LineString(xy) if len(xy) > 1 else LineString([xy[0], xy[0] + 1])
    path_gdf = gpd.GeoDataFrame({"length_km": [line.length / 1000], "start_elev_m": [float(elev[0])],
                                 "end_elev_m": [float(elev[-1])]}, geometry=[line], crs=crs)

    rows = []
    if len(settlements_utm):
        dist = settlements_utm.geometry.distance(line)
        near = settlements_utm[dist <= buffer_m].copy()
        for idx, srow in near.iterrows():
            p = srow.geometry
            along = line.project(p)
            k = int(np.argmin(np.hypot(xy[:, 0] - p.x, xy[:, 1] - p.y)))
            sc, sr = ~transform * (p.x, p.y)
            sr, sc = int(np.clip(sr, 0, H - 1)), int(np.clip(sc, 0, W - 1))
            rows.append({**srow.drop("geometry").to_dict(), "geometry": p,
                         "km_along_path": along / 1000, "offset_m": float(dist[idx]),
                         "height_above_channel_m": float(dem[sr, sc] - elev[k])})
    along_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else \
        gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    if len(along_gdf):
        along_gdf = along_gdf.sort_values("km_along_path").reset_index(drop=True)
    return path_gdf, along_gdf
