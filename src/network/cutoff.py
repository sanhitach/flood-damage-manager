"""Road-network connectivity: which settlements lose their road link to the nearest town / hospital?

Graph nodes = every OSM way vertex (shared vertices = junctions). Edges = way segments.
Scenarios:
  confirmed    - remove segments that intersect the detected flood extent (and bridges classed likely_hit)
  conservative - also remove segments within the 'near' buffer (and bridges classed possibly_hit)
Uses scipy.sparse.csgraph (fast; no networkx needed for large networks).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


def build_graph(roads: gpd.GeoDataFrame, union, buf):
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    geoms = roads.geometry.values
    coords, idx = shapely.get_coordinates(geoms, return_index=True)
    same = idx[:-1] == idx[1:]
    a, b, par = coords[:-1][same], coords[1:][same], idx[:-1][same]
    length = np.hypot(*(b - a).T)
    ok = length > 0.05
    a, b, par, length = a[ok], b[ok], par[ok], length[ok]

    keys = np.round(np.vstack([a, b]), 1)
    node_xy, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n_seg = len(a)
    u, v = inv[:n_seg], inv[n_seg:]
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    _, first = np.unique(lo.astype("int64") * len(node_xy) + hi, return_index=True)
    first = np.sort(first)
    a, b, par, length, u, v = a[first], b[first], par[first], length[first], u[first], v[first]

    seg = shapely.linestrings(np.stack([a, b], axis=1))
    if union is None:
        hit = np.zeros(len(seg), bool)
        near = np.zeros(len(seg), bool)
    else:
        hit = shapely.intersects(seg, union)
        near = shapely.intersects(seg, buf)
    bridge = roads["is_bridge"].values[par] if "is_bridge" in roads else np.zeros(len(par), bool)
    status = roads["status"].values[par] if "status" in roads else np.full(len(par), "unaffected")
    hit = hit | (bridge & (status == "likely_hit"))
    near = near | hit | (bridge & (status != "unaffected"))
    return {"node_xy": node_xy, "u": u, "v": v, "w": length, "hit": hit, "near": near, "n": len(node_xy)}


def _reach(g, keep, dest_nodes):
    n = g["n"]
    if len(dest_nodes) == 0:
        return np.zeros(n, bool), np.full(n, np.inf)
    graph = csr_matrix((g["w"][keep], (g["u"][keep], g["v"][keep])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    reach = np.isin(labels, np.unique(labels[dest_nodes]))
    dist = dijkstra(graph, directed=False, indices=np.unique(dest_nodes), min_only=True)
    return reach, dist


def _status(before_ok, after_ok):
    return np.where(~before_ok, "no_road_access_before", np.where(after_ok, "connected", "cut_off"))


def analyse(roads, union, buf, settlements, hospitals, towns, snap_m=1000.0, extra_dest_xy=None):
    """All inputs in a projected CRS (metres). settlements/hospitals/towns: point GeoDataFrames.
    Returns (settlements_with_status GeoDataFrame, info dict)."""
    info = {"warnings": []}
    g = build_graph(roads, union, buf)
    tree = cKDTree(g["node_xy"])

    def snap(gdf):
        if len(gdf) == 0:
            return np.array([], int), np.array([], float)
        xy = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
        d, i = tree.query(xy)
        return i, d

    h_node, h_d = snap(hospitals)
    t_node, t_d = snap(towns)
    hosp_nodes = h_node[h_d <= snap_m] if len(h_node) else np.array([], int)
    town_nodes = t_node[t_d <= snap_m] if len(t_node) else np.array([], int)
    extra = np.array([], int)
    if extra_dest_xy:
        _, extra = tree.query(np.array(extra_dest_xy))
        extra = np.atleast_1d(extra)
    dest_any = np.unique(np.concatenate([hosp_nodes, town_nodes, extra]).astype(int))
    if len(dest_any) == 0:
        raise RuntimeError("No hospital/town found in the AOI road network. Pass --destination LON LAT "
                           "(e.g. the district hospital) or enlarge the AOI.")
    if len(hosp_nodes) == 0:
        info["warnings"].append("No hospital mapped in OSM inside the AOI; 'hospital' column is empty.")

    all_keep = np.ones(len(g["u"]), bool)
    scen = {"before": all_keep, "confirmed": ~g["hit"], "conservative": ~g["near"]}
    res_any = {k: _reach(g, keep, dest_any) for k, keep in scen.items()}
    res_hosp = {k: _reach(g, keep, hosp_nodes) for k, keep in scen.items()} if len(hosp_nodes) else None

    s_node, s_d = snap(settlements)
    out = settlements.copy().reset_index(drop=True)
    snapped = s_d <= snap_m if len(s_d) else np.array([], bool)

    def pick(res, which):
        reach, dist = res[which]
        return reach[s_node], dist[s_node]

    if len(out):
        rb, db_ = pick(res_any, "before")
        rc, dc = pick(res_any, "confirmed")
        rk, dk = pick(res_any, "conservative")
        out["status"] = np.where(snapped, _status(rb, rc), "unmapped_no_road_nearby")
        out["status_conservative"] = np.where(snapped, _status(rb, rk), "unmapped_no_road_nearby")
        out["dist_before_km"] = np.where(snapped & rb, db_ / 1000, np.nan)
        out["dist_after_km"] = np.where(snapped & rc, dc / 1000, np.nan)
        out["detour_km"] = out["dist_after_km"] - out["dist_before_km"]
        if res_hosp:
            hb, _ = pick(res_hosp, "before")
            hc, _ = pick(res_hosp, "confirmed")
            out["hospital_status"] = np.where(snapped, _status(hb, hc), "unmapped_no_road_nearby")
        else:
            out["hospital_status"] = "no_hospital_in_aoi"
        out["snap_distance_m"] = s_d
    else:
        for c in ["status", "status_conservative", "hospital_status"]:
            out[c] = []
    info.update({
        "n_nodes": int(g["n"]), "n_segments": int(len(g["u"])),
        "n_segments_blocked_confirmed": int(g["hit"].sum()),
        "n_segments_blocked_conservative": int(g["near"].sum()),
        "n_destinations": int(len(dest_any)), "n_hospitals_snapped": int(len(hosp_nodes)),
    })
    return out, info
