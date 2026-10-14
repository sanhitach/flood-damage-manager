"""Streamlit map dashboard.   streamlit run dashboard/app.py"""
import json
import subprocess
import sys
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
ATTR = [
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 \u00a9 DLR e.V. 2010\u20132014 and \u00a9 Airbus Defence and Space GmbH "
    "2014\u20132018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "\u00a9 OpenStreetMap contributors.",
]
STATUS_COLOR = {"cut_off": "red", "connected": "green", "no_road_access_before": "gray", "unmapped_no_road_nearby": "gray"}

st.set_page_config(page_title="Flood damage from space", layout="wide")
st.title("Mapping flood damage from space")
st.caption("Educational prototype - not an operational tool. Estimates from satellite data need ground verification.")


def load(path):
    p = Path(path)
    if not p.exists():
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)
    g = gpd.read_file(p)
    if g.crs is None:
        g = g.set_crs(4326)
    return g


# ------------------------------------------------------------------ sidebar: run or pick results
with st.sidebar:
    st.header("Run an analysis")
    demo = st.checkbox("Synthetic demo (offline)", value=False)
    aoi = st.text_input("AOI  minlon,minlat,maxlon,maxlat", "85.15,28.00,85.40,28.30", disabled=demo)
    date = st.text_input("Flood date (YYYY-MM-DD)", "2026-08-26", disabled=demo)
    method = st.selectbox("Method", ["auto", "baseline", "unet"])
    up = st.text_input("Upstream point lon,lat (optional)", "")
    dest = st.text_input("Extra destination lon,lat (optional)", "")
    use_s2 = st.checkbox("Add Sentinel-2 confirmation", value=False, disabled=demo)
    if st.button("Run pipeline", type="primary"):
        cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--method", method]
        if demo:
            cmd += ["--demo"]
        else:
            cmd += ["--aoi", *[v.strip() for v in aoi.split(",")], "--date", date]
            if use_s2:
                cmd += ["--use-s2"]
        if up.strip():
            cmd += ["--upstream", *[v.strip() for v in up.split(",")]]
        if dest.strip():
            cmd += ["--destination", *[v.strip() for v in dest.split(",")]]
        with st.spinner("Running (real data can take several minutes: openEO batch jobs)..."):
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            st.error("Pipeline failed")
            st.code((r.stderr or r.stdout)[-3000:])
        else:
            st.success("Done")
    runs = sorted([p for p in OUT.glob("*/summary.json")], key=lambda p: p.stat().st_mtime, reverse=True)
    names = [p.parent.name for p in runs]
    run = st.selectbox("Results", names) if names else None

if not run:
    st.info("No results yet. Tick 'Synthetic demo' and press Run, or run run_pipeline.py.")
    st.stop()

rd = OUT / run
S = json.loads((rd / "summary.json").read_text())
if S.get("synthetic"):
    st.warning("SYNTHETIC DEMO DATA - not a real event.")
for w in S.get("warnings", []):
    if "SYNTHETIC" not in w:
        st.warning(w)

c = st.columns(5)
c[0].metric("Flood/debris area", f"{S['flood_area_km2']} km\u00b2")
c[1].metric("Buildings hit / near", f"{S['buildings']['hit']} / {S['buildings']['possible']}")
c[2].metric("Road hit (km)", S["roads"]["hit_km"])
c[3].metric("Bridges hit", f"{S['bridges']['hit']} / {S['bridges']['total']}")
c[4].metric("Settlements cut off", f"{S['settlements']['cut_off']} / {S['settlements']['total']}")

flood, roads = load(rd / "flood_polygons.geojson"), load(rd / "roads.geojson")
bld, sett = load(rd / "buildings_affected.geojson"), load(rd / "settlements.geojson")
hosp, flow = load(rd / "hospitals.geojson"), load(rd / "flowpath.geojson")
flow_s = load(rd / "flowpath_settlements.geojson")

minx, miny, maxx, maxy = S["aoi"]
m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], tiles="OpenStreetMap", control_scale=True)
m.fit_bounds([[miny, minx], [maxy, maxx]])

if len(flood):
    folium.GeoJson(flood.to_json(), name="Flood / debris",
                   style_function=lambda f: {"color": "#3b82f6" if f["properties"]["class"] == "water" else "#a0522d",
                                             "weight": 0.5, "fillOpacity": 0.55}).add_to(m)
if len(roads):
    aff = roads[roads.status != "unaffected"]
    if len(aff):
        folium.GeoJson(aff.to_json(), name="Roads hit / possibly hit",
                       style_function=lambda f: {"color": "#dc2626" if f["properties"]["status"] == "likely_hit" else "#f59e0b",
                                                 "weight": 4},
                       tooltip=folium.GeoJsonTooltip(fields=[c for c in ["name", "highway", "status", "is_bridge"] if c in aff.columns])).add_to(m)
if len(bld):
    folium.GeoJson(bld.head(8000).to_json(), name="Buildings hit / near",
                   style_function=lambda f: {"color": "#dc2626" if f["properties"]["status"] == "likely_hit" else "#f59e0b",
                                             "weight": 1, "fillOpacity": 0.7}).add_to(m)
grp = folium.FeatureGroup(name="Settlements")
for _, r in sett.iterrows():
    folium.CircleMarker([r.geometry.y, r.geometry.x], radius=8, color="black", weight=1, fill=True,
                        fill_color=STATUS_COLOR.get(r["status"], "gray"), fill_opacity=0.95,
                        tooltip=f"{r.get('name', '')}: {r['status']}",
                        popup=f"hospital access: {r.get('hospital_status', '')}<br>road km before: {r.get('dist_before_km')}").add_to(grp)
grp.add_to(m)
for _, r in hosp.iterrows():
    folium.Marker([r.geometry.y, r.geometry.x], tooltip=f"Hospital: {r.get('name', '')}",
                  icon=folium.Icon(color="blue", icon="plus-sign")).add_to(m)
if len(flow):
    folium.GeoJson(flow.to_json(), name="Flood path (DEM)", style_function=lambda f: {"color": "#7c3aed", "weight": 3, "dashArray": "6"}).add_to(m)
folium.LayerControl().add_to(m)
st_folium(m, height=620, width=1200, returned_objects=[])
st.caption("Red dot = cut off from nearest town/hospital; green = still connected. Hit = overlaps detected flood extent (not proof of structural damage).")

t1, t2, t3, t4 = st.tabs(["Settlements", "Flood path", "Situation report", "Details"])
with t1:
    if len(sett):
        cols = [c for c in ["name", "status", "status_conservative", "hospital_status", "dist_before_km", "dist_after_km", "detour_km"] if c in sett.columns]
        df = pd.DataFrame(sett[cols])
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "settlements.csv")
with t2:
    if len(flow_s):
        st.write(f"Downstream trace: {S['flowpath']['length_km']} km")
        cols = [c for c in ["name", "km_along_path", "offset_m", "height_above_channel_m"] if c in flow_s.columns]
        st.dataframe(pd.DataFrame(flow_s[cols]))
    else:
        st.write("No flow-path result. Provide an upstream point in the sidebar.")
with t3:
    pdf = rd / "report.pdf"
    if pdf.exists():
        st.download_button("Download one-page PDF report", pdf.read_bytes(), "situation_report.pdf")
    html = rd / "report.html"
    if html.exists():
        st.download_button("Download report as HTML", html.read_bytes(), "situation_report.html")
    if (rd / "preview.png").exists():
        st.image(str(rd / "preview.png"), caption="Flood extent, road status, settlements")
with t4:
    st.json({k: v for k, v in S.items() if k not in ("attribution",)})

st.divider()
st.caption(" ".join(ATTR))
