"""Smoke test: the offline synthetic valley must reproduce the 'who is cut off' example by construction."""
import json

import geopandas as gpd

import run_pipeline


def test_demo_pipeline(tmp_path):
    out = run_pipeline.main(["--demo", "--out", str(tmp_path), "--no-report"])
    s = json.loads((out / "summary.json").read_text())
    assert s["flood_area_km2"] > 0.1
    assert s["bridges"]["hit"] == 1
    st = gpd.read_file(out / "settlements.geojson").set_index("name")["status"]
    assert st["Village B"] == "cut_off" and st["Village C"] == "cut_off"
    assert st["Village A"] == "connected" and st["Village D"] == "connected"


def test_osm_snapshot_must_precede_event():
    import datetime as dt
    import pytest
    from src.data import osm_ohsome
    with pytest.raises(ValueError):
        osm_ohsome.get_layers([0, 0, 1, 1], "2026-09-01", dt.date(2026, 8, 26))
