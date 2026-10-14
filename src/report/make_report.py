"""One-page situation report (HTML + PDF) and preview map. Every number comes from summary.json (computed by
the pipeline); text is template-based, never free-generated, so nothing can be invented."""
from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.common import ATTRIBUTION, DISCLAIMER, log

HERE = Path(__file__).parent


def build_text(s: dict, lang: str) -> str:
    st, b, r, br = s["settlements"], s["buildings"], s["roads"], s["bridges"]
    if lang == "en":
        t = (f"Satellite change detection ({s['method']}) indicates about {s['flood_area_km2']} km\u00b2 affected by "
             f"flooding or debris. {b['hit']} buildings overlap the detected extent and {b['possible']} lie close to it. "
             f"About {r['hit_km']} km of road and {br['hit']} bridge(s) overlap the extent. ")
        if st["cut_off"]:
            t += (f"{st['cut_off']} of {st['total']} mapped settlements appear to have lost their road link to the "
                  f"nearest town or hospital ({st['cut_off_conservative']} under conservative assumptions). ")
        else:
            t += "No mapped settlement was found to lose its road connection. "
        return t + "These are estimates from satellite data and need ground verification."
    return (f"उपग्रह तस्बिरको परिवर्तन विश्लेषणले लगभग {s['flood_area_km2']} वर्ग कि.मी. क्षेत्र बाढी वा मलबाले प्रभावित "
            f"भएको देखाउँछ। {b['hit']} भवन प्रभावित क्षेत्रसँग मिल्छन् र {b['possible']} भवन नजिकै छन्। "
            f"लगभग {r['hit_km']} कि.मी. सडक र {br['hit']} पुल प्रभावित क्षेत्रमा परेका छन्। "
            f"नक्सांकित {st['total']} बस्तीमध्ये {st['cut_off']} बस्तीको नजिकको सहर वा अस्पतालसँगको सडक सम्पर्क "
            f"टुटेको देखिन्छ (सावधानीमूलक अनुमानमा {st['cut_off_conservative']})। "
            f"यी उपग्रह डाटामा आधारित अनुमान हुन्; भूमिमा पुष्टि आवश्यक छ।")


def render_map_png(flood_gdf, roads, settlements, aoi_bounds=None, hospitals=None) -> bytes:
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    if flood_gdf is not None and len(flood_gdf):
        for cls, color in (("water", "#3b82f6"), ("debris", "#a0522d")):
            sub = flood_gdf[flood_gdf["class"] == cls]
            if len(sub):
                sub.plot(ax=ax, color=color, linewidth=0)
    if roads is not None and len(roads):
        roads[roads.status == "unaffected"].plot(ax=ax, color="#999999", linewidth=0.5)
        roads[roads.status == "possibly_hit"].plot(ax=ax, color="#f59e0b", linewidth=1.2)
        roads[roads.status == "likely_hit"].plot(ax=ax, color="#dc2626", linewidth=1.6)
    if settlements is not None and len(settlements):
        for status, color in (("connected", "#16a34a"), ("cut_off", "#dc2626")):
            sub = settlements[settlements.status == status]
            if len(sub):
                sub.plot(ax=ax, color=color, markersize=22, edgecolor="black", linewidth=0.4)
    if hospitals is not None and len(hospitals):
        hospitals.plot(ax=ax, color="white", marker="P", markersize=40, edgecolor="black")
    if aoi_bounds:
        ax.set_xlim(aoi_bounds[0], aoi_bounds[2])
        ax.set_ylim(aoi_bounds[1], aoi_bounds[3])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5)
    ax.set_title("Flood extent, road status, settlements", fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def make_report(summary: dict, settlements_df, map_png: bytes, out_dir) -> dict:
    out_dir = Path(out_dir)
    env = Environment(loader=FileSystemLoader(HERE), autoescape=select_autoescape(["html"]))
    cut = []
    if settlements_df is not None and len(settlements_df):
        for _, r in settlements_df[settlements_df.status_conservative == "cut_off"].iterrows():
            d = r.get("dist_before_km")
            cut.append({"name": r.get("name") or "unnamed", "status_conservative": r["status_conservative"],
                        "hospital_status": r.get("hospital_status", ""),
                        "dist": "" if d is None or (isinstance(d, float) and np.isnan(d)) else f"{d:.1f} km"})
    s = dict(summary)
    s["s1_pair_text"] = (f"{s['s1_pair'].get('before_datetime','?')[:10]} \u2192 {s['s1_pair'].get('after_datetime','?')[:10]}"
                         if s.get("s1_pair") else "user-supplied rasters")
    html = env.get_template("template.html").render(
        s=s, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), disclaimer=DISCLAIMER,
        text_en=build_text(summary, "en"), text_ne=build_text(summary, "ne"), cut=cut,
        attribution=ATTRIBUTION, map_b64=base64.b64encode(map_png).decode() if map_png else "")
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    paths = {"html": str(out_dir / "report.html")}
    try:
        from weasyprint import HTML
        HTML(string=html, base_url=str(out_dir)).write_pdf(str(out_dir / "report.pdf"))
        paths["pdf"] = str(out_dir / "report.pdf")
    except Exception as e:  # weasyprint missing or pango not installed
        log.warning("PDF export skipped (%s). Open report.html and use Print -> Save as PDF.", e)
    return paths
