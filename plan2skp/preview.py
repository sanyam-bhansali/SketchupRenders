"""Debug/preview image of an extracted plan: original linework + detected walls, windows,
rooms and door lintels. Also used by the web app to show what was recognised.

    python -m plan2skp.preview PLAN.dxf OUT.png --option N
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan2skp.extract import extract  # noqa: E402
from plan2skp.build import lintel_for_door  # noqa: E402


def _fill(ax, poly, **kw):
    for g in (poly.geoms if hasattr(poly, "geoms") else [poly]):
        x, y = g.exterior.xy
        ax.fill(x, y, **kw)
        for h in g.interiors:
            x, y = h.xy
            ax.fill(x, y, color="white")


def render(plan, out_png, title=""):
    fig, ax = plt.subplots(figsize=(12, 10))
    cols = plt.cm.Pastel1.colors + plt.cm.Pastel2.colors
    for i, r in enumerate(plan["rooms"]):
        _fill(ax, r["poly"], color=cols[i % len(cols)])
        c = r["poly"].representative_point()
        ax.text(c.x, c.y, r["name"] or "Room", fontsize=8, ha="center")
    for w in plan["walls"]:
        _fill(ax, w, color="#222")
    for w in plan.get("windows", []):
        _fill(ax, w, color="#3b8fd9")
    walls_u = unary_union(plan["walls"])
    for d in plan["doors"]:
        l = lintel_for_door(d, plan["walls"], walls_u)
        if l is not None:
            _fill(ax, l, color="#e0443e")
    minx, miny, maxx, maxy = unary_union(plan["walls"] + [r["poly"] for r in plan["rooms"]]).bounds
    pad = 0.05 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title + "   (black: walls, blue: windows, red: door lintels)", fontsize=10)
    plt.savefig(out_png, dpi=80, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dxf")
    ap.add_argument("out")
    ap.add_argument("--option", type=int, default=0)
    a = ap.parse_args()
    plans = extract(a.dxf)
    render(plans[a.option], a.out, f"Option {a.option}")
