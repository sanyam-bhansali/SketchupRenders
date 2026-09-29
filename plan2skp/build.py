"""DWG/DXF floor plan -> editable 3D SketchUp model.

    python -m plan2skp.build PLAN.dwg OUT.skp [--plan N] [--wall-height 3.0] [--list]

Walls are extruded to full height; door openings get a lintel above door height; each room
gets a named floor. Everything is grouped and tagged for editing in SketchUp.
"""
import argparse
import math
import os
import sys

from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan2skp.extract import extract  # noqa: E402
from plan2skp.dwg import to_dxf  # noqa: E402

WALL_RGB = (238, 236, 230)
LINTEL_RGB = (228, 226, 220)
GLASS_RGB = (170, 200, 215)
FLOOR_RGB = (205, 196, 182)


def _axis(poly):
    """(unit direction of the long side, thickness, centre) of a wall polygon."""
    mrr = poly.minimum_rotated_rectangle
    c = list(mrr.exterior.coords)[:4]
    e1 = (c[1][0] - c[0][0], c[1][1] - c[0][1])
    e2 = (c[2][0] - c[1][0], c[2][1] - c[1][1])
    l1, l2 = math.hypot(*e1), math.hypot(*e2)
    u, thick = ((e1[0] / l1, e1[1] / l1), l2) if l1 >= l2 else ((e2[0] / l2, e2[1] / l2), l1)
    return u, thick, mrr.centroid


def lintel_for_door(door, walls, walls_union):
    """Rectangle over the door opening, in plan units, aligned with the host wall."""
    h = Point(door["hinge"])
    r = door["radius"]
    host = min(walls, key=lambda w: w.distance(h))
    if host.distance(h) > r * 0.5:
        return None
    u, thick, centre = _axis(host)
    n = (-u[1], u[0])
    # Move the hinge onto the wall centre line.
    off = (h.x - centre.x) * n[0] + (h.y - centre.y) * n[1]
    p0 = (h.x - off * n[0], h.y - off * n[1])
    grown = walls_union.buffer(thick * 0.1)
    best = None
    for s in (1, -1):
        mid = Point(p0[0] + s * u[0] * r * 0.5, p0[1] + s * u[1] * r * 0.5)
        if not grown.contains(mid):          # the opening side has no wall
            best = s
            break
    if best is None:
        return None
    p1 = (p0[0] + best * u[0] * r, p0[1] + best * u[1] * r)
    t = thick / 2
    return Polygon([(p0[0] + n[0] * t, p0[1] + n[1] * t), (p1[0] + n[0] * t, p1[1] + n[1] * t),
                    (p1[0] - n[0] * t, p1[1] - n[1] * t), (p0[0] - n[0] * t, p0[1] - n[1] * t)])


def _glass_strip(poly):
    """Thin pane (8 mm) down the middle of a window band."""
    u, thick, c = _axis(poly)
    n = (-u[1], u[0])
    half_len = max(poly.minimum_rotated_rectangle.length / 2 - thick, 0) / 2
    if half_len <= 0:
        return None
    t = 0.004
    return Polygon([(c.x + u[0] * half_len + n[0] * t, c.y + u[1] * half_len + n[1] * t),
                    (c.x - u[0] * half_len + n[0] * t, c.y - u[1] * half_len + n[1] * t),
                    (c.x - u[0] * half_len - n[0] * t, c.y - u[1] * half_len - n[1] * t),
                    (c.x + u[0] * half_len - n[0] * t, c.y + u[1] * half_len - n[1] * t)])


def build(plan, out_path, wall_height=3.0, door_height=2.1, sill_height=0.9):
    from plan2skp.skp_writer import SkpWriter
    s = plan["scale"]
    ox, oy = plan["origin"]

    def to_m(g):
        return affinity.affine_transform(g, [s, 0, 0, s, -ox * s, -oy * s])

    walls_union = unary_union(plan["walls"])
    lintels = [l for l in (lintel_for_door(d, plan["walls"], walls_union) for d in plan["doors"]) if l is not None]
    w = SkpWriter()
    ents = w.group("Walls", "Walls", w.material("Wall", WALL_RGB))
    w.extrude(ents, [to_m(p) for p in plan["walls"]], 0.0, wall_height)
    if lintels:
        ents = w.group("Door lintels", "Lintels", w.material("Wall", WALL_RGB))
        w.extrude(ents, [to_m(p) for p in lintels], door_height, wall_height)
    if plan.get("windows"):
        wins = [to_m(p) for p in plan["windows"]]
        ents = w.group("Window sills", "Windows", w.material("Wall", WALL_RGB))
        w.extrude(ents, wins, 0.0, sill_height)
        ents = w.group("Window heads", "Windows", w.material("Wall", WALL_RGB))
        w.extrude(ents, wins, door_height, wall_height)
        glass = [g.buffer(0) for g in (_glass_strip(p) for p in wins) if g is not None]
        ents = w.group("Window glass", "Windows", w.material("Glass", GLASS_RGB, 0.25))
        w.extrude(ents, glass, sill_height, door_height)
    floor_mat = w.material("Floor", FLOOR_RGB)
    counts = {}
    for r in plan["rooms"]:
        name = r["name"] or "Room"
        counts[name] = counts.get(name, 0) + 1
        label = name if counts[name] == 1 else f"{name} {counts[name]}"
        ents = w.group(f"Floor - {label}", "Floors", floor_mat)
        w.flat(ents, [to_m(r["poly"])], 0.0)
    w.save(out_path)
    return {"walls": len(plan["walls"]), "lintels": len(lintels), "rooms": len(plan["rooms"]),
            "size_m": plan["size_m"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drawing", help=".dwg or .dxf")
    ap.add_argument("out", nargs="?", help="output .skp")
    ap.add_argument("--option", type=int, default=None, help="which layout to build (see --list); default all")
    ap.add_argument("--wall-height", type=float, default=3.0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    dxf = to_dxf(a.drawing) if a.drawing.lower().endswith(".dwg") else a.drawing
    plans = extract(dxf)
    for i, p in enumerate(plans):
        names = sorted({r["name"] for r in p["rooms"] if r["name"]})
        print(f"[{i}] {p['size_m'][0]:.1f} x {p['size_m'][1]:.1f} m, {len(p['rooms'])} rooms, "
              f"{len(p['doors'])} doors: {', '.join(names[:8])}")
    if a.list:
        return
    base = a.out or os.path.splitext(a.drawing)[0] + ".skp"
    idxs = [a.option] if a.option is not None else list(range(len(plans)))
    for i in idxs:
        path = base if len(idxs) == 1 else base.replace(".skp", f"_option{i + 1}.skp")
        info = build(plans[i], path, a.wall_height)
        print(f"-> {path}: {info}")


if __name__ == "__main__":
    main()
