"""Floor-plan extraction from DXF (DWG converted with dwg.py).

Pipeline per drawing:
1. Pick wall linework: layers whose name looks like a wall layer and that hold real
   linework (in some offices the "A-WALL" layer only holds door blocks).
2. Split the file into separate plans (offices keep several layout options side by side
   in one model space): connected groups of wall geometry.
3. Node all wall lines and polygonize them into closed regions. Classify each region by
   its mean thickness (2 * area / perimeter): thin regions are walls/columns, thin
   stacked strips inside a wall band are windows, big regions are rooms.
4. Doors: block references containing a swing arc (radius 0.5-1.3 m) -> openings.
5. Rooms get names from text labels that fall inside them.

All output coordinates are metres, shifted so each plan starts near the origin.
"""
import math
import re
from collections import defaultdict

import ezdxf
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union

from plan2skp.walls import merge, sweep

WALL_LAYER = re.compile(r"wall|w-a|a-wal|masonry|brick|partition|column|col\b", re.I)
ROOM_WORDS = re.compile(r"room|bed|kitchen|living|dining|toilet|bath|wc|balcony|passage|lobby|foyer|"
                        r"study|pooja|puja|mandir|store|utility|dress|terrace|hall|drawing|guest|master|"
                        r"kid|family|servant|deck|verandah|entrance", re.I)

UNIT_TO_M = {0: 0.001, 1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0}


def _segments(entity):
    """Line segments (2D) from LINE / LWPOLYLINE / POLYLINE / ARC entities, blocks exploded."""
    t = entity.dxftype()
    if t == "INSERT":
        try:
            for v in entity.virtual_entities():
                yield from _segments(v)
        except Exception:
            return
        return
    try:
        if t == "LINE":
            yield (entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)
        elif t in ("LWPOLYLINE", "POLYLINE"):
            pts = [(p[0], p[1]) for p in (entity.get_points("xy") if t == "LWPOLYLINE" else
                                          [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices])]
            if entity.closed and pts:
                pts.append(pts[0])
            for a, b in zip(pts, pts[1:]):
                yield a, b
        elif t == "ARC":
            pts = [(p.x, p.y) for p in entity.flattening(0.01 * max(entity.dxf.radius, 1))]
            for a, b in zip(pts, pts[1:]):
                yield a, b
    except Exception:
        return


def wall_layers(msp):
    """Layers that hold wall linework (by name, and actually containing lines)."""
    stats = defaultdict(lambda: [0, 0])      # layer -> [line-ish entities, inserts]
    for e in msp:
        k = stats[e.dxf.layer]
        if e.dxftype() in ("LINE", "LWPOLYLINE", "POLYLINE", "ARC"):
            k[0] += 1
        elif e.dxftype() == "INSERT":
            k[1] += 1
    named = [l for l, (n, _) in stats.items() if WALL_LAYER.search(l) and n >= 20]
    return named or [max(stats, key=lambda l: stats[l][0])]


def door_openings(msp, scale):
    """Door blocks: a block reference whose geometry includes a swing arc."""
    doors = []
    for e in msp.query("INSERT"):
        try:
            ents = list(e.virtual_entities())
        except Exception:
            continue
        arcs = [v for v in ents if v.dxftype() == "ARC" and 0.45 <= v.dxf.radius * scale <= 1.4]
        if not arcs:
            continue
        a = max(arcs, key=lambda v: v.dxf.radius)
        c, r = (a.dxf.center.x, a.dxf.center.y), a.dxf.radius
        # Block extents can include far-away attributes; the swing arc defines the door.
        lines = [LineString(seg) for v in ents if v.dxftype() in ("LINE", "LWPOLYLINE", "ARC")
                 for seg in _segments(v) if math.dist(*seg) > 0 and
                 all(math.dist(q, c) <= r * 1.2 for q in seg)]
        doors.append({"bbox": (c[0] - r, c[1] - r, c[0] + r, c[1] + r), "hinge": c, "radius": r,
                      "width": r * scale, "block": e.dxf.name, "lines": lines})
    return doors


def labels(msp):
    out = []
    for e in msp.query("MTEXT TEXT"):
        try:
            text = e.plain_text() if e.dxftype() == "MTEXT" else e.dxf.text
            p = e.dxf.insert
        except Exception:
            continue
        text = " ".join(text.split())
        if text and ROOM_WORDS.search(text) and len(text) < 40:
            out.append((text.title(), (p.x, p.y)))
    return out


def split_plans(lines, gap):
    """Group wall lines into separate drawings (connected within `gap` drawing units)."""
    blobs = unary_union([ln.buffer(gap, resolution=2) for ln in lines])
    geoms = list(blobs.geoms) if hasattr(blobs, "geoms") else [blobs]
    groups = []
    for g in geoms:
        members = [ln for ln in lines if g.intersects(ln)]
        groups.append((g, members))
    return groups


def classify_regions(members, scale, closers=()):
    """Polygonize the wall linework; split regions into walls, windows and rooms (metres).

    `closers` (door leaves/swings) close doorways so rooms don't merge through them."""
    noded = unary_union(list(members) + list(closers))
    regions = [p for p in polygonize(noded) if p.area > 0]
    walls, rooms, thin = [], [], []
    for p in regions:
        area = p.area * scale * scale
        per = p.length * scale
        thick = 2 * area / per if per else 0
        if area < 0.002:
            continue
        if thick < 0.05:
            thin.append(p)              # glazing strips / hairline slivers
        elif thick < 0.32:
            walls.append(p)             # masonry walls, partitions, columns
        elif area > 1.2:
            rooms.append(p)
    return walls, thin, rooms


def _parts(g):
    return list(g.geoms) if hasattr(g, "geoms") else ([g] if not g.is_empty else [])


def fill_junctions(walls, scale, reach=0.13, max_area=0.12):
    """Corners/T-junctions: the square where two walls meet has no parallel line pair in
    either direction, leaving a hole. Close small gaps without bridging doorways."""
    if not walls:
        return walls
    u = unary_union(walls)
    r = reach / scale
    closed = u.buffer(r, join_style=2).buffer(-r, join_style=2)
    extra = [g for g in _parts(closed.difference(u)) if g.area * scale * scale <= max_area]
    return _parts(unary_union([u] + extra))


def find_rooms(solids, closers, scale, bridge=0.9, min_area=1.2):
    """Rooms = enclosed spaces between walls/windows, with doorways (<~0.9 m) bridged."""
    if not solids:
        return []
    g = bridge / 2 / scale
    blocked = unary_union(list(solids) + [ln.buffer(0.02 / scale) for ln in closers])
    closed = blocked.buffer(g, join_style=2).buffer(-g, join_style=2)
    rooms = []
    for part in _parts(closed):
        for hole in part.interiors:
            r = Polygon(hole).difference(blocked)
            for piece in _parts(r):
                if piece.area * scale * scale >= min_area:
                    rooms.append(piece)
    return rooms


def extract(dxf_path, min_plan_area_m2=25.0):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    scale = UNIT_TO_M.get(doc.header.get("$INSUNITS", 4), 0.001)
    layers = wall_layers(msp)
    lines = []
    for e in msp:
        if e.dxf.layer not in layers:
            continue
        for a, b in _segments(e):
            if math.dist(a, b) * scale > 0.005:
                lines.append(LineString([a, b]))
    doors = door_openings(msp, scale)
    texts = labels(msp)
    plans = []
    for outline, members in split_plans(lines, gap=1.5 / scale):
        plan_doors = [d for d in doors if outline.contains(Point(d["hinge"]))]
        closers = [ln for d in plan_doors for ln in d["lines"]]
        region_walls, thin, _ = classify_regions(members, scale, closers)
        # Main pass: parallel-line sweep (handles walls left open at doors/windows).
        segs = [tuple(ln.coords) for ln in members if len(ln.coords) == 2]
        sweep_walls, windows = sweep(segs, scale)
        windows = merge(windows, 0.005 / scale)
        walls = merge(sweep_walls + region_walls, 0.005 / scale)
        walls = fill_junctions(walls, scale)
        if windows:
            win_u = unary_union(windows)
            walls = [g for w in walls for g in _parts(w.difference(win_u)) if g.area * scale * scale > 0.003]
        rooms = find_rooms(walls + windows, closers, scale)
        footprint = unary_union(walls + windows + rooms)
        if footprint.area * scale * scale < min_plan_area_m2:
            continue
        minx, miny, maxx, maxy = footprint.bounds
        room_items = []
        for r in rooms:
            name = next((t for t, (x, y) in texts if r.contains(Point(x, y))), None)
            room_items.append({"poly": r, "name": name})
        plans.append({"origin": (minx, miny), "size_m": ((maxx - minx) * scale, (maxy - miny) * scale),
                      "walls": walls, "windows": windows, "thin": thin, "rooms": room_items, "doors": plan_doors,
                      "scale": scale, "layers": layers})
    plans.sort(key=lambda p: -p["size_m"][0] * p["size_m"][1])
    return plans
