"""Wall and window recovery from double-line CAD linework (parallel-line sweep).

Why not just polygonize: walls in real drawings are open at every door/window, so the wall
band is rarely a closed region. Instead, for every wall direction we sweep along the axis:
on each stretch, the wall-layer lines covering it are grouped by offset. Two lines 6-45 cm
apart = wall; three or more inside one band = window (outer face, glass/frame lines, inner
face). Works for any wall angle. Curved walls (arcs) and odd closed shapes are covered by the
closed-region pass in extract.py and merged in.
"""
import math
from collections import defaultdict

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

ANGLE_TOL = math.radians(1.0)


def _direction(seg):
    (x0, y0), (x1, y1) = seg
    a = math.atan2(y1 - y0, x1 - x0) % math.pi
    return a


def _rect(u, n, t0, t1, d0, d1):
    return Polygon([(u[0] * t0 + n[0] * d0, u[1] * t0 + n[1] * d0), (u[0] * t1 + n[0] * d0, u[1] * t1 + n[1] * d0),
                    (u[0] * t1 + n[0] * d1, u[1] * t1 + n[1] * d1), (u[0] * t0 + n[0] * d1, u[1] * t0 + n[1] * d1)])


def sweep(segments, scale, min_thick=0.06, max_thick=0.45, min_len=0.05):
    """Return (wall_polys, window_polys) in drawing units."""
    min_t, max_t, min_l = min_thick / scale, max_thick / scale, min_len / scale
    # Cluster segments by direction.
    groups = []
    for seg in segments:
        if math.dist(*seg) < min_l:
            continue
        a = _direction(seg)
        for g in groups:
            da = abs(a - g["a"])
            if min(da, math.pi - da) < ANGLE_TOL:
                g["segs"].append(seg)
                break
        else:
            groups.append({"a": a, "segs": [seg]})
    walls, windows = [], []
    for g in groups:
        if len(g["segs"]) < 2:
            continue
        u = (math.cos(g["a"]), math.sin(g["a"]))
        n = (-u[1], u[0])
        items = []
        for (p, q) in g["segs"]:
            t0, t1 = sorted((p[0] * u[0] + p[1] * u[1], q[0] * u[0] + q[1] * u[1]))
            d = ((p[0] + q[0]) / 2) * n[0] + ((p[1] + q[1]) / 2) * n[1]
            items.append((t0, t1, d))
        # Lines far apart in offset never interact: bucket by offset to keep the sweep small.
        items.sort(key=lambda it: it[2])
        buckets, cur = [], [items[0]]
        for it in items[1:]:
            if it[2] - cur[-1][2] <= max_t:
                cur.append(it)
            else:
                buckets.append(cur)
                cur = [it]
        buckets.append(cur)
        for b in buckets:
            if len(b) < 2:
                continue
            cuts = sorted({v for t0, t1, _ in b for v in (t0, t1)})
            prev = None
            for ta, tb in zip(cuts, cuts[1:]):
                if tb - ta < 1e-6:
                    continue
                mid = (ta + tb) / 2
                offs = sorted({round(d, 1) for t0, t1, d in b if t0 <= mid <= t1})
                # Split the offsets into bands no thicker than a wall.
                bands, cur_band = [], [offs[0]] if offs else []
                for d in offs[1:]:
                    if d - cur_band[0] <= max_t:
                        cur_band.append(d)
                    else:
                        bands.append(cur_band)
                        cur_band = [d]
                if cur_band:
                    bands.append(cur_band)
                for band in bands:
                    if len(band) < 2 or band[-1] - band[0] < min_t:
                        continue
                    kind = "window" if len(band) >= 3 else "wall"
                    (windows if kind == "window" else walls).append(_rect(u, n, ta, tb, band[0], band[-1]))
    return walls, windows


def merge(polys, eps):
    if not polys:
        return []
    u = unary_union([p.buffer(eps, join_style=2) for p in polys]).buffer(-eps, join_style=2)
    return [g for g in (u.geoms if hasattr(u, "geoms") else [u]) if g.area > 0]
