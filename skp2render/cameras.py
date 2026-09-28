"""Automatic interior cameras for models without (or in addition to) SketchUp scenes.

Pure numpy, so it runs in Blender or plain Python.

1. Find the main floor level (largest low horizontal surface area).
2. Slice the model with horizontal planes: at eye height (walls, tall furniture) and just
   above door heads (closes doorways so rooms separate). Rasterise the cuts to a grid.
3. Flood-fill free space. Components that touch the border are outside; the rest are rooms.
4. For each room, score candidate positions near the walls and view directions by visible
   floor area, visible furniture and depth, and keep the best 1-2 views.
5. Name rooms from the furniture inside them (bed -> Bedroom, sofa/TV -> Living, ...).
"""
import math
import re
import struct
import zlib
from collections import deque

import numpy as np

ROOM_TYPES = [
    ("Bathroom", r"\bwc\b|toilet|commode|basin|w\.?b\.?\b|shower|vanity|sanitary|faucet|closet"),
    ("Kitchen", r"kitchen|sink|hob|chimney|fridge|refrigerator|oven|microwave|cooktop"),
    ("Bedroom", r"\bbed\b|bed[ _]|mattress|double ?bed|king|queen|headboard"),
    ("Living", r"sofa|couch|\btv\b|tv[ _]|television|armchair|recliner|lg_tv|center table|coffee table"),
    ("Dining", r"dining|dinig"),
    ("Pooja", r"mandir|pooja|puja|temple"),
    ("Study", r"desk|study|workstation"),
]


def world_triangles(occurrences, get_def, is_visible):
    """Stack world-space triangles (N, 3, 3) of all visible occurrences."""
    out = []
    for occ in occurrences:
        if not is_visible(occ):
            continue
        arr = get_def(occ["def"])
        if arr is None:
            continue
        pos, tri = arr
        M = np.asarray(occ["xf"], np.float64)
        wp = pos @ M[:3, :3].T + M[:3, 3]
        out.append(wp[tri].astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 3, 3), np.float32)


def floor_level(tris):
    """z of the main floor: lowest horizontal level carrying a large share of area."""
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(n, axis=1)
    horiz = np.abs(n[:, 2]) > 0.98 * (2 * area + 1e-12)
    z = tris[horiz, :, 2].mean(1)
    a = area[horiz]
    if not len(z):
        return float(tris[:, :, 2].min())
    bins = np.round(z / 0.05).astype(np.int64)
    uniq, inv = np.unique(bins, return_inverse=True)
    level_area = np.bincount(inv, weights=a)
    # Walls must actually stand on the level: require geometry crossing level+1.3 m above it.
    zmin, zmax = tris[:, :, 2].min(1), tris[:, :, 2].max(1)
    order = np.argsort(uniq)
    best = level_area.max()
    for i in order:
        if level_area[i] < 0.25 * best:
            continue
        zl = uniq[i] * 0.05
        crossing = ((zmin < zl + 1.3) & (zmax > zl + 1.3)).sum()
        if crossing > 50:
            return float(zl)
    return float(uniq[np.argmax(level_area)] * 0.05)


def dominant_angle(tris, h):
    """Main wall direction of the plan (radians in [0, pi/2)): SketchUp plans aren't always
    axis-aligned, and views squared to the walls need the real wall direction."""
    h = h + 1e-4
    z = tris[:, :, 2]
    t = tris[(z.min(1) < h) & (z.max(1) > h)].astype(np.float64)
    if not len(t):
        return 0.0
    a, b = t, np.roll(t, -1, axis=1)
    za, zb = a[..., 2], b[..., 2]
    cross = (za - h) * (zb - h) < 0
    s = np.where(cross, (h - za) / np.where(cross, zb - za, 1.0), 0.0)
    p = a + s[..., None] * (b - a)
    ok = cross.sum(1) == 2
    p, cross = p[ok], cross[ok]
    idx = np.argsort(~cross, axis=1, kind="stable")[:, :2]
    d = p[np.arange(len(p)), idx[:, 1], :2] - p[np.arange(len(p)), idx[:, 0], :2]
    L = np.linalg.norm(d, axis=1)
    ang = np.mod(np.arctan2(d[:, 1], d[:, 0]), math.pi / 2)
    hist, edges = np.histogram(ang, bins=90, range=(0, math.pi / 2), weights=L)
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)
    return float(edges[int(np.argmax(hist))] + (edges[1] - edges[0]) / 2)


def slice_points(tris, h, step):
    """Sample points (x, y) along the intersection of triangles with plane z = h."""
    h = h + 1e-4
    z = tris[:, :, 2]
    m = (z.min(1) < h) & (z.max(1) > h)
    t = tris[m].astype(np.float64)
    if not len(t):
        return np.zeros((0, 2))
    a, b = t, np.roll(t, -1, axis=1)
    za, zb = a[..., 2], b[..., 2]
    cross = (za - h) * (zb - h) < 0
    s = np.where(cross, (h - za) / np.where(cross, zb - za, 1.0), 0.0)
    p = a + s[..., None] * (b - a)
    ok = cross.sum(1) == 2
    p, cross = p[ok], cross[ok]
    idx = np.argsort(~cross, axis=1, kind="stable")[:, :2]
    p0 = p[np.arange(len(p)), idx[:, 0], :2]
    p1 = p[np.arange(len(p)), idx[:, 1], :2]
    n = np.maximum(np.ceil(np.linalg.norm(p1 - p0, axis=1) / step).astype(np.int64), 1) + 1
    seg = np.repeat(np.arange(len(p0)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    tt = (np.arange(n.sum()) - start) / np.repeat(n - 1, n)
    return p0[seg] + (p1[seg] - p0[seg]) * tt[:, None]


class Grid:
    def __init__(self, pts, cell):
        lo = np.percentile(pts, 0.5, axis=0) - 1.5
        hi = np.percentile(pts, 99.5, axis=0) + 1.5
        span = hi - lo
        cell = max(cell, float(span.max()) / 1500)  # cap grid size
        self.cell, self.lo = cell, lo
        self.shape = (int(span[1] / cell) + 1, int(span[0] / cell) + 1)  # rows=y, cols=x

    def raster(self, pts):
        g = np.zeros(self.shape, bool)
        ij = ((pts - self.lo) / self.cell).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[1]) & (ij[:, 1] < self.shape[0])
        g[ij[ok, 1], ij[ok, 0]] = True
        return g

    def to_world(self, r, c):
        return self.lo[0] + (c + 0.5) * self.cell, self.lo[1] + (r + 0.5) * self.cell


def dilate(m, k=1):
    for _ in range(k):
        d = m.copy()
        d[1:] |= m[:-1]
        d[:-1] |= m[1:]
        d[:, 1:] |= m[:, :-1]
        d[:, :-1] |= m[:, 1:]
        m = d
    return m


def box_blur(a, r):
    c = np.cumsum(np.cumsum(np.pad(a, ((r + 1, r), (r + 1, r))), 0), 1)
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def label(free):
    """4-connected components (BFS). Returns label image (0 = not free)."""
    lab = np.zeros(free.shape, np.int32)
    H, W = free.shape
    n = 0
    fr = free.copy()
    for r0, c0 in zip(*np.nonzero(fr)):
        if lab[r0, c0]:
            continue
        n += 1
        lab[r0, c0] = n
        q = deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < H and 0 <= cc < W and fr[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = n
                    q.append((rr, cc))
    return lab, n


def distance_to_wall(room):
    """Chessboard-ish distance (cells) from each room cell to the nearest non-room cell."""
    dist = np.zeros(room.shape, np.int32)
    m = room.copy()
    k = 0
    while m.any():
        k += 1
        dist[m] = k
        e = m.copy()
        e[1:] &= m[:-1]
        e[:-1] &= m[1:]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        e[0, :] = e[-1, :] = False
        e[:, 0] = e[:, -1] = False
        m = e
    return dist


def cast_rays(origins, angles, blocked, furniture, cell, detail, max_dist=14.0):
    """2D ray march. Returns (distance m, furniture cells passed, detail at hit) per ray."""
    angles = np.asarray(angles, np.float64)
    R = len(origins)
    if angles.ndim == 1:                        # same fan for every origin
        angles = np.broadcast_to(angles, (R, len(angles)))
    A = angles.shape[1]
    step = 0.5
    nsteps = int(max_dist / cell / step)
    o = np.repeat(origins[:, None, :], A, 1).reshape(-1, 2).astype(np.float64)
    a = angles.reshape(-1)
    d = np.stack([np.cos(a), np.sin(a)], 1) * step
    dist = np.full(len(o), nsteps, np.float64)
    furn = np.zeros(len(o), np.int32)
    det = np.zeros(len(o), np.float32)
    first_furn = np.full(len(o), 1e9)
    alive = np.ones(len(o), bool)
    H, W = blocked.shape
    for k in range(1, nsteps):
        idx = np.nonzero(alive)[0]
        if not len(idx):
            break
        p = o[idx] + d[idx] * k
        c = p[:, 0].astype(np.int64)
        r = p[:, 1].astype(np.int64)
        out = (c < 0) | (r < 0) | (c >= W) | (r >= H)
        c, r = np.clip(c, 0, W - 1), np.clip(r, 0, H - 1)
        hit = out | blocked[r, c]
        fh = furniture[r, c] & ~hit
        furn[idx] += fh
        first_furn[idx[fh]] = np.minimum(first_furn[idx[fh]], k)
        dist[idx[hit]] = k
        det[idx[hit]] = detail[r[hit], c[hit]]
        alive[idx[hit]] = False
    return ((dist * step * cell).reshape(R, A), furn.reshape(R, A), det.reshape(R, A),
            (first_furn * step * cell).reshape(R, A))


def under_cabinet_strips(tris, nrm, tri_area, floor, grid):
    """Where a wall-cabinet underside (1.3-1.85 m) hangs over a counter (0.8-1.0 m), return
    LED strip placements. Works on raw geometry, so it doesn't matter how the kitchen is grouped."""
    horiz = np.abs(nrm[:, 2]) > 0.98 * (2 * tri_area + 1e-12)
    zc = tris[:, :, 2].mean(1) - floor

    def mask(sel):
        m = np.zeros(grid.shape, bool)
        t = tris[sel]
        lo = ((t[:, :, :2].min(1) - grid.lo) / grid.cell).astype(np.int64)
        hi = ((t[:, :, :2].max(1) - grid.lo) / grid.cell).astype(np.int64) + 1
        for (c0, r0), (c1, r1) in zip(lo, hi):
            if (c1 - c0) * (r1 - r0) < 40000:
                m[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = True
        return m

    under = mask(horiz & (nrm[:, 2] < 0) & (zc > 1.3) & (zc < 1.85))
    counter = mask(horiz & (nrm[:, 2] > 0) & (zc > 0.8) & (zc < 1.0))
    both = under & counter
    lab, n = label(both)
    out = []
    for k in range(1, n + 1):
        rr, cc = np.nonzero(lab == k)
        if len(rr) * grid.cell ** 2 < 0.08:
            continue
        lo = grid.lo + np.array([cc.min(), rr.min()]) * grid.cell
        hi = grid.lo + np.array([cc.max() + 1, rr.max() + 1]) * grid.cell
        size = hi - lo
        if size.max() < 0.35:
            continue
        sel = horiz & (nrm[:, 2] < 0) & (zc > 1.3) & (zc < 1.85)
        cz = tris[sel, :, 2].mean(1)
        cxy = tris[sel, :, :2].mean(1)
        inside = (cxy[:, 0] >= lo[0]) & (cxy[:, 0] <= hi[0]) & (cxy[:, 1] >= lo[1]) & (cxy[:, 1] <= hi[1])
        z = float(np.median(cz[inside])) if inside.any() else floor + 1.45
        c = (lo + hi) / 2
        out.append({"centre": [float(c[0]), float(c[1]), z - 0.015], "along_x": bool(size[0] >= size[1]),
                    "length": float(size.max() * 0.92)})
    return out


def largest_patch(tris, grid, room_mask):
    """Area (m2) of the largest connected patch covered by (roughly horizontal) triangles."""
    if not len(tris):
        return 0.0
    m = np.zeros(grid.shape, bool)
    lo = ((tris[:, :, :2].min(1) - grid.lo) / grid.cell).astype(np.int64)
    hi = ((tris[:, :, :2].max(1) - grid.lo) / grid.cell).astype(np.int64) + 1
    for (c0, r0), (c1, r1) in zip(lo, hi):
        m[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = True   # tri bbox (furniture tops are rectangles)
    m &= dilate(room_mask, 2)
    rr, cc = np.nonzero(m)
    if not len(rr):
        return 0.0
    r0, c0 = rr.min(), cc.min()
    lab, n = label(m[r0:rr.max() + 1, c0:cc.max() + 1])
    return float(np.bincount(lab.ravel())[1:].max()) * grid.cell ** 2 if n else 0.0


def room_type(names):
    counts = {}
    for t, rx in ROOM_TYPES:
        counts[t] = sum(1 for n in names if re.search(rx, n, re.I))
    best = max(counts, key=counts.get)
    return best if counts[best] else "Room"


def analyze(tris, occ_centres, cell=0.06, eye=1.3, log=print):
    """Find the floor, rooms, room types and ceiling heights. Returns a Plan (or None)."""
    floor = floor_level(tris)
    step = cell * 0.5
    eye_cut = slice_points(tris, floor + eye, step)
    head_cut = slice_points(tris, floor + 2.2, step)
    low_cut = slice_points(tris, floor + 0.5, step)
    if len(eye_cut) < 50:
        log("  auto-cameras: no walls found at eye height")
        return None
    axis = dominant_angle(tris, floor + eye)
    grid = Grid(eye_cut, cell)
    cell = grid.cell
    walls_eye = dilate(grid.raster(eye_cut), 1)
    walls = walls_eye | dilate(grid.raster(head_cut), 1)
    furniture = grid.raster(low_cut) & ~walls
    # Detail map: how busy the geometry is (panelling, shelves, fluting) - many short cut
    # segments per cell at several heights. Plain walls produce few long segments.
    detail = np.zeros(grid.shape, np.float32)
    for hh in (0.9, 1.5, 2.0):
        pts = slice_points(tris, floor + hh, 1e9)  # endpoints only (step huge -> 2 pts/segment)
        ij = ((pts - grid.lo) / cell).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < grid.shape[1]) & (ij[:, 1] < grid.shape[0])
        np.add.at(detail, (ij[ok, 1], ij[ok, 0]), 1.0)
    detail = box_blur(detail, 4)
    detail = np.minimum(detail / max(np.percentile(detail[detail > 0], 90), 1e-6), 1.0) if (detail > 0).any() else detail
    lab, n = label(~walls)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))

    rooms = []
    for k in range(1, n + 1):
        if k in border:
            continue
        m = lab == k
        area = m.sum() * cell * cell
        if area < 2.0 or area > 200:
            continue
        rr, cc = np.nonzero(m)
        r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        crop = m[r0:r1, c0:c1]
        dist = distance_to_wall(np.pad(crop, 1))[1:-1, 1:-1]
        if dist.max() * cell < 0.45:  # narrower than ~0.9 m: wall cavity / shaft
            continue
        rooms.append({"id": k, "mask": m, "area": area, "box": (r0, r1, c0, c1), "dist": dist})
    log(f"  auto-cameras: floor z={floor:.2f} m, grid {grid.shape[1]}x{grid.shape[0]} @ {cell * 100:.0f} cm, "
        f"{len(rooms)} rooms")

    # Room names: furniture names first, then geometry cues (surfaces at bed / counter height).
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    nrm = np.cross(e1, e2)
    tri_area = 0.5 * np.linalg.norm(nrm, axis=1)
    up = nrm[:, 2] > 0.98 * (2 * tri_area + 1e-12)          # upward-facing horizontal
    cen = tris[up].mean(1)
    hz, ha = cen[:, 2] - floor, tri_area[up]
    hc = ((cen[:, :2] - grid.lo) / cell).astype(np.int64)
    inside = (hc[:, 0] >= 0) & (hc[:, 1] >= 0) & (hc[:, 0] < grid.shape[1]) & (hc[:, 1] < grid.shape[0])
    hlab = np.zeros(len(hc), np.int32)
    hlab[inside] = lab[hc[inside, 1], hc[inside, 0]]

    def room_of(pts_xy):
        ij = ((pts_xy - grid.lo) / cell).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < grid.shape[1]) & (ij[:, 1] < grid.shape[0])
        out = np.zeros(len(ij), np.int32)
        out[ok] = lab[ij[ok, 1], ij[ok, 0]]
        return out

    # Mostly-upward faces incl. draped bedding / cushions (for the bed test).
    soft = nrm[:, 2] > 0.7 * (2 * tri_area + 1e-12)
    soft_tris = tris[soft]
    soft_c = soft_tris.mean(1)
    soft_hz, soft_lab = soft_c[:, 2] - floor, room_of(soft_c[:, :2])

    # Horizontal faces of either orientation (ceilings are often modelled facing up).
    horiz = np.abs(nrm[:, 2]) > 0.98 * (2 * tri_area + 1e-12)
    cen_any = tris[horiz].mean(1)
    hz_any, ta_any = cen_any[:, 2] - floor, tri_area[horiz]
    hlab_any = room_of(cen_any[:, :2])
    # Wall tops: vertical faces just inside each room (walls sit in the 'walls' mask, so probe
    # a few cm towards the face's front side).
    vert = np.abs(nrm[:, 2]) < 0.2 * (2 * tri_area + 1e-12)
    vt = tris[vert]
    vn = nrm[vert][:, :2] / np.maximum(np.linalg.norm(nrm[vert][:, :2], axis=1, keepdims=True), 1e-12)
    vc = vt.mean(1)[:, :2]
    vlab = np.maximum(room_of(vc + vn * cell * 2.5), room_of(vc - vn * cell * 2.5))
    vtop = vt[:, :, 2].max(1) - floor
    keep = vtop > 2.0
    vlab, vtop = np.where(keep, vlab, 0), vtop
    for rm in rooms:
        names = []
        for (x, y, z), name in occ_centres:
            if not (floor - 0.2 < z < floor + 2.6):
                continue
            c = int((x - grid.lo[0]) / cell)
            r = int((y - grid.lo[1]) / cell)
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and rm["mask"][r, c]:
                names.append(name)
        t = room_type(names)
        if t == "Room":
            sel = hlab == rm["id"]
            # A bed is one big continuous surface at mattress height (>= ~1.6 m2); sofa seats and
            # coffee tables at similar heights are smaller, separate pieces.
            bed = largest_patch(soft_tris[(soft_lab == rm["id"]) & (soft_hz > 0.36) & (soft_hz < 0.8)],
                                grid, rm["mask"])
            counter = ha[sel & (hz > 0.82) & (hz < 0.97)].sum()
            table = largest_patch(tris[up][sel & (hz > 0.70) & (hz < 0.80)], grid, rm["mask"])
            rm["cues"] = {"bed_patch_m2": round(bed, 2), "table_m2": round(table, 2),
                          "counter_m2": round(float(counter), 2)}
            if rm["area"] < 4.5:
                t = "Bathroom" if bed < 1.0 else "Room"
            elif counter > 1.0 and counter > bed:
                t = "Kitchen"
            elif bed >= 2.6 or (bed >= 1.2 and table < 0.9):
                t = "Bedroom"
            elif table >= 0.9 and rm["area"] >= 12:
                t = "Living"
        rm["type"] = t
        # Content & ceiling: is there anything to show, and does the model have a ceiling here?
        rm["content"] = int(furniture[rm["mask"]].sum()) + len(names)
        above = hlab_any == rm["id"]
        rm["has_ceiling"] = bool(float(ta_any[above & (hz_any > 2.1)].sum()) > 0.4 * rm["area"])
        vert_sel = vlab == rm["id"]
        top = vtop[vert_sel]
        rm["ceiling_z"] = floor + float(np.clip(np.percentile(top, 90) if len(top) else 2.9, 2.4, 3.8))
    all_rooms = list(rooms)                      # every room gets light, even empty ones
    rooms = [r for r in rooms if r["content"] >= 8 or r["area"] >= 8]
    rest = [r for r in rooms if r["type"] == "Room" and r["area"] >= 10]
    if rest and not any(r["type"] == "Living" for r in rooms):
        max(rest, key=lambda r: r["area"])["type"] = "Living"
    counts = {}
    for rm in sorted(rooms, key=lambda r: -r["area"]):
        t = rm["type"]
        counts[t] = counts.get(t, 0) + 1
    seen = {}
    for rm in sorted(rooms, key=lambda r: -r["area"]):
        t = rm["type"]
        seen[t] = seen.get(t, 0) + 1
        rm["name"] = t if counts[t] == 1 else f"{t} {seen[t]}"

    log(f"  auto-cameras: floor z={floor:.2f} m, plan axis {math.degrees(axis):.0f} deg, "
        f"{len(rooms)} rooms with content")
    for r in all_rooms:
        r.setdefault("name", r.get("type", "Room"))
    strips = under_cabinet_strips(tris, nrm, tri_area, floor, grid)
    return Plan(grid=grid, floor=floor, eye=eye, axis=axis, lab=lab, walls=walls, walls_eye=walls_eye,
                furniture=furniture, detail=detail, rooms=rooms, all_rooms=all_rooms, strips=strips)


class Plan:
    """Rooms of one floor plus everything needed to propose and pick camera views."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.cell = self.grid.cell

    # ---------------------------------------------------------------- ceilings / info
    def ceilings(self):
        out = []
        for r in self.all_rooms:
            if not r["has_ceiling"]:
                out.append({"room": r["name"],
                            **ceiling_mesh(dilate(r["mask"], 3) & ~(self.lab > 0) | r["mask"], self.grid,
                                           r["ceiling_z"])})
        return out

    def room_lights(self):
        """One soft ceiling light rectangle per room (centre, size) just under the ceiling."""
        out = []
        for r in self.all_rooms:
            rr, cc = np.nonzero(r["mask"])
            lo = self.to_world([cc.min(), rr.min()])
            hi = self.to_world([cc.max() + 1, rr.max() + 1])
            size = np.maximum((hi - lo) * 0.6, 0.4)
            c = (lo + hi) / 2
            out.append({"room": r["name"], "centre": [float(c[0]), float(c[1]), r["ceiling_z"] - 0.04],
                        "size": [float(size[0]), float(size[1])]})
        return out

    def room_info(self):
        return [{"name": r["name"], "type": r["type"], "area_m2": round(r["area"], 1),
                 "has_ceiling": r["has_ceiling"], "cues": r.get("cues")} for r in self.rooms]

    def write_png(self, path):
        write_plan(path, self.walls, self.furniture, self.rooms)

    # ---------------------------------------------------------------- candidates
    def to_world(self, cells_xy):
        return self.grid.lo + np.asarray(cells_xy, np.float64) * self.cell

    def to_cells(self, xy):
        return (np.asarray(xy, np.float64) - self.grid.lo) / self.cell

    def free(self, rm, xy, margin=0.2):
        """Camera position is inside the room, off walls/tall furniture, with some clearance."""
        c, r = self.to_cells(xy).astype(int)
        H, W = self.walls.shape
        if not (0 <= r < H and 0 <= c < W) or not rm["mask"][r, c]:
            return False
        k = max(1, int(margin / self.cell))
        return not self.walls_eye[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1].any()

    def candidates(self, rm):
        """Hero (one-point, squared to a wall) and corner (two-point diagonal) proposals."""
        rr, cc = np.nonzero(rm["mask"])
        P = self.to_world(np.stack([cc + 0.5, rr + 0.5], 1))
        ring = dilate(rm["mask"], 3) & self.walls
        wr, wc = np.nonzero(ring)
        WP = self.to_world(np.stack([wc + 0.5, wr + 0.5], 1))
        wdet = self.detail[wr, wc] + 0.05
        small = rm["area"] < 7.0
        out = []
        for k in range(4):
            h = self.axis + k * math.pi / 2
            u, v = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
            s, t = P @ u, P @ v
            back, front = np.percentile(s, 1), np.percentile(s, 99)
            depth = front - back
            if depth < 1.8:
                continue
            band = s > front - 0.6
            width = np.percentile(t[band], 98) - np.percentile(t[band], 2) if band.sum() > 3 else np.ptp(t)
            fw = (WP @ u) > front - 0.05                     # cells of the wall we're facing
            tc = float(np.average((WP @ v)[fw], weights=wdet[fw])) if fw.any() else float(np.median(t))
            tc = float(np.clip(tc, np.percentile(t, 10), np.percentile(t, 90)))
            for back_off in (0.28, min(0.5, 0.15 * depth), 0.9):
                for dt in (0.0, -0.18 * width, 0.18 * width):
                    xy = (back + back_off) * u + (tc + dt) * v
                    if not self.free(rm, xy):
                        continue
                    D = front - (back + back_off)
                    hf = math.degrees(2 * math.atan((width / 2 + 0.25) / max(D, 0.5)))
                    out.append({"kind": "front", "xy": xy, "heading": h, "hfov": float(np.clip(hf, 66, 96))})
        # Corner / diagonal views: near-wall spots, looking across the room.
        dist = rm["dist"]
        r0, r1, c0, c1 = rm["box"]
        lo_d, hi_d = (0.25, 0.6) if small else (0.35, 0.8)
        cand = np.argwhere((dist * self.cell >= lo_d) & (dist * self.cell <= hi_d))
        stride = max(1, len(cand) // 60)
        centroid = P.mean(0)
        for rc in cand[::stride]:
            xy = self.to_world([rc[1] + c0 + 0.5, rc[0] + r0 + 0.5])
            if not self.free(rm, xy, 0.2):
                continue
            base = math.atan2(*(centroid - xy)[::-1])       # towards the room centre
            for dh in (-0.45, -0.2, 0.0, 0.2, 0.45):
                out.append({"kind": "corner", "xy": xy, "heading": base + dh, "hfov": 90.0 if small else 76.0})
        return out

    def prescore(self, rm, cands):
        """Cheap 2D score (coverage, furniture, feature walls, obstruction) for ranking."""
        if not cands:
            return np.zeros(0)
        blocked = self.walls_eye | self.walls
        origins = np.array([self.to_cells(c["xy"]) for c in cands])
        headings = np.array([c["heading"] for c in cands])
        hf = np.radians([c["hfov"] for c in cands])
        offs = np.linspace(-0.5, 0.5, 31)
        angles = (headings[:, None] + offs[None, :] * hf[:, None])       # (N, 31)
        d, f, det, ff = cast_rays(origins, angles, blocked, self.furniture, self.cell, self.detail)
        area = 0.5 * (np.minimum(d, 12) ** 2).sum(1) * (hf / 30)
        area_n = np.minimum(area / rm["area"], 1.3)
        furn_n = np.minimum(f.sum(1) * self.cell / max(math.sqrt(rm["area"]), 1) / 8, 1)
        centre = d[:, 15]
        return (0.8 * area_n + 0.45 * furn_n + 0.6 * det.mean(1) + 0.2 * np.minimum(centre, 6) / 6
                - 1.0 * (centre < 1.2) - 0.6 * (ff[:, 10:21] < 0.9).mean(1))

    def camera(self, rm, c, aspect):
        eye_h = 1.4 if rm["type"] == "Bathroom" else 1.25
        x, y = c["xy"]
        z = self.floor + eye_h
        h = c["heading"]
        vfov = math.degrees(2 * math.atan(math.tan(math.radians(c["hfov"]) / 2) / aspect))
        return {"eye": [float(x), float(y), z], "target": [float(x + math.cos(h) * 5), float(y + math.sin(h) * 5), z],
                "up": [0, 0, 1], "perspective": True, "fov": vfov, "fov_is_height": True, "aspect": None}

    # ---------------------------------------------------------------- selection
    def choose_views(self, aspect=16 / 9, scorer=None, max_per_room=2, top_k=18, log=print):
        """Rank candidates in 2D, re-score the best ones with the 3D scorer, pick diverse views.

        scorer(list_of_camera_dicts) -> list of (score, set_of_object_ids)
        """
        views = []
        for rm in self.rooms:
            cands = self.candidates(rm)
            if not cands:
                continue
            pre = self.prescore(rm, cands)
            order = np.argsort(-pre)
            # Keep the best of each kind so both view types reach the 3D stage.
            picked = [i for i in order if cands[i]["kind"] == "front"][:top_k // 3]
            picked += [i for i in order if i not in picked][:top_k - len(picked)]
            cams = [self.camera(rm, cands[i], aspect) for i in picked]
            if scorer:
                s3 = scorer(cams)
            else:
                s3 = [(0.0, set()) for _ in picked]
            pmax = max(float(pre[picked].max()), 1e-6)
            total = [0.35 * pre[i] / pmax + s for i, (s, _) in zip(picked, s3)]
            ranked = sorted(range(len(picked)), key=lambda j: -total[j])
            limit = 1 if rm["area"] < 7.0 else max_per_room
            chosen = []
            for j in ranked:
                if len(chosen) >= limit:
                    break
                if chosen:
                    j0 = chosen[0]
                    c0, c1 = cands[picked[j0]], cands[picked[j]]
                    dh = abs((c1["heading"] - c0["heading"] + math.pi) % (2 * math.pi) - math.pi)
                    sep = float(np.linalg.norm(np.array(c1["xy"]) - np.array(c0["xy"])))
                    o0, o1 = s3[j0][1], s3[j][1]
                    overlap = len(o0 & o1) / max(len(o0 | o1), 1)
                    if total[j] < 0.7 * total[j0] or (dh < math.radians(60) and sep < 1.5) or overlap > 0.8:
                        continue
                chosen.append(j)
            label_kind = {"front": "Front", "corner": "Corner"}
            for n_, j in enumerate(chosen):
                c = cands[picked[j]]
                suffix = f" - {label_kind[c['kind']]}" if len(chosen) > 1 else ""
                if len(chosen) > 1 and n_ == 1 and cands[picked[chosen[0]]]["kind"] == c["kind"]:
                    suffix = f" - {label_kind[c['kind']]} 2"
                views.append({"name": rm["name"] + suffix, "auto": True, "room": rm["name"], "kind": c["kind"],
                              "score": round(float(total[j]), 2), "camera": cams[j]})
                rm.setdefault("cams", []).append((self.to_cells(c["xy"]), c["heading"], c["hfov"]))
        return views


def ceiling_mesh(mask, grid, z):
    """Quads (one per horizontal run of cells) covering a room mask at height z."""
    verts, faces = [], []
    cell, (x0, y0) = grid.cell, grid.lo
    for r in np.nonzero(mask.any(1))[0]:
        row = np.concatenate([[False], mask[r], [False]])
        starts = np.nonzero(row[1:] & ~row[:-1])[0]
        ends = np.nonzero(~row[1:] & row[:-1])[0]
        for s, e in zip(starts, ends):
            xa, xb = x0 + s * cell, x0 + e * cell
            ya, yb = y0 + r * cell, y0 + (r + 1) * cell
            k = len(verts)
            verts += [(xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z)]
            faces.append((k, k + 3, k + 2, k + 1))  # facing down
    return {"z": z, "verts": verts, "faces": faces}


# --------------------------------------------------------------------------- plan preview
PALETTE = [(239, 154, 154), (144, 202, 249), (165, 214, 167), (255, 224, 130), (206, 147, 216),
           (128, 222, 234), (255, 171, 145), (197, 202, 233), (230, 238, 156), (188, 170, 164)]


def write_plan(path, walls, furniture, rooms):
    H, W = walls.shape
    img = np.full((H, W, 3), 250, np.uint8)
    for i, rm in enumerate(rooms):
        img[rm["mask"]] = PALETTE[i % len(PALETTE)]
    img[furniture] = (img[furniture] * 0.7).astype(np.uint8)
    img[walls] = (40, 40, 40)
    for rm in rooms:
        for (ox, oy), a, fov in rm.get("cams", []):
            for da in np.radians(np.linspace(-fov / 2, fov / 2, 25)):
                for k in range(0, 40):
                    c, r = int(ox + math.cos(a + da) * k), int(oy + math.sin(a + da) * k)
                    if 0 <= r < H and 0 <= c < W:
                        img[r, c] = (220, 30, 30) if abs(da) < 0.02 or k > 36 else img[r, c]
            rr, cc = int(oy), int(ox)
            img[max(rr - 3, 0):rr + 4, max(cc - 3, 0):cc + 4] = (200, 0, 0)
    img = img[::-1]  # y up
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(H))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
