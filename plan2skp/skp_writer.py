"""Write extruded plan geometry to a native .skp with the SketchUp C API (SketchUpAPI.dll).

Each element becomes a named group on a tag (layer) with a material, so designers get a
clean, editable model: Walls / Lintels / Sills / Floors tags, one group per room floor.
"""
import ctypes
import os
import sys
from ctypes import byref, c_bool, c_size_t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skp2render.sketchup_api import SketchUpAPI, Ref, Point3D, Color  # noqa: E402

INCH = 0.0254


def _ring_coords(ring):
    pts = list(ring.coords)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


class SkpWriter:
    def __init__(self):
        self.api = SketchUpAPI()
        self.dll = self.api.dll
        self.model = Ref()
        self.api.SUModelCreate(byref(self.model))
        self.root = Ref()
        self.api.SUModelGetEntities(self.model, byref(self.root))
        self.layers, self.materials = {}, {}
        self.lo = [1e18, 1e18, 1e18]
        self.hi = [-1e18, -1e18, -1e18]

    # ------------------------------------------------------------------ tags / materials
    def layer(self, name):
        if name not in self.layers:
            l = Ref()
            self.api.SULayerCreate(byref(l))
            self.api.SULayerSetName(l, name.encode("utf-8"))
            arr = (Ref * 1)(l)
            self.api.SUModelAddLayers(self.model, 1, arr)
            self.layers[name] = l
        return self.layers[name]

    def material(self, name, rgb, opacity=1.0):
        if name not in self.materials:
            m = Ref()
            self.api.SUMaterialCreate(byref(m))
            self.api.SUMaterialSetName(m, name.encode("utf-8"))
            col = Color(*rgb, 255)
            self.api.SUMaterialSetColor(m, byref(col))
            if opacity < 1.0:
                self.api.SUMaterialSetOpacity(m, ctypes.c_double(opacity))
                self.api.SUMaterialSetUseOpacity(m, ctypes.c_bool(True))
            arr = (Ref * 1)(m)
            self.api.SUModelAddMaterials(self.model, 1, arr)
            self.materials[name] = m
        return self.materials[name]

    # ------------------------------------------------------------------ geometry
    def group(self, name, tag, material=None):
        g = Ref()
        self.api.SUGroupCreate(byref(g))
        self.api.SUEntitiesAddGroup(self.root, g)
        self.api.SUGroupSetName(g, name.encode("utf-8"))
        self.dll.SUGroupToDrawingElement.restype = Ref
        el = self.dll.SUGroupToDrawingElement(g)
        self.api.SUDrawingElementSetLayer(el, self.layer(tag))
        if material is not None:
            self.api.SUDrawingElementSetMaterial(el, material)
        ents = Ref()
        self.api.SUGroupGetEntities(g, byref(ents))
        return ents

    def _vertex(self, gi, x, y, z):
        for i, v in enumerate((x, y, z)):
            self.lo[i], self.hi[i] = min(self.lo[i], v), max(self.hi[i], v)
        p = Point3D(x / INCH, y / INCH, z / INCH)
        self.api.SUGeometryInputAddVertex(gi, byref(p))

    def _face(self, gi, indices, holes=()):
        loop = Ref()
        self.api.SULoopInputCreate(byref(loop))
        for i in indices:
            self.api.SULoopInputAddVertexIndex(loop, c_size_t(i))
        fi = c_size_t()
        self.api.SUGeometryInputAddFace(gi, byref(loop), byref(fi))
        for h in holes:
            hl = Ref()
            self.api.SULoopInputCreate(byref(hl))
            for i in h:
                self.api.SULoopInputAddVertexIndex(hl, c_size_t(i))
            self.api.SUGeometryInputFaceAddInnerLoop(gi, fi, byref(hl))

    def extrude(self, entities, polygons, z0, z1):
        """Closed solids from shapely polygons (outward-facing faces, holes supported)."""
        from shapely.geometry.polygon import orient
        gi = Ref()
        self.api.SUGeometryInputCreate(byref(gi))
        n = 0
        for poly in polygons:
            poly = orient(poly, sign=1.0)            # exterior CCW, holes CW
            rings = [_ring_coords(poly.exterior)] + [_ring_coords(r) for r in poly.interiors]
            if len(rings[0]) < 3:
                continue
            base = []
            for ring in rings:
                idx = []
                for (x, y) in ring:
                    self._vertex(gi, x, y, z0)
                    self._vertex(gi, x, y, z1)
                    idx.append((n, n + 1))
                    n += 2
                base.append(idx)
            # top (CCW from above -> normal up) and bottom (reversed -> normal down)
            self._face(gi, [t for _, t in base[0]], [[t for _, t in r] for r in base[1:]])
            self._face(gi, [b for b, _ in base[0]][::-1], [[b for b, _ in r][::-1] for r in base[1:]])
            for ring in base:                        # sides face outward for CCW / CW rings
                for k in range(len(ring)):
                    (a0, a1), (b0, b1) = ring[k], ring[(k + 1) % len(ring)]
                    self._face(gi, [a0, b0, b1, a1])
        if n:
            self.api.SUEntitiesFill(entities, gi, c_bool(True))
        self.api.SUGeometryInputRelease(byref(gi))

    def flat(self, entities, polygons, z):
        from shapely.geometry.polygon import orient
        gi = Ref()
        self.api.SUGeometryInputCreate(byref(gi))
        n = 0
        for poly in polygons:
            poly = orient(poly, sign=1.0)
            rings = [_ring_coords(poly.exterior)] + [_ring_coords(r) for r in poly.interiors]
            idx = []
            for ring in rings:
                ids = []
                for (x, y) in ring:
                    self._vertex(gi, x, y, z)
                    ids.append(n)
                    n += 1
                idx.append(ids)
            self._face(gi, idx[0], idx[1:])
        if n:
            self.api.SUEntitiesFill(entities, gi, c_bool(True))
        self.api.SUGeometryInputRelease(byref(gi))

    def overview_camera(self):
        """Open the model on a 3D bird's-eye view of the whole plan (not an empty view)."""
        if self.lo[0] > self.hi[0]:
            return
        cx, cy = (self.lo[0] + self.hi[0]) / 2, (self.lo[1] + self.hi[1]) / 2
        span = max(self.hi[0] - self.lo[0], self.hi[1] - self.lo[1])
        eye = Point3D((cx - 0.55 * span) / INCH, (cy - 0.85 * span) / INCH, (1.1 * span) / INCH)
        tgt = Point3D(cx / INCH, cy / INCH, 0.0)
        up = Point3D(0.0, 0.0, 1.0)
        cam = Ref()
        self.api.SUModelGetCamera(self.model, byref(cam))
        self.api.SUCameraSetOrientation(cam, byref(eye), byref(tgt), byref(up))
        self.api.SUCameraSetPerspectiveFrustumFOV(cam, ctypes.c_double(35.0))

    def save(self, path):
        self.overview_camera()
        self.api.SUModelSaveToFile(self.model, os.path.abspath(path).encode("utf-8"))
        self.api.SUModelRelease(byref(self.model))
