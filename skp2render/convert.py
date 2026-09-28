"""SKP -> render package converter.

Output directory layout:
    scene.json      materials, definitions, instance occurrences, cameras, sun, lights
    geometry.npz    per-definition mesh buffers (metres, local space)
    textures/       material texture images

Geometry is kept instanced: every component/group definition is extracted once
(in its local space) and referenced by occurrences carrying a world transform
and the material it inherits from its parent chain.
"""
import ctypes
import json
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skp2render.sketchup_api import (SketchUpAPI, Ref, Point3D, Color, Transformation, byref,
                                     c_bool, c_double, c_int, c_size_t, c_int64)

INCH = 0.0254
INHERIT = -1
BIG_FACE_VERTS = 4000  # faces above this bypass SketchUp's (quadratic) triangulator

# SUTypedValueType
TV_INT32, TV_FLOAT, TV_DOUBLE, TV_BOOL, TV_TIME, TV_STRING, TV_VECTOR, TV_ARRAY = 3, 4, 5, 6, 8, 9, 10, 11


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:60] or "tex"


class Converter:
    def __init__(self, skp_path, out_dir, log=print):
        self.api = SketchUpAPI()
        self.skp_path = skp_path
        self.out_dir = out_dir
        self.log = log
        os.makedirs(os.path.join(out_dir, "textures"), exist_ok=True)
        self.model = self.api.open_model(skp_path)
        self.materials = []          # list of dicts
        self.mat_index = {}          # SUMaterialRef ptr -> index
        self.layers = {}             # ptr -> dict(name, visible, index)
        self.definitions = {}        # def ptr -> index
        self.def_meta = []           # list of dicts
        self.geom = {}               # arrays for npz
        self.occurrences = []
        self.lights = []
        self.stats = {"faces": 0, "triangles": 0, "occ_triangles": 0}
        # Reusable buffers the mesh helper writes into (grown on demand).
        self._P = np.empty((1 << 16, 3), np.float64)   # positions (inches)
        self._N = np.empty_like(self._P)               # normals
        self._S = np.empty_like(self._P)               # STQ texture coords
        self._I = np.empty((1 << 16, 3), np.uint64)    # triangle indices (size_t)
        dll = self.api.dll
        for fn in ("SUMeshHelperGetVertexIndices", "SUMeshHelperGetVertices", "SUMeshHelperGetNormals",
                   "SUMeshHelperGetFrontSTQCoords", "SUMeshHelperGetBackSTQCoords"):
            getattr(dll, fn).argtypes = [Ref, c_size_t, ctypes.c_void_p, ctypes.POINTER(c_size_t)]

    def _grow_buffers(self, nv, nt, vused, tused):
        def grow(buf, need, used):
            if need <= len(buf):
                return buf
            new = np.empty((max(need, 2 * len(buf)), 3), buf.dtype)
            new[:used] = buf[:used]
            return new
        self._P, self._N, self._S = (grow(b, nv, vused) for b in (self._P, self._N, self._S))
        self._I = grow(self._I, nt, tused)

    # ------------------------------------------------------------------ materials
    def read_materials(self):
        api = self.api
        for mat in api.list("SUModelGetNumMaterials", "SUModelGetMaterials", self.model):
            name = api.string("SUMaterialGetName", mat)
            col = Color()
            api.dll.SUMaterialGetColor(mat, byref(col))
            opacity = api.get_double("SUMaterialGetOpacity", mat) or 1.0
            use_opacity = bool(api.get_bool("SUMaterialGetUseOpacity", mat))
            entry = {"name": name, "color": [col.r / 255, col.g / 255, col.b / 255],
                     "opacity": opacity if use_opacity else 1.0, "texture": None}
            tex = Ref()
            if api.dll.SUMaterialGetTexture(mat, byref(tex)) == 0 and tex.valid():
                w, h, ss, ts = c_size_t(), c_size_t(), c_double(), c_double()
                api.SUTextureGetDimensions(tex, byref(w), byref(h), byref(ss), byref(ts))
                src_name = api.string("SUTextureGetFileName", tex)
                ext = os.path.splitext(src_name)[1].lower()
                if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    ext = ".png"
                fname = f"{len(self.materials):03d}_{safe_name(os.path.splitext(os.path.basename(src_name))[0])}{ext}"
                path = os.path.join(self.out_dir, "textures", fname)
                mtype = c_int()
                api.dll.SUMaterialGetType(mat, byref(mtype))
                if self.write_texture(tex, path, colorized=mtype.value == 2):
                    alpha = c_bool()
                    api.dll.SUTextureGetUseAlphaChannel(tex, byref(alpha))
                    entry["texture"] = {"file": "textures/" + fname, "px": [w.value, h.value],
                                        "s_scale": ss.value, "t_scale": ts.value,
                                        "alpha": bool(alpha.value), "source_name": src_name}
            self.mat_index[mat.key()] = len(self.materials)
            self.materials.append(entry)

    def write_texture(self, tex, path, colorized):
        """Write a material texture; colorized materials get SketchUp's tinted image."""
        dll = self.api.dll
        if colorized:
            rep = Ref()
            if dll.SUImageRepCreate(byref(rep)) == 0:
                try:
                    if (dll.SUTextureGetColorizedImageRep(tex, byref(rep)) == 0
                            and dll.SUImageRepSaveToFile(rep, path.encode("utf-8")) == 0):
                        return True
                finally:
                    dll.SUImageRepRelease(byref(rep))
        return dll.SUTextureWriteToFile(tex, path.encode("utf-8")) == 0

    def mat_of(self, getter, ref):
        m = Ref()
        if getattr(self.api.dll, getter)(ref, byref(m)) == 0 and m.valid():
            return self.mat_index.get(m.key(), INHERIT)
        return INHERIT

    # ------------------------------------------------------------------ layers
    def read_layers(self):
        api = self.api
        for i, layer in enumerate(api.list("SUModelGetNumLayers", "SUModelGetLayers", self.model)):
            self.layers[layer.key()] = {"index": i, "name": api.string("SULayerGetName", layer),
                                        "visible": bool(api.get_bool("SULayerGetVisibility", layer))}

    def element_state(self, element):
        """(hidden, layer_index, layer_visible) for a drawing element."""
        hidden = bool(self.api.get_bool("SUDrawingElementGetHidden", element))
        layer = Ref()
        info = None
        if self.api.dll.SUDrawingElementGetLayer(element, byref(layer)) == 0 and layer.valid():
            info = self.layers.get(layer.key())
        if info is None:
            return hidden, 0, True
        return hidden, info["index"], info["visible"]

    # ------------------------------------------------------------------ geometry
    def extract_faces(self, entities):
        """Triangulate all visible faces of an entities collection (local space, metres)."""
        api, dll = self.api, self.api.dll
        faces = api.list("SUEntitiesGetNumFaces", "SUEntitiesGetFaces", entities)
        # The API writes straight into large reusable numpy buffers; all per-vertex maths is
        # done once, vectorised, after the loop (per-face Python work dominated conversion).
        vused = tused = 0
        rv, rnv, rnt, rmat, rback = [], [], [], [], []
        ntri, nvert, got = c_size_t(), c_size_t(), c_size_t()
        mesh = Ref()
        renderable = self.renderable_layers
        get_idx, get_pts = dll.SUMeshHelperGetVertexIndices, dll.SUMeshHelperGetVertices
        get_nrm = dll.SUMeshHelperGetNormals
        get_front, get_back = dll.SUMeshHelperGetFrontSTQCoords, dll.SUMeshHelperGetBackSTQCoords
        pa, na, sa, ia = (b.ctypes.data for b in (self._P, self._N, self._S, self._I))
        big = []
        fverts = c_size_t()
        for face in faces:
            el = dll.SUFaceToDrawingElement(face)
            hidden, layer_idx, _ = self.element_state(el)
            if hidden or layer_idx not in renderable:
                continue
            front = self.mat_of("SUFaceGetFrontMaterial", face)
            back = self.mat_of("SUFaceGetBackMaterial", face)
            # Designers often paint only the side they see; prefer front, fall back to back.
            use_back = front == INHERIT and back != INHERIT
            dll.SUFaceGetNumVertices(face, byref(fverts))
            if fverts.value > BIG_FACE_VERTS:
                # SketchUp's triangulator is quadratic on huge faces (lattices, hatches):
                # export the loops and let Blender's scanfill triangulate them.
                big.append(self.face_loops(face, back if use_back else front, use_back))
                continue
            if dll.SUMeshHelperCreate(byref(mesh), face) != 0:
                continue
            dll.SUMeshHelperGetNumTriangles(mesh, byref(ntri))
            dll.SUMeshHelperGetNumVertices(mesh, byref(nvert))
            nt, nv = ntri.value, nvert.value
            if nt:
                if vused + nv > len(self._P) or tused + nt > len(self._I):
                    self._grow_buffers(vused + nv, tused + nt, vused, tused)
                    pa, na, sa, ia = (b.ctypes.data for b in (self._P, self._N, self._S, self._I))
                get_idx(mesh, nt * 3, ia + tused * 24, byref(got))
                get_pts(mesh, nv, pa + vused * 24, byref(got))
                get_nrm(mesh, nv, na + vused * 24, byref(got))
                (get_back if use_back else get_front)(mesh, nv, sa + vused * 24, byref(got))
                rv.append(vused); rnv.append(nv); rnt.append(nt)
                rmat.append(back if use_back else front); rback.append(use_back)
                vused += nv
                tused += nt
            dll.SUMeshHelperRelease(byref(mesh))
        if big:
            self.pending_big = big
        if not rv:
            return None
        self.stats["faces"] += len(rv)
        nt_arr, nv_arr = np.array(rnt), np.array(rnv)
        tri = self._I[:tused].astype(np.int64) + np.repeat(np.array(rv), nt_arr)[:, None]
        slot_ids = np.repeat(np.array(rmat, np.int32), nt_arr)
        pos = self._P[:vused] * INCH
        nrm = self._N[:vused].copy()
        nrm[np.repeat(np.array(rback), nv_arr)] *= -1
        s = self._S[:vused]
        q = np.where(np.abs(s[:, 2]) < 1e-12, 1.0, s[:, 2])
        uv = s[:, :2] / q[:, None]
        # SketchUp's mesh helper doesn't wind triangles consistently; make every triangle's
        # winding agree with the (authoritative) face normal.
        geo_n = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
        flip = (geo_n * nrm[tri].sum(1)).sum(1) < 0
        tri[flip] = tri[flip][:, ::-1]
        return {
            "pos": pos.astype(np.float32),
            "nrm": nrm.astype(np.float32),
            "uv": uv.astype(np.float32),
            "tri": tri.astype(np.int32),
            "mat": slot_ids,
        }

    def face_loops(self, face, mat, use_back):
        """Outer + inner loops of a face (metres) with planar UVs, for external triangulation."""
        api, dll = self.api, self.api.dll
        loops = []
        outer = Ref()
        dll.SUFaceGetOuterLoop(face, byref(outer))
        loops.append(outer)
        loops += api.list("SUFaceGetNumInnerLoops", "SUFaceGetInnerLoops", face)
        n = Point3D()
        dll.SUFaceGetNormal(face, byref(n))
        normal = np.array(n.tuple()) * (-1 if use_back else 1)
        pts, lens = [], []
        p = Point3D()
        for loop in loops:
            verts = api.list("SULoopGetNumVertices", "SULoopGetVertices", loop)
            for v in verts:
                dll.SUVertexGetPosition(v, byref(p))
                pts.append(p.tuple())
            lens.append(len(verts))
        pts = np.array(pts)
        # Planar UV in texture repeats (SketchUp default projection for unprojected faces).
        axis_u = np.cross(normal, [0, 0, 1]) if abs(normal[2]) < 0.9 else np.array([1.0, 0, 0])
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)
        uv = np.stack([pts @ axis_u, pts @ axis_v], 1)
        tex = self.materials[mat]["texture"] if mat != INHERIT else None
        if tex:
            uv *= [tex["s_scale"], tex["t_scale"]]
        return {"pts": (pts * INCH).astype(np.float32), "uv": uv.astype(np.float32),
                "lens": np.array(lens, np.int32), "normal": normal.astype(np.float32), "mat": mat}

    def definition(self, def_ref, name, is_group):
        """Extract a definition once; return its index."""
        key = def_ref.key()
        if key in self.definitions:
            return self.definitions[key]
        idx = len(self.def_meta)
        self.definitions[key] = idx
        ents = Ref()
        self.api.SUComponentDefinitionGetEntities(def_ref, byref(ents))
        meta = {"name": name, "group": is_group, "tris": 0, "children": []}
        self.def_meta.append(meta)
        light = self.definition_light(def_ref, idx)
        if light:
            meta["light"] = light
        self.store_geometry(idx, meta, self.extract_faces(ents))
        meta["children"] = self.children_of(ents)
        return idx

    def store_geometry(self, idx, meta, g):
        pts = []
        if g is not None:
            for k, v in g.items():
                self.geom[f"d{idx}_{k}"] = v
            meta["tris"] = int(len(g["tri"]))
            self.stats["triangles"] += len(g["tri"])
            pts.append(g["pos"])
        big = getattr(self, "pending_big", None)
        self.pending_big = None
        if big:
            meta["big_faces"] = []
            for k, bf in enumerate(big):
                for key in ("pts", "uv", "lens"):
                    self.geom[f"d{idx}_big{k}_{key}"] = bf[key]
                meta["big_faces"].append({"mat": bf["mat"], "normal": bf["normal"].tolist()})
                pts.append(bf["pts"])
                meta["tris"] += len(bf["pts"])  # approximate (one tri per vertex)
        if pts:
            allp = np.concatenate(pts)
            meta["bbox"] = [allp.min(0).tolist(), allp.max(0).tolist()]

    def children_of(self, entities):
        api, dll = self.api, self.api.dll
        out = []
        items = [(g, True) for g in api.list("SUEntitiesGetNumGroups", "SUEntitiesGetGroups", entities)]
        items += [(c, False) for c in api.list("SUEntitiesGetNumInstances", "SUEntitiesGetInstances", entities)]
        for ref, is_group in items:
            el = (dll.SUGroupToDrawingElement if is_group else dll.SUComponentInstanceToDrawingElement)(ref)
            # Model-level hidden state only reflects the last active scene; keep everything
            # and resolve visibility per view.
            hidden, layer_idx, layer_vis = self.element_state(el)
            eid = (dll.SUGroupToEntity if is_group else dll.SUComponentInstanceToEntity)(ref).key()
            d = Ref()
            if is_group:
                api.SUGroupGetDefinition(ref, byref(d))
                name = api.string("SUGroupGetName", ref)
            else:
                api.SUComponentInstanceGetDefinition(ref, byref(d))
                name = api.string("SUComponentInstanceGetName", ref)
            def_name = api.string("SUComponentDefinitionGetName", d)
            tr = Transformation()
            (api.SUGroupGetTransform if is_group else api.SUComponentInstanceGetTransform)(ref, byref(tr))
            m = np.array(tr.values).reshape(4, 4).T  # column-major -> row-major
            m[:3, 3] *= INCH
            out.append({
                "def": self.definition(d, def_name, is_group),
                "name": name,
                "xf": m.tolist(),
                "mat": self.mat_of("SUDrawingElementGetMaterial", el),
                "layer": layer_idx,
                "hidden": hidden,
                "eid": eid,
                "attrs": self.light_attributes(ref, is_group),
            })
        return out

    def light_attributes(self, ref, is_group):
        """Read attribute dictionaries that carry light settings (Enscape, V-Ray...)."""
        api, dll = self.api, self.api.dll
        ent = (dll.SUGroupToEntity if is_group else dll.SUComponentInstanceToEntity)(ref)
        n = c_size_t()
        if dll.SUEntityGetNumAttributeDictionaries(ent, byref(n)) != 0 or n.value == 0:
            return None
        dicts = (Ref * n.value)()
        dll.SUEntityGetAttributeDictionaries(ent, n.value, dicts, byref(n))
        found = {}
        for dref in dicts[: n.value]:
            dname = self.api.string("SUAttributeDictionaryGetName", dref)
            if not re.search(r"enscape|light|vray|lumion|d5", dname, re.I):
                continue
            found[dname] = self.read_dictionary(dref)
        return found or None

    def definition_light(self, def_ref, idx):
        """Parse Enscape light proxies (stored as XML on the component definition)."""
        dll = self.api.dll
        dll.SUComponentDefinitionToEntity.restype = Ref
        ent = dll.SUComponentDefinitionToEntity(def_ref)
        d = Ref()
        if dll.SUEntityGetAttributeDictionary(ent, b"Enscape.Light", byref(d)) != 0 or not d.valid():
            return None
        xml = self.read_dictionary(d).get("LightData")
        if not isinstance(xml, str):
            return None
        import base64
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(xml.encode("utf-8"))
        except ET.ParseError:
            return None
        kind = root.get("{http://www.w3.org/2001/XMLSchema-instance}type", "SketchupLight")
        light = {"source": "enscape", "kind": kind.replace("Sketchup", "").replace("Light", "").lower() or "point"}
        for child in root:
            tag = child.tag
            if tag == "IesData" and child.text:
                path = os.path.join(self.out_dir, "textures", f"light_{idx}.ies")
                with open(path, "wb") as f:
                    f.write(base64.b64decode(child.text))
                light["ies"] = "textures/" + os.path.basename(path)
            elif tag != "OriginalIesFile" and child.text:
                try:
                    light[tag.lower()] = float(child.text)
                except ValueError:
                    light[tag.lower()] = child.text
        for k in ("length", "width", "height", "radius"):
            if k in light:
                light[k] *= INCH
        return light

    def read_dictionary(self, dref):
        api, dll = self.api, self.api.dll
        n = c_size_t()
        dll.SUAttributeDictionaryGetNumKeys(dref, byref(n))
        keys = (Ref * n.value)()
        for i in range(n.value):
            api.SUStringCreate(byref(keys[i]))
        dll.SUAttributeDictionaryGetKeys(dref, n.value, keys, byref(n))
        out = {}
        for i in range(n.value):
            k = self._str(keys[i])
            tv = Ref()
            api.SUTypedValueCreate(byref(tv))
            if dll.SUAttributeDictionaryGetValue(dref, k.encode("utf-8"), byref(tv)) == 0:
                out[k] = self.typed(tv)
            api.SUTypedValueRelease(byref(tv))
            api.SUStringRelease(byref(keys[i]))
        return out

    def _str(self, sref):
        n = c_size_t()
        self.api.SUStringGetUTF8Length(sref, byref(n))
        b = ctypes.create_string_buffer(n.value + 1)
        self.api.SUStringGetUTF8(sref, n.value + 1, b, byref(n))
        return b.value.decode("utf-8", "replace")

    def typed(self, tv):
        dll = self.api.dll
        t = c_int()
        dll.SUTypedValueGetType(tv, byref(t))
        t = t.value
        if t == TV_INT32:
            v = c_int(); dll.SUTypedValueGetInt32(tv, byref(v)); return v.value
        if t == TV_FLOAT:
            v = ctypes.c_float(); dll.SUTypedValueGetFloat(tv, byref(v)); return v.value
        if t == TV_DOUBLE:
            v = c_double(); dll.SUTypedValueGetDouble(tv, byref(v)); return v.value
        if t == TV_BOOL:
            v = c_bool(); dll.SUTypedValueGetBool(tv, byref(v)); return v.value
        if t == TV_STRING:
            return self.api.string("SUTypedValueGetString", tv)
        if t == TV_TIME:
            v = c_int64(); dll.SUTypedValueGetTime(tv, byref(v)); return v.value
        if t == TV_VECTOR:
            v = (c_double * 3)(); dll.SUTypedValueGetVector3d(tv, v); return list(v)
        if t == TV_ARRAY:
            n = c_size_t()
            dll.SUTypedValueGetNumArrayItems(tv, byref(n))
            items = (Ref * n.value)()
            for i in range(n.value):
                self.api.SUTypedValueCreate(byref(items[i]))
            dll.SUTypedValueGetArrayItems(tv, n.value, items, byref(n))
            out = [self.typed(items[i]) for i in range(n.value)]
            for i in range(n.value):
                self.api.SUTypedValueRelease(byref(items[i]))
            return out
        return None

    # ------------------------------------------------------------------ occurrences
    def flatten(self, children, parent_xf, parent_mat, parent_layers, parent_eids, parent_hidden, path):
        """Expand the instance tree into world-space occurrences.

        Each occurrence keeps its ancestor chain (entity ids, layers, model-hidden flag) so
        the renderer can apply any scene's hidden-object / hidden-layer state.
        """
        for ch in children:
            xf = parent_xf @ np.array(ch["xf"])
            mat = ch["mat"] if ch["mat"] != INHERIT else parent_mat
            layers = parent_layers + [ch["layer"]]
            eids = parent_eids + [ch["eid"]]
            hidden = parent_hidden or ch["hidden"]
            meta = self.def_meta[ch["def"]]
            name_path = path + [ch["name"] or meta["name"]]
            occ = {"def": ch["def"], "xf": xf.tolist(), "mat": mat, "layers": layers,
                   "eids": eids, "model_hidden": hidden, "path": name_path}
            if ch.get("attrs"):
                occ["attrs"] = ch["attrs"]
            self.occurrences.append(occ)
            self.stats["occ_triangles"] += meta["tris"]
            self.flatten(meta["children"], xf, mat, layers, eids, hidden, name_path)

    # ------------------------------------------------------------------ cameras / sun
    def camera(self, cam):
        api = self.api
        e, t, u = Point3D(), Point3D(), Point3D()
        api.SUCameraGetOrientation(cam, byref(e), byref(t), byref(u))
        c = {"eye": [v * INCH for v in e.tuple()], "target": [v * INCH for v in t.tuple()], "up": list(u.tuple()),
             "perspective": bool(api.get_bool("SUCameraGetPerspective", cam)),
             "fov": api.get_double("SUCameraGetPerspectiveFrustumFOV", cam),
             "fov_is_height": api.get_bool("SUCameraGetFOVIsHeight", cam),
             "aspect": api.get_double("SUCameraGetAspectRatio", cam),
             "two_d": bool(api.get_bool("SUCameraGet2D", cam))}
        zn, zf = c_double(), c_double()
        if api.dll.SUCameraGetClippingDistances(cam, byref(zn), byref(zf)) == 0:
            c["clip"] = [zn.value * INCH, zf.value * INCH]
        if not c["perspective"]:
            c["ortho_height"] = (api.get_double("SUCameraGetOrthographicFrustumHeight", cam) or 0) * INCH
        if c["two_d"]:
            cen = Point3D()
            if api.dll.SUCameraGetCenter2D(cam, byref(cen)) == 0:
                c["center_2d"] = list(cen.tuple())
            c["scale_2d"] = api.get_double("SUCameraGetScale2D", cam)
        return c

    def shadow_info(self, si):
        keys = ["Latitude", "Longitude", "SunDirection", "ShadowTime", "TZOffset", "Light", "Dark",
                "DisplayShadows", "UseSunForAllShading", "City", "Country"]
        out = {}
        for k in keys:
            tv = Ref()
            self.api.SUTypedValueCreate(byref(tv))
            if self.api.dll.SUShadowInfoGetValue(si, k.encode(), byref(tv)) == 0:
                out[k] = self.typed(tv)
            self.api.SUTypedValueRelease(byref(tv))
        return out

    def read_scenes(self):
        api = self.api
        scenes = []
        layer_by_ptr = self.layers
        for sc in api.list("SUModelGetNumScenes", "SUModelGetScenes", self.model):
            cam = Ref()
            api.SUSceneGetCamera(sc, byref(cam))
            s = {"name": api.string("SUSceneGetName", sc), "camera": self.camera(cam)}
            if api.get_bool("SUSceneGetUseHiddenLayers", sc):
                # Layers stored with a scene are the ones hidden in that scene.
                s["hidden_layers"] = [layer_by_ptr[l.key()]["index"]
                                      for l in api.list("SUSceneGetNumLayers", "SUSceneGetLayers", sc)
                                      if l.key() in layer_by_ptr]
            use_hidden = api.get_bool("SUSceneGetUseHiddenObjects", sc)
            if use_hidden is None:
                use_hidden = api.get_bool("SUSceneGetUseHidden", sc)
            if use_hidden:
                s["hidden_eids"] = [e.key() for e in
                                    api.list("SUSceneGetNumHiddenEntities", "SUSceneGetHiddenEntities", sc)]
            if api.get_bool("SUSceneGetUseShadowInfo", sc):
                si = Ref()
                if api.dll.SUSceneGetShadowInfo(sc, byref(si)) == 0:
                    s["shadow"] = self.shadow_info(si)
            scenes.append(s)
        return scenes

    # ------------------------------------------------------------------ run
    def run(self):
        t0 = time.time()
        api = self.api
        self.read_layers()
        self.read_materials()
        scenes = self.read_scenes()
        # Loose faces can't be toggled per view; keep faces on layers visible in any view.
        self.renderable_layers = {l["index"] for l in self.layers.values() if l["visible"]}
        for s in scenes:
            hidden = set(s.get("hidden_layers", []))
            self.renderable_layers |= {l["index"] for l in self.layers.values() if l["index"] not in hidden}
        self.log(f"  materials: {len(self.materials)}  ({time.time() - t0:.1f}s)")
        root = Ref()
        api.SUModelGetEntities(self.model, byref(root))
        root_meta = {"name": "<root>", "group": True, "tris": 0, "children": []}
        self.def_meta.append(root_meta)
        self.definitions["root"] = 0
        self.store_geometry(0, root_meta, self.extract_faces(root))
        root_meta["children"] = self.children_of(root)
        self.log(f"  geometry: {self.stats['faces']} faces, {self.stats['triangles']} unique tris "
                 f"({time.time() - t0:.1f}s)")
        self.occurrences.append({"def": 0, "xf": np.eye(4).tolist(), "mat": INHERIT, "layers": [],
                                 "eids": [], "model_hidden": False, "path": ["<root>"]})
        self.flatten(root_meta["children"], np.eye(4), INHERIT, [], [], False, [])
        self.stats["occ_triangles"] += root_meta["tris"]

        cam = Ref()
        api.SUModelGetCamera(self.model, byref(cam))
        si = Ref()
        api.SUModelGetShadowInfo(self.model, byref(si))
        scene = {
            "source": os.path.basename(self.skp_path),
            "units": "m",
            "materials": self.materials,
            "layers": sorted(self.layers.values(), key=lambda l: l["index"]),
            "definitions": [{k: v for k, v in d.items() if k != "children"} for d in self.def_meta],
            "occurrences": self.occurrences,
            "model_camera": self.camera(cam),
            "scenes": scenes,
            "shadow": self.shadow_info(si),
            "stats": self.stats,
        }
        with open(os.path.join(self.out_dir, "scene.json"), "w", encoding="utf-8") as f:
            json.dump(scene, f)
        np.savez(os.path.join(self.out_dir, "geometry.npz"), **self.geom)
        self.log(f"  occurrences: {len(self.occurrences)}, rendered tris: {self.stats['occ_triangles']}, "
                 f"scenes: {len(scene['scenes'])}  total {time.time() - t0:.1f}s")
        self.api.SUModelRelease(byref(self.model))
        return scene


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("skp")
    ap.add_argument("out")
    a = ap.parse_args()
    Converter(a.skp, a.out).run()


if __name__ == "__main__":
    main()
