"""Build a Cycles scene from a converted SketchUp package and render its views.

Run inside Blender:
    blender -b --factory-startup -P blender_build.py -- --package DIR --out DIR [options]
"""
import argparse
import json
import math
import os
import re
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skp2render import materials as matlib  # noqa: E402

INHERIT = -1
ADAPTIVE_THRESHOLD = float(os.environ.get("SKP_ADAPTIVE", "0.02"))
WARM_K = 3300.0  # interior practical lights (warm white, as Indian interior renders are expected)
LUMENS_PER_WATT = 683.0 / 4.0  # empirical Blender-light calibration; exposure is auto-balanced anyway


def log(*a):
    print("[build]", *a, flush=True)


# --------------------------------------------------------------------------- scene setup
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    return scene


def setup_cycles(scene, samples, width, height):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    backend = None
    for dev_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == dev_type]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == dev_type
                backend = dev_type
                break
        except TypeError:
            continue
    c = scene.cycles
    c.device = "GPU" if backend else "CPU"
    log("device:", backend or "CPU")
    c.samples = samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = ADAPTIVE_THRESHOLD
    c.use_denoising = True
    try:
        c.denoiser = "OPENIMAGEDENOISE"
        c.denoising_use_gpu = True
        c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        c.denoising_prefilter = "ACCURATE"   # cleaner albedo/normal guides -> no blotchy walls
        c.denoising_quality = "HIGH"
    except (AttributeError, TypeError):
        pass
    c.adaptive_min_samples = 32
    c.max_bounces = 10
    c.diffuse_bounces = 5
    c.glossy_bounces = 5
    c.transmission_bounces = 10
    c.transparent_max_bounces = 16
    c.sample_clamp_indirect = 8.0
    c.blur_glossy = 1.0
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.use_light_tree = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    vs = scene.view_settings
    vs.view_transform = "AgX"
    for look in ("AgX - Medium High Contrast", "Medium High Contrast", "AgX - Base Contrast"):
        try:
            vs.look = look
            break
        except TypeError:
            continue
    # Neutralise the warm interior lights so white walls stay white (Blender 4.3+).
    if hasattr(vs, "use_white_balance"):
        vs.use_white_balance = True
        vs.white_balance_temperature = 4300
    return backend


# --------------------------------------------------------------------------- geometry
class Builder:
    def __init__(self, pkg, opts):
        pkg = os.path.abspath(pkg)
        self.pkg = pkg
        self.opts = opts
        with open(os.path.join(pkg, "scene.json"), encoding="utf-8") as f:
            self.data = json.load(f)
        self.geom = np.load(os.path.join(pkg, "geometry.npz"))
        self.mats = self.data["materials"]
        self.bl_mats = {}
        self.meshes = {}
        self.objects = []          # (object, occurrence)
        self.coll = bpy.data.collections.new("SketchUp")
        bpy.context.scene.collection.children.link(self.coll)
        self.light_coll = bpy.data.collections.new("Lights")
        bpy.context.scene.collection.children.link(self.light_coll)

    def material(self, idx):
        if idx not in self.bl_mats:
            info = None if idx == INHERIT else self.mats[idx]
            self.bl_mats[idx] = matlib.build_material(info, self.pkg)
        return self.bl_mats[idx]

    def def_arrays(self, def_idx):
        """Definition mesh buffers, including huge faces triangulated here with scanfill."""
        if not hasattr(self, "_def_cache"):
            self._def_cache = {}
        if def_idx in self._def_cache:
            return self._def_cache[def_idx]
        from mathutils.geometry import tessellate_polygon
        p = f"d{def_idx}_"
        parts = []
        if p + "tri" in self.geom.files:
            parts.append((self.geom[p + "pos"], self.geom[p + "nrm"], self.geom[p + "uv"],
                          self.geom[p + "tri"], self.geom[p + "mat"]))
        for k, bf in enumerate(self.data["definitions"][def_idx].get("big_faces", [])):
            pts, uvs, lens = (self.geom[f"{p}big{k}_{s}"] for s in ("pts", "uv", "lens"))
            starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
            loops = [[tuple(v) for v in pts[s:s + n]] for s, n in zip(starts, lens)]
            t = np.array(tessellate_polygon(loops), np.int64).reshape(-1, 3)
            if not len(t):
                continue
            nrm = np.repeat(np.array([bf["normal"]], np.float32), len(pts), 0)
            parts.append((pts, nrm, uvs, t, np.full(len(t), bf["mat"], np.int32)))
        if not parts:
            self._def_cache[def_idx] = None
            return None
        offs = np.cumsum([0] + [len(x[0]) for x in parts[:-1]])
        out = (np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts]),
               np.concatenate([x[2] for x in parts]),
               np.concatenate([x[3].astype(np.int64) + o for x, o in zip(parts, offs)]),
               np.concatenate([x[4] for x in parts]))
        self._def_cache[def_idx] = out
        return out

    def mesh_for(self, def_idx, inherited):
        key = (def_idx, inherited)
        if key in self.meshes:
            return self.meshes[key]
        arrays = self.def_arrays(def_idx)
        if arrays is None:
            self.meshes[key] = None
            return None
        pos, nrm, uv, tri, slot = arrays
        uv = uv.copy()
        ok = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
        tri, slot = tri[ok], slot[ok]
        if len(tri) == 0:
            self.meshes[key] = None
            return None
        # Faces with the default material take the parent's material. Their SketchUp UVs are
        # in inches, so scale them by the inherited texture's repeat size.
        inh_tex = inherited != INHERIT and self.mats[inherited]["texture"]
        if inh_tex:
            vsel = np.zeros(len(pos), bool)
            vsel[tri[slot == INHERIT].ravel()] = True
            uv[vsel, 0] *= inh_tex["s_scale"]
            uv[vsel, 1] *= inh_tex["t_scale"]
        eff = np.where(slot == INHERIT, inherited, slot)
        uniq, slot_idx = np.unique(eff, return_inverse=True)

        me = bpy.data.meshes.new(f"def{def_idx}_m{inherited}")
        me.vertices.add(len(pos))
        me.vertices.foreach_set("co", pos.ravel())
        nt = len(tri)
        me.loops.add(nt * 3)
        me.loops.foreach_set("vertex_index", tri.ravel().astype(np.int32))
        me.polygons.add(nt)
        me.polygons.foreach_set("loop_start", np.arange(0, nt * 3, 3, dtype=np.int32))
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", uv[tri.ravel()].ravel())
        for m in uniq:
            me.materials.append(self.material(int(m)))
        me.polygons.foreach_set("material_index", slot_idx.astype(np.int32))
        me.update()
        if self.opts.custom_normals:
            # SketchUp smooths normals across soft edges even at 90 degree corners; on big flat
            # faces that interpolates into dark streaks. Keep smoothing only on gentle curvature.
            p = pos[tri]
            gn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            gn /= np.maximum(np.linalg.norm(gn, axis=1, keepdims=True), 1e-12)
            ln = nrm[tri]                                    # (nt, 3, 3) per-corner normals
            ok = (ln * gn[:, None, :]).sum(2) > math.cos(math.radians(35))
            ln = np.where(ok[:, :, None], ln, gn[:, None, :])
            me.polygons.foreach_set("use_smooth", np.ones(nt, bool))
            me.normals_split_custom_set(ln.reshape(-1, 3))
        self.meshes[key] = me
        return me

    def build_geometry(self):
        t0 = time.time()
        defs = self.data["definitions"]
        for i, occ in enumerate(self.data["occurrences"]):
            d = defs[occ["def"]]
            if d.get("light"):
                continue  # light proxy geometry is not rendered
            me = self.mesh_for(occ["def"], occ["mat"])
            if me is None:
                continue
            ob = bpy.data.objects.new(f"{d['name'][:40]}_{i}", me)
            ob.matrix_world = Matrix(occ["xf"])
            self.coll.objects.link(ob)
            self.objects.append((ob, occ))
        log(f"geometry: {len(self.objects)} objects, {len(self.meshes)} meshes, {time.time() - t0:.1f}s")

    # ----------------------------------------------------------------------- lights
    def build_lights(self):
        defs = self.data["definitions"]
        n = 0
        for i, occ in enumerate(self.data["occurrences"]):
            d = defs[occ["def"]]
            light = d.get("light")
            xf = Matrix(occ["xf"])
            if light:
                if self.opts.model_lights:     # designer's Enscape lights (can be switched off)
                    self.add_enscape_light(light, xf, f"{d['name']}_{i}", occ)
                    n += 1
            elif self.opts.fixture_lights:
                if self.add_fixture_light(d, xf, f"{d['name']}_{i}", occ):
                    n += 1
        found = self.detect_ceiling_fixtures() if self.opts.fixture_lights else 0
        log(f"lights: {n} from model data, {found} detected ceiling fixtures")

    FIXTURE_EXCLUDE = re.compile(r"fan|\bac\b|air ?con|smoke|sprinkler|speaker|camera|sensor|curtain|rod|hook|"
                                 r"frame|photo|clock|switch|socket|plant|vase", re.I)

    def detect_ceiling_fixtures(self, max_lights=120):
        """Downlights/spots in the geometry: small objects hanging on the ceiling.

        Most SketchUp models carry no light data at all, but designers do model the fixtures.
        A candidate is a compact object (<= 40 cm across, <= 30 cm tall) whose top touches a
        ceiling (ray up hits within 12 cm) and which has open space below it.
        """
        scene = bpy.context.scene
        dg = bpy.context.evaluated_depsgraph_get()
        defs = self.data["definitions"]
        lit = []
        for ob, occ in list(self.objects):
            if ob.type != "MESH" or occ["def"] == 0:
                continue
            d = defs[occ["def"]]
            label = d["name"] + " " + " ".join(occ["path"])
            if self.FIXTURE_EXCLUDE.search(label) or d.get("light"):
                continue
            corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
            lo = Vector([min(c[i] for c in corners) for i in range(3)])
            hi = Vector([max(c[i] for c in corners) for i in range(3)])
            size = hi - lo
            if not (0.04 <= max(size.x, size.y) <= 0.4 and size.z <= 0.3 and lo.z > 1.9):
                continue
            centre = (lo + hi) / 2
            top = Vector((centre.x, centre.y, hi.z - 0.005))
            # Mounted on a ceiling: something (ceiling slab or the fixture's own trim) right above.
            hit_up, *_ = scene.ray_cast(dg, top, Vector((0, 0, 1)), distance=0.15)
            if not hit_up:
                continue
            below = Vector((centre.x, centre.y, lo.z - 0.01))
            hit_dn, loc_dn, *_ = scene.ray_cast(dg, below, Vector((0, 0, -1)), distance=4.0)
            if not hit_dn or (below - loc_dn).length < 1.2:
                continue
            key = (round(centre.x / 0.08), round(centre.y / 0.08))
            if any(abs(key[0] - k[0]) <= 1 and abs(key[1] - k[1]) <= 1 for k in lit):
                continue                      # nested sub-parts of the same fixture
            lit.append(key)
            self.add_downlight(f"Downlight_{len(lit)}", below, max(size.x, size.y), occ)
            if len(lit) >= max_lights:
                break
        return len(lit)

    def add_downlight(self, name, pos, fixture_size, occ):
        ld = bpy.data.lights.new(name, "SPOT")
        ld.spot_size = math.radians(100)
        ld.spot_blend = 0.75
        ld.energy = 650 / LUMENS_PER_WATT * self.opts.light_scale
        ld.shadow_soft_size = 0.03
        ld.color = kelvin_to_rgb(WARM_K)
        ob = bpy.data.objects.new(name, ld)
        ob.location = pos
        self.light_coll.objects.link(ob)
        self.objects.append((ob, occ))
        # The lens itself glows (camera sees a lit downlight, bloom picks it up).
        r = max(0.02, fixture_size * 0.3)
        me = bpy.data.meshes.new(name + "_lens")
        k = 12
        verts = [(pos.x + r * math.cos(2 * math.pi * i / k), pos.y + r * math.sin(2 * math.pi * i / k), pos.z + 0.004)
                 for i in range(k)]
        me.from_pydata(verts, [], [tuple(range(k))[::-1]])
        mat = bpy.data.materials.new(name + "_glow")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*kelvin_to_rgb(WARM_K), 1)
        em.inputs["Strength"].default_value = 25.0
        nt.links.new(em.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs[0])
        me.materials.append(mat)
        lens = bpy.data.objects.new(name + "_lens", me)
        lens.visible_shadow = False
        self.light_coll.objects.link(lens)
        self.objects.append((lens, occ))

    # Real-world light output. Enscape users routinely crank values 10-300x (Enscape
    # auto-exposes), which would blow out a physically based render; clamp to plausible ranges.
    LUMEN_RANGE = {"spot": (350, 1400), "point": (300, 1600), "rect": (400, 2500), "linear_per_m": (200, 700)}

    def add_enscape_light(self, light, xf, name, occ):
        lum = float(light.get("luminosity", 1000.0))
        kind = light["kind"]
        if kind == "linear":
            length = light.get("length", 1.0) * xf.to_scale().y
            lo, hi = self.LUMEN_RANGE["linear_per_m"]
            lum = float(np.clip(lum / max(length, 0.05), lo, hi)) * max(length, 0.05)
            watts = lum / LUMENS_PER_WATT * self.opts.light_scale
            self.add_tube(name, xf, length, watts, occ)
            return
        key = "rect" if kind in ("rectangle", "rectangular", "area") else ("spot" if kind in ("spot", "ies") else "point")
        lum = float(np.clip(lum, *self.LUMEN_RANGE[key]))
        watts = lum / LUMENS_PER_WATT * self.opts.light_scale
        if kind in ("rectangle", "rectangular", "area"):
            ld = bpy.data.lights.new(name, "AREA")
            ld.shape = "RECTANGLE"
            ld.size = max(light.get("width", 0.3), 0.01)
            ld.size_y = max(light.get("height", light.get("length", 0.3)), 0.01)
        elif kind == "spot":
            ld = bpy.data.lights.new(name, "SPOT")
            ld.spot_size = math.radians(light.get("coneangle", light.get("angle", 60.0)))
            ld.spot_blend = 0.3
            ld.shadow_soft_size = 0.03
        else:  # point / ies / sphere
            ld = bpy.data.lights.new(name, "POINT")
            ld.shadow_soft_size = max(light.get("radius", 0.02), 0.02)
        ld.energy = watts
        ld.color = kelvin_to_rgb(light.get("temperature", WARM_K))
        if light.get("ies"):
            self.attach_ies(ld, os.path.join(self.pkg, light["ies"]))
        ob = bpy.data.objects.new(name, ld)
        loc, rot, _ = xf.decompose()
        ob.location = loc
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = rot
        self.light_coll.objects.link(ob)
        self.objects.append((ob, occ))

    def attach_ies(self, ld, path):
        if not os.path.exists(path):
            return
        ld.use_nodes = True
        nt = ld.node_tree
        emit = next(n for n in nt.nodes if n.type == "EMISSION")
        ies = nt.nodes.new("ShaderNodeTexIES")
        ies.mode = "EXTERNAL"
        ies.filepath = path
        # IES texture is normalised to its peak; strength input scales it.
        nt.links.new(ies.outputs["Fac"], emit.inputs["Strength"])

    def add_tube(self, name, xf, length, watts, occ):
        """Linear light: an emissive cylinder along local Y, hidden from camera."""
        radius = 0.008
        segs = 8
        ang = np.linspace(0, 2 * np.pi, segs, endpoint=False)
        ring = np.stack([np.cos(ang) * radius, np.zeros(segs), np.sin(ang) * radius], 1)
        verts = np.concatenate([ring + [0, -length / 2, 0], ring + [0, length / 2, 0]])
        faces = [(k, (k + 1) % segs, segs + (k + 1) % segs, segs + k) for k in range(segs)]
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts.tolist(), [], faces)
        area = 2 * np.pi * radius * max(length, 0.01)
        mat = bpy.data.materials.new(name + "_emit")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*kelvin_to_rgb(WARM_K), 1)
        em.inputs["Strength"].default_value = watts / area
        nt.links.new(em.outputs[0], out.inputs[0])
        me.materials.append(mat)
        ob = bpy.data.objects.new(name, me)
        # Keep only rotation+translation; length already includes scale.
        loc, rot, _ = xf.decompose()
        ob.matrix_world = Matrix.LocRotScale(loc, rot, Vector((1, 1, 1)))
        ob.visible_camera = False
        ob.visible_shadow = False
        self.light_coll.objects.link(ob)
        self.objects.append((ob, occ))

    FIXTURE_RE = re.compile(r"(pendant|chandelier|wall ?lamp|floor ?lamp|table ?lamp|down ?light|spot ?light|"
                            r"ceiling ?light|hanging ?light|\blamp\b|sconce|\bcob\b)", re.I)

    def add_fixture_light(self, d, xf, name, occ):
        """Name-based fixture detection for models without light data."""
        if not self.FIXTURE_RE.search(d["name"]) or "bbox" not in d:
            return False
        lo, hi = Vector(d["bbox"][0]), Vector(d["bbox"][1])
        centre = (lo + hi) / 2
        is_down = re.search(r"down ?light|spot|cob|ceiling", d["name"], re.I)
        if is_down:
            p = Vector((centre.x, centre.y, lo.z - 0.02))
        elif re.search(r"pendant|chandelier|hanging", d["name"], re.I):
            p = Vector((centre.x, centre.y, lo.z + (hi.z - lo.z) * 0.15))
        else:
            p = centre
        ld = bpy.data.lights.new(name, "SPOT" if is_down else "POINT")
        if is_down:
            ld.spot_size = math.radians(90)
            ld.spot_blend = 0.6
        ld.energy = 800 / LUMENS_PER_WATT * self.opts.light_scale
        ld.shadow_soft_size = 0.04
        ld.color = kelvin_to_rgb(3000.0)
        ob = bpy.data.objects.new(name, ld)
        ob.matrix_world = xf @ Matrix.Translation(p)
        if not is_down:
            ob.matrix_world = Matrix.Translation(ob.matrix_world.to_translation())
        self.light_coll.objects.link(ob)
        self.objects.append((ob, occ))
        return True

    def add_window_portals(self, max_portals=48):
        """Light portals over windows: Cycles then aims daylight samples through the glass,
        which removes most of the noise in daylit interiors at the same sample count."""
        scene = bpy.context.scene
        dg = bpy.context.evaluated_depsgraph_get()
        n = 0
        seen = set()
        for ob, occ in self.objects:
            if ob.type != "MESH" or n >= max_portals:
                continue
            if not any(m and m.get("sketchup_class") == "glass" for m in ob.data.materials):
                continue
            corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
            lo = Vector([min(c[i] for c in corners) for i in range(3)])
            hi = Vector([max(c[i] for c in corners) for i in range(3)])
            size = hi - lo
            thin = min(range(2), key=lambda i: size[i])
            wide = 1 - thin
            if size[thin] > 0.12 or size[wide] * size.z < 0.25 or size.z < 0.4:
                continue
            key = (round(lo.x, 1), round(lo.y, 1), round(hi.x, 1), round(hi.y, 1), round(lo.z, 1))
            if key in seen:
                continue
            seen.add(key)
            centre = (lo + hi) / 2
            normal = Vector((1, 0, 0)) if thin == 0 else Vector((0, 1, 0))
            inside = []
            for s in (1, -1):
                p = centre + normal * (s * 0.6)
                hit, *_ = scene.ray_cast(dg, p, Vector((0, 0, 1)), distance=6.0)
                inside.append(hit)
            if inside[0] == inside[1]:
                continue
            into_room = normal if inside[0] else -normal
            ld = bpy.data.lights.new(f"Portal_{n}", "AREA")
            ld.shape = "RECTANGLE"
            ld.size, ld.size_y = size[wide] + 0.05, size.z + 0.05
            try:
                ld.cycles.is_portal = True
            except AttributeError:
                bpy.data.lights.remove(ld)
                return 0
            pob = bpy.data.objects.new(f"Portal_{n}", ld)
            pob.location = centre
            pob.rotation_mode = "QUATERNION"
            pob.rotation_quaternion = into_room.to_track_quat("-Z", "Z")
            self.light_coll.objects.link(pob)
            n += 1
        return n

    # ----------------------------------------------------------------------- sun & sky
    @staticmethod
    def _cloud_layer(nt, sep, sky_color):
        """Soft procedural clouds on a sky 'plane' (x/z, y/z), faded toward the horizon."""
        def math_node(op, a, b):
            m = nt.nodes.new("ShaderNodeMath")
            m.operation = op
            for i, v in enumerate((a, b)):
                if isinstance(v, (int, float)):
                    m.inputs[i].default_value = v
                else:
                    nt.links.new(v, m.inputs[i])
            return m.outputs[0]

        zp = math_node("ADD", sep.outputs["Z"], 0.12)
        u = math_node("DIVIDE", sep.outputs["X"], zp)
        v = math_node("DIVIDE", sep.outputs["Y"], zp)
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(u, comb.inputs["X"])
        nt.links.new(v, comb.inputs["Y"])
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 0.9
        noise.inputs["Detail"].default_value = 8.0
        noise.inputs["Roughness"].default_value = 0.58
        nt.links.new(comb.outputs[0], noise.inputs["Vector"])
        cover = nt.nodes.new("ShaderNodeMapRange")
        cover.inputs["From Min"].default_value = 0.5
        cover.inputs["From Max"].default_value = 0.72
        nt.links.new(noise.outputs["Fac"], cover.inputs["Value"])
        fade = nt.nodes.new("ShaderNodeMapRange")          # no clouds below / at the horizon
        fade.inputs["From Min"].default_value = 0.0
        fade.inputs["From Max"].default_value = 0.18
        nt.links.new(sep.outputs["Z"], fade.inputs["Value"])
        amount = math_node("MULTIPLY", cover.outputs[0], fade.outputs[0])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        a = next(s for s in mix.inputs if s.name == "A" and s.type == "RGBA")
        b = next(s for s in mix.inputs if s.name == "B" and s.type == "RGBA")
        nt.links.new(amount, mix.inputs["Factor"])
        nt.links.new(sky_color, a)
        b.default_value = (0.97, 0.97, 0.98, 1.0)
        return next(s for s in mix.outputs if s.type == "RGBA")

    def build_environment(self, shadow):
        """Sky + ground dome and sun from SketchUp's shadow settings.

        What the camera sees through windows uses a separate strength (set per view after
        exposure) so exteriors look like a real interior photo instead of black or blown out.
        """
        scene = bpy.context.scene
        world = bpy.data.worlds.new("World")
        scene.world = world
        world.use_nodes = True
        nt = world.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputWorld")
        night = self.opts.mode == "night"
        coord = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(coord.outputs["Generated"], sep.inputs[0])
        remap = nt.nodes.new("ShaderNodeMapRange")          # z in [-1, 1] -> [0, 1]
        remap.inputs["From Min"].default_value = -1.0
        nt.links.new(sep.outputs["Z"], remap.inputs["Value"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        els = ramp.color_ramp.elements
        if night:
            stops = [(0.0, (0.004, 0.004, 0.005)), (0.5, (0.01, 0.012, 0.02)), (1.0, (0.004, 0.006, 0.015))]
        else:
            stops = [(0.0, (0.25, 0.24, 0.22)), (0.495, (0.33, 0.32, 0.30)),
                     (0.505, (0.85, 0.9, 1.0)), (1.0, (0.32, 0.5, 0.9))]
        els[0].position, els[0].color = stops[0][0], (*stops[0][1], 1)
        els[1].position, els[1].color = stops[-1][0], (*stops[-1][1], 1)
        for pos, col in stops[1:-1]:
            e = els.new(pos)
            e.color = (*col, 1)
        nt.links.new(remap.outputs[0], ramp.inputs[0])
        bg_light = nt.nodes.new("ShaderNodeBackground")
        bg_cam = nt.nodes.new("ShaderNodeBackground")
        for bg in (bg_light, bg_cam):
            bg.inputs["Strength"].default_value = self.opts.sky_strength
        nt.links.new(ramp.outputs["Color"], bg_light.inputs["Color"])
        cam_color = ramp.outputs["Color"] if night else self._cloud_layer(nt, sep, ramp.outputs["Color"])
        nt.links.new(cam_color, bg_cam.inputs["Color"])
        path = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
        nt.links.new(bg_light.outputs[0], mix.inputs[1])
        nt.links.new(bg_cam.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs[0])
        self.backdrop = bg_cam
        sun_dir = Vector(shadow.get("SunDirection") or (-0.4, -0.7, 0.55)).normalized()
        if sun_dir.z > 0.02 and not night:
            ld = bpy.data.lights.new("Sun", "SUN")
            ld.energy = self.opts.sun_strength
            ld.angle = math.radians(0.8)
            ld.color = (1.0, 0.96, 0.9)
            ob = bpy.data.objects.new("Sun", ld)
            ob.rotation_mode = "QUATERNION"
            ob.rotation_quaternion = (-sun_dir).to_track_quat("-Z", "Y")
            self.light_coll.objects.link(ob)

    # ----------------------------------------------------------------------- visibility
    def analyze_plan(self):
        """Room analysis (floor, rooms, ceilings, per-room lights). Cached; also used when the
        model has its own scenes, for lighting."""
        if hasattr(self, "plan"):
            return self.plan
        from skp2render import cameras
        defs = self.data["definitions"]
        hidden_layers = {l["index"] for l in self.data["layers"] if not l["visible"]}

        def visible(occ):
            return (not occ["model_hidden"] and not hidden_layers.intersection(occ["layers"])
                    and not defs[occ["def"]].get("light"))

        def get_def(i):
            a = self.def_arrays(i)
            return None if a is None else (a[0], a[3])

        occs = self.data["occurrences"]
        tris = cameras.world_triangles(occs, get_def, visible)
        centres = []
        for occ in occs:
            d = defs[occ["def"]]
            if "bbox" not in d or not visible(occ) or occ["def"] == 0:
                continue
            c = (np.array(d["bbox"][0]) + np.array(d["bbox"][1])) / 2
            M = np.array(occ["xf"])
            centres.append((tuple(M[:3, :3] @ c + M[:3, 3]), d["name"] + " " + " ".join(occ["path"])))
        self.plan = cameras.analyze(tris, centres, log=log)
        if self.plan is not None:
            self.ceiling_data = self.plan.ceilings()
            self.room_light_data = self.plan.room_lights()
            if self.opts.auto_ceiling:
                log(f"ceilings added: {self.add_ceilings(self.ceiling_data)}")
        return self.plan

    def auto_views(self, opts, plan_png):
        plan = self.analyze_plan()
        if plan is None:
            return [], []
        views = plan.choose_views(aspect=opts.aspect, scorer=lambda cams: self.score_cameras(cams, opts.aspect),
                                  log=log)
        plan.write_png(plan_png)
        return views, plan.room_info()

    def add_room_lights(self, room_lights, strength):
        """Soft warm ceiling light in every room, so rooms seen through doors/glass aren't black."""
        for i, rl in enumerate(room_lights):
            ld = bpy.data.lights.new(f"RoomAmbient_{i}", "AREA")
            ld.shape = "RECTANGLE"
            ld.size, ld.size_y = rl["size"]
            ld.energy = strength * rl["size"][0] * rl["size"][1]
            ld.color = kelvin_to_rgb(3800.0)
            ob = bpy.data.objects.new(f"RoomAmbient_{i}", ld)
            ob.location = rl["centre"]
            ob.visible_camera = False
            ob.visible_glossy = False
            self.light_coll.objects.link(ob)
        return len(room_lights)

    def add_strips(self, strips):
        """Warm LED strips under kitchen wall cabinets (placements from the plan analysis)."""
        for i, s in enumerate(strips):
            pos = Vector(s["centre"])
            # add_tube runs along local Y: rotate 90 degrees about Z when the run is along X.
            m = Matrix.Translation(pos) @ (Matrix.Rotation(math.pi / 2, 4, "Z") if s["along_x"] else Matrix.Identity(4))
            self.add_tube(f"UnderCabinet_{i}", m, s["length"], 450 * s["length"] / LUMENS_PER_WATT,
                          {"layers": [], "eids": [], "model_hidden": False, "def": 0})
        return len(strips)

    def add_under_cabinet_strips(self, floor):
        """Kitchen wall cabinets hanging over a counter get a warm LED strip underneath."""
        scene = bpy.context.scene
        dg = bpy.context.evaluated_depsgraph_get()
        n = 0
        for ob, occ in list(self.objects):
            if ob.type != "MESH" or occ["def"] == 0:
                continue
            corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
            lo = Vector([min(c[i] for c in corners) for i in range(3)])
            hi = Vector([max(c[i] for c in corners) for i in range(3)])
            size = hi - lo
            depth, width = sorted((size.x, size.y))
            if not (floor + 1.25 < lo.z < floor + 1.8 and 0.22 < depth < 0.5 and width > 0.35
                    and 0.25 < size.z < 1.2):
                continue
            centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z - 0.01))
            hit, loc, *_ = scene.ray_cast(dg, centre, Vector((0, 0, -1)), distance=1.0)
            if not hit or not (floor + 0.78 < loc.z < floor + 1.02):
                continue                     # no counter directly below
            along = Vector((1, 0, 0)) if size.x >= size.y else Vector((0, 1, 0))
            across = Vector((-along.y, along.x, 0))
            # Front = the side facing open space (the back side touches a wall).
            back_hit = [scene.ray_cast(dg, centre + s * across * (depth / 2 + 0.02), s * across, distance=0.25)[0]
                        for s in (1, -1)]
            front = across if not back_hit[0] else -across
            pos = centre + front * (depth * 0.3)
            # add_tube runs along local Y: rotate 90 degrees about Z when the cabinet runs along X.
            m = Matrix.Translation(pos) @ (Matrix.Rotation(math.pi / 2, 4, "Z") if along.x else Matrix.Identity(4))
            self.add_tube(f"UnderCabinet_{n}", m, width * 0.9, 450 * width / LUMENS_PER_WATT, occ)
            n += 1
        return n

    def structural_objects(self):
        """Walls / slabs / the loose root geometry: 'empty' surfaces when judging a view."""
        names = set()
        for ob, occ in self.objects:
            if ob.type != "MESH":
                continue
            if occ["def"] == 0:
                names.add(ob.name)
                continue
            corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
            dims = sorted(max(c[i] for c in corners) - min(c[i] for c in corners) for i in range(3))
            label = (self.data["definitions"][occ["def"]]["name"] + " " + " ".join(occ["path"])).lower()
            # Walls/slabs, and doors: a door filling the frame is not a good subject.
            if (dims[2] > 2.3 and dims[0] < 0.35) or re.search(r"door|shutter|frame|architrave", label):
                names.add(ob.name)
        return names

    def score_cameras(self, cams, aspect, nx=24, ny=14):
        """Judge candidate views by ray casting through their actual frames.

        Rewards many distinct objects/materials in view and something interesting in the
        centre; penalises obstructions at the lens, bare structure and looking into the void.
        Returns [(score, set_of_object_names)].
        """
        scene = bpy.context.scene
        if not hasattr(self, "_structural"):
            self._structural = self.structural_objects()
        self.apply_view_visibility({})
        dg = bpy.context.evaluated_depsgraph_get()
        out = []
        for cam in cams:
            eye = Vector(cam["eye"])
            fwd = (Vector(cam["target"]) - eye).normalized()
            right = fwd.cross(Vector((0, 0, 1))).normalized()
            up = right.cross(fwd)
            tv = math.tan(math.radians(cam["fov"]) / 2)
            th = tv * aspect
            n = near = void = plain = centre_n = centre_hit = low_n = low_close = 0
            objs, mats = {}, set()
            for iy in range(ny):
                y = (iy + 0.5) / ny * 2 - 1
                for ix in range(nx):
                    x = (ix + 0.5) / nx * 2 - 1
                    d = (fwd + right * (x * th) + up * (y * tv)).normalized()
                    hit, loc, _, idx, ob, _ = scene.ray_cast(dg, eye, d, distance=40.0)
                    n += 1
                    central = abs(x) < 0.5 and abs(y) < 0.6
                    centre_n += central
                    if not hit:
                        void += 1
                        continue
                    depth = (loc - eye).dot(fwd)
                    if depth < 0.9:
                        near += 1
                    if y < -0.33:          # lower third: big furniture right in front crowds the shot
                        low_n += 1
                        low_close += depth < 1.5
                    name = ob.name
                    if name in self._structural or name.startswith("Ceiling"):
                        plain += 1
                    else:
                        objs[name] = objs.get(name, 0) + 1
                        centre_hit += central
                    try:
                        m = ob.data.materials[ob.data.polygons[idx].material_index]
                        mats.add(m.name if m else "")
                    except (IndexError, AttributeError):
                        pass
            # Objects covering at least 2 rays count as 'in the shot'.
            seen = {k for k, v in objs.items() if v >= 2}
            score = (1.0 * min(len(seen) / 10, 1.0) + 0.5 * min(len(mats) / 10, 1.0)
                     + 0.7 * centre_hit / max(centre_n, 1)
                     - 1.5 * near / n - 0.8 * max(0.0, plain / n - 0.6) - 0.5 * void / n
                     - 0.6 * max(0.0, low_close / max(low_n, 1) - 0.3))
            out.append((score, seen))
        return out

    def add_ceilings(self, rooms):
        """Models often omit ceilings; a photoreal interior needs one (plain white paint)."""
        n = 0
        mat = None
        for c in rooms:
            if not c["faces"]:
                continue
            if mat is None:
                mat = matlib.build_material({"name": "Auto Ceiling", "color": [0.93, 0.93, 0.91],
                                             "opacity": 1.0, "texture": None}, self.pkg)
            me = bpy.data.meshes.new(f"ceiling_{c['room']}")
            me.from_pydata(c["verts"], [], c["faces"])
            me.materials.append(mat)
            ob = bpy.data.objects.new(f"Ceiling {c['room']}", me)
            self.coll.objects.link(ob)
            n += 1
        return n

    def apply_view_visibility(self, view):
        """Apply a SketchUp scene's hidden objects / hidden layers (model state if not stored)."""
        if "hidden_layers" in view:
            hidden_layers = set(view["hidden_layers"])
        else:
            hidden_layers = {l["index"] for l in self.data["layers"] if not l["visible"]}
        hidden_eids = set(view["hidden_eids"]) if "hidden_eids" in view else None
        shown = 0
        for ob, occ in self.objects:
            if hidden_eids is not None:
                hid = any(e in hidden_eids for e in occ["eids"])
            else:
                hid = occ["model_hidden"]
            hid = hid or bool(hidden_layers.intersection(occ["layers"]))
            ob.hide_render = hid
            ob.hide_viewport = hid  # keeps ray_cast (fill-light probe) consistent
            shown += not hid
        return shown


def kelvin_to_rgb(k):
    """Blackbody colour on the Planckian locus (Kim et al. fit), as linear Rec.709, max = 1.

    Must match the locus Blender's white balance uses; the old sRGB-curve approximation was
    low in green, and white-balancing it turned warm-lit white walls pink.
    """
    t = max(1667.0, min(25000.0, float(k)))
    if t <= 4000:
        x = -0.2661239e9 / t ** 3 - 0.2343589e6 / t ** 2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t ** 3 + 2.1070379e6 / t ** 2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    X, Y, Z = x / y, 1.0, (1 - x - y) / y
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    rgb = [max(v, 0.0) for v in (r, g, b)]
    m = max(rgb)
    return tuple(v / m for v in rgb)


# --------------------------------------------------------------------------- cameras
def make_camera(name, cam, aspect):
    cd = bpy.data.cameras.new(name)
    eye, tgt, up = Vector(cam["eye"]), Vector(cam["target"]), Vector(cam["up"])
    fwd = (tgt - eye).normalized()
    right = fwd.cross(up).normalized()
    true_up = right.cross(fwd).normalized()
    rot = Matrix((right, true_up, -fwd)).transposed()
    ob = bpy.data.objects.new(name, cd)
    ob.matrix_world = Matrix.Translation(eye) @ rot.to_4x4()
    cd.clip_start = 0.02
    cd.clip_end = 2000
    if cam.get("perspective", True):
        fov = math.radians(cam.get("fov") or 35.0)
        scale2d = cam.get("scale_2d") or 1.0
        if cam.get("fov_is_height", True):
            cd.sensor_fit = "VERTICAL"
            half = math.tan(fov / 2) / scale2d
        else:
            cd.sensor_fit = "HORIZONTAL"
            half = math.tan(fov / 2) / scale2d
        cd.angle = 2 * math.atan(half)
        c2 = cam.get("center_2d")
        cd.angle = 2 * math.atan(math.tan(fov / 2) * scale2d)  # 2D zoom (verified vs SketchUp)
        if c2 and cd.sensor_fit == "VERTICAL":
            # Two-point perspective pan: SketchUp's center_2d is a fraction of image height
            # (verified against SketchUp exports); Blender shift is a fraction of the larger side.
            k = 1.0 / aspect if aspect >= 1 else 1.0
            cd.shift_x = c2[0] * k
            cd.shift_y = c2[1] * k
    else:
        cd.type = "ORTHO"
        cd.ortho_scale = cam.get("ortho_height", 10.0) * aspect
    bpy.context.scene.collection.objects.link(ob)
    return ob


def auto_near_clip(scene, cam_ob, max_wall=0.6):
    """Designers often park the camera inside a wall; SketchUp's near plane hides that wall.

    Cast rays across the frustum: if the first thing hit close by is a back face (we're
    inside/behind geometry), move the near clip plane just past it.
    """
    dg = bpy.context.evaluated_depsgraph_get()
    mw = cam_ob.matrix_world
    eye = mw.to_translation()
    cd = cam_ob.data
    frame = cd.view_frame(scene=scene)
    dirs = []
    for u in (0.1, 0.3, 0.5, 0.7, 0.9):
        for v in (0.1, 0.3, 0.5, 0.7, 0.9):
            p_cam = frame[2].lerp(frame[3], u).lerp(frame[1].lerp(frame[0], u), v)
            dirs.append((mw.to_3x3() @ p_cam).normalized())
    clip = cd.clip_start
    view_axis = mw.to_3x3() @ Vector((0, 0, -1))
    # 1) Centre of frame blocked right at the lens: peel that surface and anything packed
    #    behind it (door frames, wall layers) until a real gap opens up.
    centre = [d for i, d in enumerate(dirs) if i in (6, 7, 8, 11, 12, 13, 16, 17, 18)]
    peel = clip
    for d in centre:
        t_axis = d.dot(view_axis)
        depth, first = clip, True
        for _ in range(8):
            hit, loc, *_ = scene.ray_cast(dg, eye + d * (depth / t_axis + 1e-4), d, distance=max_wall)
            if not hit:
                break
            hd = (loc - eye).dot(view_axis)
            if (first and hd > 0.12) or (not first and hd - depth > 0.35):
                break
            depth, first = hd, False
        if not first:
            peel = max(peel, depth + 0.01)
    clip = peel
    # 2) Peel layers when (nearly) the whole frame is blocked close to the lens.
    for _ in range(4):
        close, backface = [], False
        for d in dirs:
            # Ray length along d that corresponds to the planar near clip distance.
            t0 = clip / max(d.dot(mw.to_3x3() @ Vector((0, 0, -1))), 1e-3)
            hit, loc, nrm, *_ = scene.ray_cast(dg, eye + d * t0, d, distance=max_wall)
            if hit:
                depth = (loc - eye).dot(mw.to_3x3() @ Vector((0, 0, -1)))
                if depth < clip + 0.25:
                    close.append(depth)
        # Nobody frames a view of a surface a few cm away: if nearly the whole frame is
        # blocked, clip past it. (Face orientation is unreliable in real models, so it isn't used.)
        if len(close) >= 0.8 * len(dirs):
            clip = max(close) + 0.01
        else:
            break
    cd.clip_start = clip
    return clip


def room_fill_light(scene, cam_ob, strength):
    """Soft ceiling fill sized to the room around the camera (hidden from camera & reflections)."""
    dg = bpy.context.evaluated_depsgraph_get()
    fwd = -(cam_ob.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    # Probe the room from open space in front of the lens (the camera may sit inside a wall):
    # halfway to the first surface past the near plane, at most 1.5 m ahead.
    start = cam_ob.matrix_world.to_translation() + fwd * (cam_ob.data.clip_start + 0.01)
    hit, loc, *_ = scene.ray_cast(dg, start, fwd, distance=3.0)
    ahead = min(((loc - start).length * 0.5) if hit else 1.5, 1.5)
    eye = start + fwd * ahead

    def cast(d, maxd=30.0):
        hit, loc, *_ = scene.ray_cast(dg, eye, Vector(d), distance=maxd)
        return (loc - eye).length if hit else None

    up = cast((0, 0, 1), 8.0)
    ceiling = eye.z + (up if up else 1.2)
    ext = {}
    for key, d in (("px", (1, 0, 0)), ("nx", (-1, 0, 0)), ("py", (0, 1, 0)), ("ny", (0, -1, 0))):
        ext[key] = min(cast(d) or 4.0, 8.0)
    cx = eye.x + (ext["px"] - ext["nx"]) / 2
    cy = eye.y + (ext["py"] - ext["ny"]) / 2
    sx = max((ext["px"] + ext["nx"]) * 0.7, 0.5)
    sy = max((ext["py"] + ext["ny"]) * 0.7, 0.5)
    ld = bpy.data.lights.new("RoomFill", "AREA")
    ld.shape = "RECTANGLE"
    ld.size, ld.size_y = sx, sy
    ld.energy = strength * sx * sy
    ld.color = kelvin_to_rgb(3800.0)
    ob = bpy.data.objects.new("RoomFill", ld)
    ob.location = (cx, cy, ceiling - 0.08)
    ob.visible_camera = False
    ob.visible_glossy = False
    scene.collection.objects.link(ob)
    # Bounce fill: a large soft omni light mid-room lifts the ceiling and upper walls, the way
    # real interiors are lit by light bouncing off light floors and walls.
    bl = bpy.data.lights.new("BounceFill", "POINT")
    bl.shadow_soft_size = 0.7
    bl.energy = strength * sx * sy * 1.6
    bl.color = kelvin_to_rgb(3800.0)
    bob = bpy.data.objects.new("BounceFill", bl)
    bob.location = (cx, cy, max(ceiling - 1.0, eye.z))
    bob.visible_camera = False
    bob.visible_glossy = False
    scene.collection.objects.link(bob)
    # Photographer's fill: soft light from just above/behind the lens so no view is black.
    cf = bpy.data.lights.new("CameraFill", "AREA")
    cf.shape = "DISK"
    cf.size = 1.2
    cf.energy = strength * 3.0
    cf.color = kelvin_to_rgb(4200.0)
    cob = bpy.data.objects.new("CameraFill", cf)
    up = cam_ob.matrix_world.to_3x3() @ Vector((0, 1, 0))
    cob.matrix_world = Matrix.Translation(start + fwd * 0.05 + up * 0.15) @ \
        cam_ob.matrix_world.to_3x3().to_4x4()
    cob.visible_camera = False
    cob.visible_glossy = False
    scene.collection.objects.link(cob)
    return [ob, cob, bob]


def save_editable_blend(b, views, opts):
    """Self-contained .blend (textures packed, one camera per view) for manual touch-ups."""
    scene = bpy.context.scene
    b.apply_view_visibility({})
    cams = [make_camera(v["name"], v["camera"], v["camera"].get("aspect") or opts.aspect) for v in views]
    if cams:
        scene.camera = cams[0]
    try:
        bpy.ops.file.pack_all()
    except RuntimeError as e:
        log(f"could not pack textures: {e}")
    path = os.path.join(opts.out, "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
    log(f"saved editable scene: {path}")
    for c in cams:
        bpy.data.objects.remove(c, do_unlink=True)


def setup_bloom(scene):
    """Subtle glow around bright light sources (lamps, strips), as a camera lens produces."""
    try:
        tree = bpy.data.node_groups.new("Bloom", "CompositorNodeTree")
        scene.compositing_node_group = tree
    except (AttributeError, TypeError):
        return False
    try:
        nodes, links = tree.nodes, tree.links
        rl = nodes.new("CompositorNodeRLayers")
        glare = nodes.new("CompositorNodeGlare")
        out = nodes.new("NodeGroupOutput")
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        for name, value in (("Type", "Bloom"), ("Quality", "High")):
            if name in glare.inputs:
                glare.inputs[name].default_value = value
        for name, value in (("Threshold", 3.0), ("Strength", 0.25), ("Size", 0.5)):
            if name in glare.inputs:
                glare.inputs[name].default_value = value
        links.new(rl.outputs["Image"], glare.inputs["Image"])
        links.new(glare.outputs["Image"], out.inputs[0])
        return True
    except Exception as e:  # compositor API differs between Blender versions; bloom is optional
        log(f"bloom disabled: {e}")
        scene.compositing_node_group = None
        return False


# --------------------------------------------------------------------------- exposure
def auto_exposure(scene, target, tmpdir):
    """Quick low-res render, measure log-average luminance, set exposure."""
    r, c = scene.render, scene.cycles
    saved = (r.resolution_percentage, c.samples, r.image_settings.file_format, r.filepath)
    vs = scene.view_settings
    vs.exposure = 0.0
    r.resolution_percentage = 20
    c.samples = 24
    r.image_settings.file_format = "OPEN_EXR"
    path = os.path.join(tmpdir, f"_exposure_{os.getpid()}.exr")  # unique per parallel process
    r.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)[:, :3]
    bpy.data.images.remove(img)
    px = px[np.isfinite(px).all(1)]
    lum = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lo, hi = np.percentile(lum, [5, 97]) if len(lum) else (0.0, 1.0)
    sel = (lum >= lo) & (lum <= hi)
    core = lum[sel]
    key = float(np.exp(np.mean(np.log(core + 1e-5)))) if len(core) else 0.18
    ev = math.log2(target / max(key, 1e-6))
    ev = max(-6.0, min(10.0, ev))
    vs.exposure = ev
    if hasattr(vs, "white_balance_temperature"):
        # Auto white balance: estimate the view's colour temperature (grey-world, McCamy) and
        # correct most of it, keeping a touch of warmth that interiors are expected to have.
        mean = px[sel].mean(0) if sel.any() else np.ones(3)
        X, Y, Z = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ mean
        s = X + Y + Z
        if s > 1e-9:
            x, y = X / s, Y / s
            n = (x - 0.3320) / (0.1858 - y)
            cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
            cct = max(2500.0, min(12000.0, cct))
            # Correct only part of the cast and keep interiors warm (client expectation);
            # strongly blue daylight views are still pulled back toward neutral.
            vs.white_balance_temperature = max(4600.0, min(6500.0, 0.35 * cct + 0.65 * 5200.0))
    r.resolution_percentage, c.samples, r.image_settings.file_format, r.filepath = saved
    return key, ev, getattr(vs, "white_balance_temperature", 6500)


# --------------------------------------------------------------------------- main
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--aspect", type=float, default=16 / 9)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--scenes", default="all", help="comma-separated scene names or indices, or 'all'")
    ap.add_argument("--max-views", type=int, default=0)
    ap.add_argument("--light-scale", type=float, default=1.0)
    ap.add_argument("--sun-strength", type=float, default=4.0)
    ap.add_argument("--sky-strength", type=float, default=1.0)
    ap.add_argument("--mode", choices=["day", "night"], default="day")
    ap.add_argument("--window-brightness", type=float, default=0.9,
                    help="scene-linear brightness of the exterior seen through windows, after exposure")
    ap.add_argument("--fill", type=float, default=6.0, help="room fill W/m^2 (0 disables)")
    ap.add_argument("--exposure-target", type=float, default=0.36)
    ap.add_argument("--no-fixture-lights", dest="fixture_lights", action="store_false")
    ap.add_argument("--no-model-lights", dest="model_lights", action="store_false",
                    help="ignore light data stored in the model (e.g. Enscape lights)")
    ap.add_argument("--save-blend", action="store_true")
    ap.add_argument("--auto-cameras", choices=["missing", "always", "never"], default="missing",
                    help="generate room cameras when the model has no scenes (or always)")
    ap.add_argument("--plan-only", action="store_true", help="stop after camera planning")
    ap.add_argument("--time-budget", type=float, default=0,
                    help="seconds for the whole Blender stage; per-view time limits are derived")
    ap.add_argument("--no-portals", dest="portals", action="store_false")
    ap.add_argument("--views-file", help="render exactly these planned views (parallel shard)")
    ap.add_argument("--room-light", type=float, default=3.0, help="per-room ambient ceiling light, W/m2")
    ap.add_argument("--no-room-lights", dest="room_lights", action="store_false")
    ap.add_argument("--report", default="report.json", help="report file name inside --out")
    ap.add_argument("--no-bloom", dest="bloom", action="store_false")
    ap.add_argument("--no-auto-ceiling", dest="auto_ceiling", action="store_false",
                    help="don't add ceilings to rooms the model left open")
    ap.add_argument("--flat-normals", dest="custom_normals", action="store_false")
    return ap.parse_args(argv)


def main():
    opts = parse_args()
    opts.out = os.path.abspath(opts.out)
    t0 = time.time()
    os.makedirs(opts.out, exist_ok=True)
    scene = reset_scene()
    height = int(round(opts.width / opts.aspect))
    setup_cycles(scene, opts.samples, opts.width, height)
    b = Builder(opts.package, opts)
    b.build_geometry()
    b.build_lights()
    b.build_environment(b.data.get("shadow", {}))
    if opts.mode == "day" and opts.portals:
        log(f"window portals: {b.add_window_portals()}")
    if opts.bloom:
        setup_bloom(scene)

    if opts.views_file:
        # Render shard: views (and generated ceilings) were planned by a separate process.
        with open(opts.views_file) as f:
            planned = json.load(f)
        views = planned["views"]
        if opts.auto_ceiling:
            b.add_ceilings(planned.get("ceilings", []))
        if opts.room_lights:
            b.add_room_lights(planned.get("room_lights", []), opts.room_light)
        if opts.fixture_lights:
            log(f"under-cabinet strips: {b.add_strips(planned.get('strips', []))}")
    else:
        views = [dict(s) for s in b.data["scenes"]]
        rooms = []
        b.analyze_plan()
        if opts.auto_cameras == "always" or (opts.auto_cameras == "missing" and not views):
            tp = time.time()
            auto, rooms = b.auto_views(opts, os.path.join(opts.out, "plan.png"))
            log(f"auto cameras: {len(auto)} views in {len(rooms)} rooms ({time.time() - tp:.1f}s)")
            views += auto
        if not views:
            views = [{"name": "Current view", "camera": b.data["model_camera"]}]
        for i, v in enumerate(views):
            v["index"] = i
        if opts.scenes != "all":
            want = [w.strip() for w in opts.scenes.split(",")]
            views = [v for i, v in enumerate(views) if v["name"] in want or str(i) in want]
        if opts.max_views:
            views = views[: opts.max_views]
        with open(os.path.join(opts.out, "views.json"), "w") as f:
            json.dump({"views": [{"name": v["name"], "auto": v.get("auto", False)} for v in views],
                       "rooms": rooms}, f, indent=1)
        with open(os.path.join(opts.out, "views_full.json"), "w") as f:
            plan = getattr(b, "plan", None)
            json.dump({"views": views, "rooms": rooms or (plan.room_info() if plan else []),
                       "ceilings": getattr(b, "ceiling_data", []),
                       "room_lights": getattr(b, "room_light_data", []),
                       "floor": plan.floor if plan else None, "strips": plan.strips if plan else []}, f)
        if opts.plan_only:
            if opts.save_blend:
                save_editable_blend(b, views, opts)
            log(f"planned {len(views)} views")
            return
    log(f"setup done in {time.time() - t0:.1f}s; rendering {len(views)} views at {opts.width}x{height}")
    if opts.save_blend and not opts.views_file:
        save_editable_blend(b, views, opts)

    report = []
    render_start = time.time()
    for i, v in enumerate(views):
        tv = time.time()
        if opts.time_budget:
            # Share what's left of the budget between the remaining views (~3 s overhead each).
            left = opts.time_budget - (time.time() - t0)
            scene.cycles.time_limit = max(6.0, left / (len(views) - i) - 3.0)
        # A SketchUp camera with a fixed aspect ratio defines the designer's frame: render that.
        aspect = v["camera"].get("aspect") or opts.aspect
        scene.render.resolution_y = int(round(opts.width / aspect))
        cam_ob = make_camera(v["name"], v["camera"], aspect)
        scene.camera = cam_ob
        b.apply_view_visibility(v)
        clip = auto_near_clip(scene, cam_ob)
        fill =room_fill_light(scene, cam_ob, opts.fill) if opts.fill > 0 else None
        b.backdrop.inputs["Strength"].default_value = opts.sky_strength
        key, ev, wb = auto_exposure(scene, opts.exposure_target, opts.out)
        if opts.mode == "day":
            # "Window pull": exterior brightness relative to the exposed interior.
            b.backdrop.inputs["Strength"].default_value = opts.window_brightness / (2 ** ev) / 0.75
        fname = f"{v.get('index', i):02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', v['name'])}.png"
        scene.render.filepath = os.path.join(opts.out, fname)
        bpy.ops.render.render(write_still=True)
        dt = time.time() - tv
        log(f"view {v.get('index', i)} '{v['name']}': clip={clip:.3f} key={key:.4f} ev={ev:+.2f} wb={wb:.0f}K {dt:.1f}s -> {fname}")
        report.append({"view": v["name"], "file": fname, "seconds": round(dt, 1), "exposure": round(ev, 2)})
        for ob in fill or []:
            bpy.data.objects.remove(ob, do_unlink=True)
    exr = os.path.join(opts.out, f"_exposure_{os.getpid()}.exr")
    if os.path.exists(exr):
        os.remove(exr)
    with open(os.path.join(opts.out, opts.report), "w") as f:
        json.dump({"total_seconds": round(time.time() - t0, 1), "views": report}, f, indent=1)
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
