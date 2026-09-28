"""SketchUp material -> physically based Cycles material.

SketchUp materials only carry colour, texture and opacity. We keep those exactly
(the designer's look) and infer the missing surface response (roughness,
reflectance, bump, sheen, transmission, emission) from a material class
guessed from the material name and texture file name.
"""
import os
import re

import bpy

# (class, regex) - first match wins; order matters.
RULES = [
    ("invisible", r"glass_inner"),
    ("mirror", r"mirror"),
    ("emissive", r"(^|[^a-z])(led|glow|emissi\w*|neon|bulb|light ?strip|cove ?light|profile ?light|mandir ?light)([^a-z]|$)"),
    ("glass", r"glass|glaz|window ?pane|translucent|transparent|\bclear\b"),
    ("metal", r"(^|[^a-z])(ss|steel|stainless|chrome|metal\w*|alumin\w*|brass|gold|copper|bronze|nickel|iron|titanium|rose ?gold)([^a-z]|$)"),
    ("water", r"water"),
    ("leather", r"leather|rexine"),
    ("fabric", r"fabric|cushion|cusion|curtain|cortain|carpet|rug|linen|velvet|cloth|textile|sofa|upholster|bed ?sheet|pillow|towel|blind|drape|jute|wool"),
    ("stone", r"marble|granite|stone|quartz|onyx|travertine|terrazzo|slate|kota|\bcorian\b"),
    ("tile", r"tile|porcelain|ceramic|vitrified|mosaic"),
    ("wood", r"wood|veneer|oak|walnut|teak|ash|maple|cherry|pine|birch|timber|plywood|mdf|parquet|floor ?board"),
    ("laminate", r"laminate|lam\b|sunmica|acrylic|lacquer|pu\b|duco|high ?gloss"),
    ("wallpaper", r"wall ?paper|wallcover"),
    ("plant", r"leaf|leaves|plant|grass|foliage|monstera|palm"),
    ("plastic", r"plastic|rubber|pvc"),
    ("concrete", r"concrete|cement|plaster|microtopping|texture paint"),
]

# Surface response per class.
PRESETS = {
    "paint":     dict(rough=0.6, bump=0.0),
    "wood":      dict(rough=0.42, bump=0.15, rough_var=0.25),
    "laminate":  dict(rough=0.35, bump=0.03),
    "stone":     dict(rough=0.12, bump=0.03, coat=0.3),
    "tile":      dict(rough=0.15, bump=0.02, coat=0.2),
    "fabric":    dict(rough=0.9, bump=0.35, sheen=0.6),
    "leather":   dict(rough=0.45, bump=0.12, sheen=0.1),
    "metal":     dict(rough=0.22, metallic=1.0),
    "mirror":    dict(rough=0.01, metallic=1.0, color=(0.93, 0.93, 0.93)),
    "wallpaper": dict(rough=0.7, bump=0.12),
    "plant":     dict(rough=0.5, bump=0.1, translucent=0.25),
    "plastic":   dict(rough=0.35),
    "concrete":  dict(rough=0.8, bump=0.25),
    "water":     dict(rough=0.02, glass=True),
    "textured":  dict(rough=0.45, bump=0.05),
}

BEVEL_CLASSES = {"paint", "wood", "laminate", "stone", "tile", "metal", "textured", "plastic", "leather"}

SKETCHUP_COLOR_NAME =re.compile(r"^\[?(\d{4}_|color )", re.I)


def classify(info):
    if info is None:
        return "paint"
    name = info["name"]
    tex = info.get("texture") or {}
    probe = (name + " " + os.path.splitext(tex.get("source_name", ""))[0]).lower().replace("_", " ")
    if info.get("opacity", 1.0) <= 0.02:
        return "invisible"
    for cls, rx in RULES:
        if cls == "metal" and SKETCHUP_COLOR_NAME.search(name):
            continue  # "[0131_Silver]" is a paint colour, not metal
        if re.search(rx, probe):
            return cls
    if info.get("opacity", 1.0) < 0.9:
        return "glass"
    return "textured" if tex else "paint"


def _principled(nt):
    return next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")


def _set(node, name, value):
    if name in node.inputs:
        node.inputs[name].default_value = value


def build_material(info, pkg_dir):
    cls = classify(info)
    name = info["name"] if info else "SketchUp Default"
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = _principled(nt)
    out = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    color = tuple(info["color"]) if info else (0.86, 0.86, 0.84)
    lin = tuple(srgb_to_linear(c) for c in color)
    mat["sketchup_class"] = cls

    if cls == "invisible":
        nodes.remove(bsdf)
        tr = nodes.new("ShaderNodeBsdfTransparent")
        links.new(tr.outputs[0], out.inputs["Surface"])
        return mat

    if cls in ("glass", "water") or PRESETS.get(cls, {}).get("glass"):
        return _thin_glass(mat, lin, info)

    p = PRESETS.get(cls, PRESETS["paint"])
    tex_node = None
    tex = info.get("texture") if info else None
    if tex:
        path = os.path.join(pkg_dir, tex["file"])
        if os.path.exists(path):
            tex_node = nodes.new("ShaderNodeTexImage")
            tex_node.image = bpy.data.images.load(path, check_existing=True)
            tex_node.interpolation = "Cubic" if min(tex["px"]) < 512 else "Linear"
            links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
            if tex.get("alpha"):
                links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])
    if tex_node is None:
        base = p.get("color", lin)
        bsdf.inputs["Base Color"].default_value = (*base, 1.0)

    _set(bsdf, "Roughness", p.get("rough", 0.5))
    _set(bsdf, "Metallic", p.get("metallic", 0.0))
    if cls == "metal" and tex_node is None and max(lin) < 0.03:
        # Near-black "metal" colours in SketchUp usually mean dark anodised / black steel.
        bsdf.inputs["Base Color"].default_value = (0.04, 0.04, 0.045, 1.0)
    if p.get("coat"):
        _set(bsdf, "Coat Weight", p["coat"])
        _set(bsdf, "Coat Roughness", 0.03)
    if p.get("sheen"):
        _set(bsdf, "Sheen Weight", p["sheen"])
        _set(bsdf, "Sheen Roughness", 0.5)
    if p.get("translucent"):
        _set(bsdf, "Subsurface Weight", p["translucent"])

    bevel = None
    if cls in BEVEL_CLASSES:
        # SketchUp edges are razor sharp; real furniture/joinery edges catch light.
        bevel = nodes.new("ShaderNodeBevel")
        bevel.samples = 6
        bevel.inputs["Radius"].default_value = 0.004 if cls != "metal" else 0.002
        links.new(bevel.outputs[0], bsdf.inputs["Normal"])
    if tex_node is not None:
        # Roughness variation and bump derived from texture luminance.
        bw = nodes.new("ShaderNodeRGBToBW")
        links.new(tex_node.outputs["Color"], bw.inputs[0])
        if p.get("rough_var"):
            mr = nodes.new("ShaderNodeMapRange")
            mr.inputs["To Min"].default_value = p["rough"] + p["rough_var"] / 2
            mr.inputs["To Max"].default_value = max(p["rough"] - p["rough_var"] / 2, 0.02)
            links.new(bw.outputs[0], mr.inputs["Value"])
            links.new(mr.outputs[0], bsdf.inputs["Roughness"])
        if p.get("bump"):
            bump = nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = p["bump"]
            bump.inputs["Distance"].default_value = 0.002
            links.new(bw.outputs[0], bump.inputs["Height"])
            if bevel is not None:
                links.new(bevel.outputs[0], bump.inputs["Normal"])
            links.new(bump.outputs[0], bsdf.inputs["Normal"])

    if cls == "emissive":
        emit_col = lin if max(lin) > 0.05 else (1.0, 0.8, 0.55)
        _set(bsdf, "Emission Color", (*emit_col, 1.0))
        _set(bsdf, "Emission Strength", 6.0)
    return mat


def _thin_glass(mat, lin, info):
    """Single-sided SketchUp glass: fresnel mix of clear transmission and reflection.

    Shadow rays see it as transparent so sunlight enters through windows.
    """
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    opacity = info.get("opacity", 0.3) if info else 0.3
    # SketchUp opacity -> per-surface transmission tint (0.06 opaque grey glass is ~95% clear;
    # windows usually have several glass surfaces in a row, so keep each one light).
    tint = tuple(max(0.0, 1.0 - min(opacity, 0.6) * 0.5 * (1.0 - c)) for c in lin)
    transp = nodes.new("ShaderNodeBsdfTransparent")
    transp.inputs["Color"].default_value = (*tint, 1)
    gloss = nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Roughness"].default_value = 0.0
    fres = nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = 1.5
    mix = nodes.new("ShaderNodeMixShader")
    links.new(fres.outputs[0], mix.inputs[0])
    links.new(transp.outputs[0], mix.inputs[1])
    links.new(gloss.outputs[0], mix.inputs[2])
    if opacity >= 0.6:
        # Mostly opaque "glass" (lacquered / back-painted glass): glossy dielectric over colour.
        diff = nodes.new("ShaderNodeBsdfPrincipled")
        diff.inputs["Base Color"].default_value = (*lin, 1)
        diff.inputs["Roughness"].default_value = 0.05
        _set(diff, "Coat Weight", 1.0)
        links.new(diff.outputs[0], out.inputs["Surface"])
        mat["sketchup_class"] = "lacquered_glass"
        return mat
    links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
