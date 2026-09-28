"""Side-by-side SketchUp reference vs render sheets.

blender -b --factory-startup -P tools/compare.py -- REF_DIR RENDER_DIR OUT_DIR [height]
Pairs files by scene name (the part after the NN_ index prefix).
"""
import os
import re
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
ref_dir, ren_dir, out_dir = args[:3]
H = int(args[3]) if len(args) > 3 else 450
os.makedirs(out_dir, exist_ok=True)


def key(f):
    return re.sub(r"^\d+_", "", os.path.splitext(f)[0]).lower()


def load(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    ys = (np.arange(H) * h / H).astype(int)
    W = int(round(w * H / h))
    xs = (np.arange(W) * w / W).astype(int)
    return px[ys][:, xs]


refs = {key(f): f for f in os.listdir(ref_dir) if f.endswith(".png")}
for f in sorted(os.listdir(ren_dir)):
    if not f.endswith(".png") or key(f) not in refs:
        continue
    a, b = load(os.path.join(ref_dir, refs[key(f)])), load(os.path.join(ren_dir, f))
    gap = np.ones((H, 8, 4), np.float32)
    sheet = np.concatenate([a, gap, b], axis=1)
    out = bpy.data.images.new("sheet", sheet.shape[1], sheet.shape[0], alpha=True)
    out.pixels = sheet.ravel()
    out.filepath_raw = os.path.join(out_dir, f)
    out.file_format = "PNG"
    out.save()
    bpy.data.images.remove(out)
    print("sheet", f)
