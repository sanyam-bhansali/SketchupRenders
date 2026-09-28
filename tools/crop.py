"""Crop the same normalised region from a reference and a render, side by side.

blender -b --factory-startup -P tools/crop.py -- REF.png RENDER.png OUT.png x0 y0 x1 y1   (0..1, top-left origin)
"""
import sys

import bpy
import numpy as np

a = sys.argv[sys.argv.index("--") + 1:]
ref, ren, out = a[:3]
x0, y0, x1, y1 = map(float, a[3:7])
H = 400


def crop(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1]  # top-left origin
    c = px[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    ys = (np.arange(H) * c.shape[0] / H).astype(int)
    W = int(H * c.shape[1] / c.shape[0])
    xs = (np.arange(W) * c.shape[1] / W).astype(int)
    return c[ys][:, xs]


s = np.concatenate([crop(ref), np.ones((H, 6, 4), np.float32), crop(ren)], axis=1)[::-1]
img = bpy.data.images.new("c", s.shape[1], s.shape[0], alpha=True)
img.pixels = s.ravel()
img.filepath_raw = out
img.file_format = "PNG"
img.save()
