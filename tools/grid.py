"""Stack several comparison sheets vertically into one image.

blender -b --factory-startup -P tools/grid.py -- OUT.png row_height IMG1 IMG2 ...
"""
import sys

import bpy
import numpy as np

a = sys.argv[sys.argv.index("--") + 1:]
out, H, files = a[0], int(a[1]), a[2:]
rows = []
for f in files:
    img = bpy.data.images.load(f)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    ys = (np.arange(H) * h / H).astype(int)
    W = int(w * H / h)
    xs = (np.arange(W) * w / W).astype(int)
    rows.append(px[ys][:, xs])
width = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=1.0) for r in rows]
sep = np.ones((6, width, 4), np.float32)
stack = []
for r in reversed(rows):  # Blender images are bottom-up
    stack += [r, sep]
s = np.concatenate(stack[:-1], axis=0)
img = bpy.data.images.new("g", s.shape[1], s.shape[0], alpha=True)
img.pixels = s.ravel()
img.filepath_raw = out
img.file_format = "PNG"
img.save()
