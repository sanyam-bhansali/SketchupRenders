"""One call for the web app: DWG/DXF in -> every layout option as .skp + preview PNG."""
import json
import os
import time

from plan2skp.build import build
from plan2skp.dwg import to_dxf
from plan2skp.extract import extract
from plan2skp.preview import render


def process(src_path, out_dir, wall_height=3.0):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    dxf = to_dxf(src_path, out_dir) if src_path.lower().endswith(".dwg") else src_path
    t1 = time.time()
    plans = extract(dxf)
    options = []
    for i, p in enumerate(plans):
        names = sorted({r["name"] for r in p["rooms"] if r["name"]})
        skp = f"option{i + 1}.skp"
        png = f"option{i + 1}.png"
        render(p, os.path.join(out_dir, png), f"Option {i + 1}")
        info = build(p, os.path.join(out_dir, skp), wall_height)
        options.append({"option": i + 1, "skp": skp, "preview": png, "size_m": [round(v, 1) for v in p["size_m"]],
                        "rooms": len(p["rooms"]), "room_names": names, "doors": len(p["doors"]),
                        "windows": len(p.get("windows", [])), "lintels": info["lintels"]})
    result = {"options": options, "dwg_to_dxf_seconds": round(t1 - t0, 1),
              "total_seconds": round(time.time() - t0, 1), "wall_height_m": wall_height}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    return result
