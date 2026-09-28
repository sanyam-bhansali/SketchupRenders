"""SKP -> renders pipeline shared by the CLI (render.py) and the web server.

Runs the converter with Blender's bundled Python, then the Blender/Cycles stage, and reports
progress through a callback: on_event(kind, **data).
    kind: "stage"  (stage=converting|rendering, message)
          "plan"   (views=total)
          "view"   (index, name, file, seconds)
          "log"    (line)
"""
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLENDER = os.environ.get("BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
BLENDER_PY = os.environ.get(
    "BLENDER_PY", os.path.join(os.path.dirname(BLENDER), "5.1", "python", "bin", "python.exe"))

QUALITY = {
    "draft": {"width": 960, "samples": 32},
    "preview": {"width": 1280, "samples": 96},
    "final": {"width": 1920, "samples": 512},
    "4k": {"width": 3840, "samples": 768},
}

DEFAULTS = {"quality": "final", "mode": "day", "scenes": "all", "auto_cameras": "missing",
            "time_budget": 0, "aspect": 16 / 9}

_VIEW_RE = re.compile(r"\[build\] view (\d+) '(.+)': .* ([\d.]+)s -> (\S+\.png)")
_TOTAL_RE = re.compile(r"rendering (\d+) views")


class PipelineError(RuntimeError):
    pass


def run(skp, out_dir, work_dir, options=None, on_event=None):
    opts = dict(DEFAULTS, **(options or {}))
    emit = on_event or (lambda kind, **data: None)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    emit("stage", stage="converting", message="Reading SketchUp model")
    conv = subprocess.run([BLENDER_PY, "-m", "skp2render.convert", skp, work_dir], cwd=ROOT,
                          capture_output=True, text=True, errors="replace")
    for line in conv.stdout.splitlines():
        emit("log", line=line)
    if conv.returncode != 0:
        raise PipelineError("SketchUp conversion failed:\n" + (conv.stderr or conv.stdout)[-2000:])
    t1 = time.time()

    q = QUALITY[opts["quality"]]
    emit("stage", stage="rendering", message="Building scene")
    cmd = [BLENDER, "-b", "--factory-startup", "-P", os.path.join(ROOT, "skp2render", "blender_build.py"), "--",
           "--package", work_dir, "--out", out_dir, "--width", str(q["width"]), "--samples", str(q["samples"]),
           "--scenes", opts["scenes"], "--mode", opts["mode"], "--aspect", str(opts["aspect"]),
           "--auto-cameras", opts["auto_cameras"]]
    if opts.get("time_budget"):
        # The budget covers the whole job; give Blender what conversion left over.
        cmd += ["--time-budget", str(max(30.0, float(opts["time_budget"]) - (t1 - t0)))]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if not line.startswith("[build]") and "Error" not in line and "Traceback" not in line:
            continue
        tail = (tail + [line])[-40:]
        emit("log", line=line)
        m = _TOTAL_RE.search(line)
        if m:
            emit("plan", views=int(m.group(1)))
            emit("stage", stage="rendering", message="Rendering views")
            continue
        m = _VIEW_RE.search(line)
        if m:
            emit("view", index=int(m.group(1)), name=m.group(2), seconds=float(m.group(3)),
                 file=os.path.basename(m.group(4)))
    if proc.wait() != 0:
        raise PipelineError("Render stage failed:\n" + "\n".join(tail[-15:]))
    t2 = time.time()

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path) as f:
        report = json.load(f)
    views_path = os.path.join(out_dir, "views.json")
    if os.path.exists(views_path):
        with open(views_path) as f:
            report["rooms"] = json.load(f).get("rooms", [])
    report.update({"convert_seconds": round(t1 - t0, 1), "render_stage_seconds": round(t2 - t1, 1),
                   "total_seconds": round(t2 - t0, 1), "options": opts})
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)
    return report
