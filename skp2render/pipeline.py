"""SKP -> renders pipeline shared by the CLI (render.py) and the web server.

    convert (once)  ->  plan views (1 Blender process)  ->  render shards in parallel

Rendering runs on a shared pool of GPU "slots" (SKP_GPU_SLOTS, default = GPUs x
SKP_SLOTS_PER_GPU). Every Blender render process takes a slot, so parallel shards of one
job and several concurrent jobs all share the same hardware limit. On multi-GPU machines
each slot is pinned to its own GPU with CUDA_VISIBLE_DEVICES.

Progress is reported through on_event(kind, **data):
    "stage" (stage, message) | "plan" (views) | "view" (index, name, file, seconds) | "log" (line)
"""
import json
import os
import queue
import re
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLENDER = os.environ.get("BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
BLENDER_PY = os.environ.get(
    "BLENDER_PY", os.path.join(os.path.dirname(BLENDER), "5.1", "python", "bin", "python.exe"))
BUILD = os.path.join(ROOT, "skp2render", "blender_build.py")

QUALITY = {
    "draft": {"width": 960, "samples": 32},
    "preview": {"width": 1280, "samples": 96},
    "final": {"width": 1920, "samples": 512},
    "4k": {"width": 3840, "samples": 768},
}

DEFAULTS = {"quality": "final", "mode": "day", "scenes": "all", "auto_cameras": "missing",
            "time_budget": 0, "aspect": 16 / 9, "parallel": 0, "save_blend": False}

_VIEW_RE = re.compile(r"\[build\] view (\d+) '(.+)': .* ([\d.]+)s -> (\S+\.png)")
_TOTAL_RE = re.compile(r"rendering (\d+) views")
_PLANNED_RE = re.compile(r"planned (\d+) views")


class PipelineError(RuntimeError):
    pass


def _gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout
        return max(1, sum(1 for l in out.splitlines() if l.startswith("GPU ")))
    except (OSError, subprocess.SubprocessError):
        return 1


class SlotPool:
    """Hardware slots for Blender render processes: (slot id, gpu index)."""

    def __init__(self):
        gpus = _gpu_count()
        per_gpu = int(os.environ.get("SKP_SLOTS_PER_GPU", "2"))
        total = int(os.environ.get("SKP_GPU_SLOTS", str(gpus * per_gpu)))
        self.gpus, self.size = gpus, max(1, total)
        self.q = queue.Queue()
        for i in range(self.size):
            self.q.put((i, i % gpus))

    def acquire(self):
        return self.q.get()

    def release(self, slot):
        self.q.put(slot)


SLOTS = SlotPool()


def _blender(args, on_line, gpu=None):
    env = dict(os.environ)
    if gpu is not None and SLOTS.gpus > 1:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    proc = subprocess.Popen([BLENDER, "-b", "--factory-startup", "-P", BUILD, "--"] + args,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=env)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("[build]") or "Error" in line or "Traceback" in line:
            tail = (tail + [line])[-40:]
            on_line(line)
    return proc.wait(), tail


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
    common = ["--package", work_dir, "--out", out_dir, "--width", str(q["width"]), "--samples", str(q["samples"]),
              "--mode", opts["mode"], "--aspect", str(opts["aspect"])]
    lock = threading.Lock()

    def on_line(line):
        with lock:
            emit("log", line=line)
            m = _VIEW_RE.search(line)
            if m:
                emit("view", index=int(m.group(1)), name=m.group(2), seconds=float(m.group(3)),
                     file=os.path.basename(m.group(4)))

    # 1) Plan all views once (SketchUp scenes + automatic room cameras).
    emit("stage", stage="rendering", message="Planning views")
    slot = SLOTS.acquire()
    try:
        plan_args = ["--scenes", opts["scenes"], "--auto-cameras", opts["auto_cameras"], "--plan-only"]
        if opts.get("save_blend"):
            plan_args.append("--save-blend")
        code, tail = _blender(common + plan_args, lambda l: emit("log", line=l), slot[1])
    finally:
        SLOTS.release(slot)
    if code != 0:
        raise PipelineError("View planning failed:\n" + "\n".join(tail[-15:]))
    with open(os.path.join(out_dir, "views_full.json")) as f:
        planned = json.load(f)
    views = planned["views"]
    emit("plan", views=len(views))
    emit("stage", stage="rendering", message=f"Rendering {len(views)} views")

    # 2) Split views into shards and render them concurrently on free GPU slots.
    n_shards = opts.get("parallel") or SLOTS.size
    n_shards = max(1, min(n_shards, len(views)))
    shards = [views[i::n_shards] for i in range(n_shards)]
    budget = None
    if opts.get("time_budget"):
        budget = max(30.0, float(opts["time_budget"]) - (time.time() - t0))
    errors, reports = [], []

    def run_shard(k, shard_views):
        path = os.path.join(out_dir, f"_shard{k}.json")
        with open(path, "w") as f:
            json.dump({"views": shard_views, "ceilings": planned.get("ceilings", []),
                       "room_lights": planned.get("room_lights", []), "floor": planned.get("floor"),
                       "strips": planned.get("strips", [])}, f)
        args = common + ["--views-file", path, "--report", f"_report{k}.json"]
        if budget:
            args += ["--time-budget", str(budget)]
        slot = SLOTS.acquire()
        try:
            code, tail = _blender(args, on_line, slot[1])
        finally:
            SLOTS.release(slot)
        if code != 0:
            errors.append(f"shard {k}:\n" + "\n".join(tail[-12:]))
            return
        with open(os.path.join(out_dir, f"_report{k}.json")) as f:
            reports.append(json.load(f))
        os.remove(path)

    threads = [threading.Thread(target=run_shard, args=(k, s)) for k, s in enumerate(shards)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise PipelineError("Render stage failed:\n" + "\n\n".join(errors))
    t2 = time.time()

    all_views = sorted((v for r in reports for v in r["views"]), key=lambda v: v["file"])
    report = {"views": all_views, "rooms": planned.get("rooms", []), "shards": n_shards,
              "convert_seconds": round(t1 - t0, 1), "render_stage_seconds": round(t2 - t1, 1),
              "total_seconds": round(t2 - t0, 1), "options": opts}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    for k in range(n_shards):
        p = os.path.join(out_dir, f"_report{k}.json")
        if os.path.exists(p):
            os.remove(p)
    return report
