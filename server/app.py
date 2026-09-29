"""HTTP API + web UI for SketchUp -> render jobs."""
import os
import re
import shutil

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from skp2render import pipeline
from server.jobs import JobStore, Worker

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_UPLOAD = 500 * 1024 * 1024
SERVABLE = re.compile(r"^[\w .()-]+\.(png|jpg|json|blend)$", re.I)

app = FastAPI(title="SketchUp Render")
store = JobStore()
JOB_WORKERS = int(os.environ.get("SKP_JOB_WORKERS", "2"))
workers = [Worker(store, i) for i in range(JOB_WORKERS)]


@app.on_event("startup")
def _start_workers():
    for w in workers:
        if not w.is_alive():
            w.start()


@app.get("/api/system")
def system():
    return {"gpus": pipeline.SLOTS.gpus, "gpu_slots": pipeline.SLOTS.size, "job_workers": JOB_WORKERS}


def _public(job):
    job = dict(job)
    if job.get("status") != "failed":
        job.pop("error", None)
    return job


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), quality: str = Form("final"), mode: str = Form("day"),
                     auto_cameras: str = Form("missing"), time_budget: float = Form(0),
                     save_blend: bool = Form(False)):
    if not file.filename.lower().endswith(".skp"):
        raise HTTPException(400, "Please upload a SketchUp .skp file")
    if quality not in pipeline.QUALITY or mode not in ("day", "night") or \
            auto_cameras not in ("missing", "always", "never"):
        raise HTTPException(400, "Invalid options")
    options = {"quality": quality, "mode": mode, "auto_cameras": auto_cameras, "time_budget": time_budget,
               "save_blend": save_blend}
    job_id = store.create(os.path.basename(file.filename), options)
    dest = os.path.join(store.job_dir(job_id), "input.skp")
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                shutil.rmtree(store.job_dir(job_id), ignore_errors=True)
                store.update(job_id, status="failed", error="File too large")
                raise HTTPException(413, "File too large (max 500 MB)")
            f.write(chunk)
    with open(dest, "rb") as f:
        if b"S\x00k\x00e\x00t\x00c\x00h\x00U\x00p\x00" not in f.read(64):
            store.update(job_id, status="failed", stage="failed", message="Not a SketchUp file",
                         error="The uploaded file is not a SketchUp model")
            raise HTTPException(400, "Not a SketchUp model")
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs():
    return [_public(j) for j in store.list()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job = _public(job)
    rdir = os.path.join(store.job_dir(job_id), "renders")
    job["has_plan"] = os.path.exists(os.path.join(rdir, "plan.png"))
    job["has_blend"] = os.path.exists(os.path.join(rdir, "scene.blend"))
    return job


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    if not re.fullmatch(r"[0-9a-f]{12}", job_id) or not SERVABLE.match(name):
        raise HTTPException(404)
    path = os.path.join(store.job_dir(job_id), "renders", name)
    if not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path)


# ---------------------------------------------------------------- 2D plan (DWG) -> 3D SketchUp
PLANS_DIR = os.path.join(os.path.dirname(store.job_dir("x")), "..", "plans")
PLAN_FILES = re.compile(r"^option\d+\.(skp|png)$|^result\.json$")


@app.post("/api/plans")
async def create_plan(file: UploadFile = File(...), wall_height: float = Form(3.0)):
    """Convert a DWG/DXF floor plan into editable SketchUp models (one per layout option)."""
    import uuid
    from starlette.concurrency import run_in_threadpool
    from plan2skp.service import process
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in (".dwg", ".dxf"):
        raise HTTPException(400, "Please upload an AutoCAD .dwg (or .dxf) floor plan")
    if not 2.4 <= wall_height <= 4.5:
        raise HTTPException(400, "Wall height must be between 2.4 and 4.5 m")
    plan_id = uuid.uuid4().hex[:12]
    d = os.path.abspath(os.path.join(PLANS_DIR, plan_id))
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "plan" + ext)
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    try:
        result = await run_in_threadpool(process, src, d, wall_height)
    except Exception as e:  # report conversion problems to the UI
        raise HTTPException(422, f"Could not read this drawing: {e}")
    return {"id": plan_id, "filename": os.path.basename(file.filename), **result}


@app.get("/api/plans/{plan_id}/files/{name}")
def plan_file(plan_id: str, name: str):
    if not re.fullmatch(r"[0-9a-f]{12}", plan_id) or not PLAN_FILES.match(name):
        raise HTTPException(404)
    path = os.path.abspath(os.path.join(PLANS_DIR, plan_id, name))
    if not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path, filename=name if name.endswith(".skp") else None)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
