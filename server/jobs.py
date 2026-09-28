"""Job store (SQLite) and the render worker (one job at a time per GPU)."""
import json
import os
import sqlite3
import threading
import time
import traceback
import uuid

from skp2render import pipeline

DATA_DIR = os.environ.get("SKP_DATA", os.path.join(pipeline.ROOT, "data"))
JOBS_DIR = os.path.join(DATA_DIR, "jobs")

FIELDS = ["id", "filename", "created", "status", "stage", "message", "progress", "total", "done",
          "options", "views", "report", "error", "started", "finished"]
JSON_FIELDS = {"options", "views", "report"}


class JobStore:
    def __init__(self, path=None):
        os.makedirs(JOBS_DIR, exist_ok=True)
        self.db = sqlite3.connect(path or os.path.join(DATA_DIR, "jobs.db"), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, filename TEXT, created REAL, status TEXT,"
                " stage TEXT, message TEXT, progress REAL, total INTEGER, done INTEGER, options TEXT,"
                " views TEXT, report TEXT, error TEXT, started REAL, finished REAL)")
            # Jobs interrupted by a server restart can't resume mid-render; mark them failed.
            self.db.execute("UPDATE jobs SET status='failed', error='Interrupted by server restart' "
                            "WHERE status IN ('converting', 'rendering')")
            self.db.commit()

    def job_dir(self, job_id):
        return os.path.join(JOBS_DIR, job_id)

    def create(self, filename, options):
        job_id = uuid.uuid4().hex[:12]
        os.makedirs(self.job_dir(job_id), exist_ok=True)
        with self.lock:
            self.db.execute(
                "INSERT INTO jobs (id, filename, created, status, stage, message, progress, total, done,"
                " options, views) VALUES (?, ?, ?, 'queued', 'queued', 'Waiting for GPU', 0, 0, 0, ?, '[]')",
                (job_id, filename, time.time(), json.dumps(options)))
            self.db.commit()
        return job_id

    def update(self, job_id, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = [json.dumps(v) if k in JSON_FIELDS else v for k, v in fields.items()]
        with self.lock:
            self.db.execute(f"UPDATE jobs SET {cols} WHERE id=?", vals + [job_id])
            self.db.commit()

    def _row(self, row):
        if row is None:
            return None
        d = dict(zip(FIELDS, row))
        for k in JSON_FIELDS:
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    def get(self, job_id):
        with self.lock:
            row = self.db.execute(f"SELECT {', '.join(FIELDS)} FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(row)

    def list(self, limit=50):
        with self.lock:
            rows = self.db.execute(f"SELECT {', '.join(FIELDS)} FROM jobs ORDER BY created DESC LIMIT ?",
                                   (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def next_queued(self):
        with self.lock:
            row = self.db.execute(f"SELECT {', '.join(FIELDS)} FROM jobs WHERE status='queued' "
                                  "ORDER BY created LIMIT 1").fetchone()
        return self._row(row)


class Worker(threading.Thread):
    def __init__(self, store):
        super().__init__(daemon=True, name="render-worker")
        self.store = store

    def run(self):
        while True:
            job = self.store.next_queued()
            if job is None:
                time.sleep(1.0)
                continue
            self.process(job)

    def process(self, job):
        store, jid = self.store, job["id"]
        d = store.job_dir(jid)
        views = []
        state = {"total": 0}
        store.update(jid, status="converting", stage="converting", message="Reading SketchUp model",
                     progress=0.02, started=time.time())

        def on_event(kind, **e):
            if kind == "stage":
                fields = {"stage": e["stage"], "message": e["message"]}
                if e["stage"] == "rendering" and not views:
                    fields.update(status="rendering", progress=0.1)
                store.update(jid, **fields)
            elif kind == "plan":
                state["total"] = e["views"]
                store.update(jid, total=e["views"], progress=0.12,
                             message=f"Rendering {e['views']} views")
            elif kind == "view":
                views.append({"name": e["name"], "file": e["file"], "seconds": e["seconds"]})
                total = max(state["total"], len(views))
                store.update(jid, views=views, done=len(views), progress=0.12 + 0.88 * len(views) / total,
                             message=f"Rendered {len(views)} of {total}: {e['name']}")

        try:
            report = pipeline.run(os.path.join(d, "input.skp"), os.path.join(d, "renders"),
                                  os.path.join(d, "work"), job["options"], on_event)
            store.update(jid, status="done", stage="done", progress=1.0, report=report,
                         message=f"{len(views)} renders in {report['total_seconds']:.0f}s", finished=time.time())
        except Exception as e:  # report any failure to the UI instead of killing the worker
            msg = str(e) if isinstance(e, pipeline.PipelineError) else traceback.format_exc()
            store.update(jid, status="failed", stage="failed", message="Failed", error=msg[-4000:],
                         finished=time.time())
