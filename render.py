"""One-click SketchUp -> photoreal renders (command line).

    python render.py "model.skp" [--quality draft|preview|final|4k] [--scenes "A,B"]
                     [--mode day|night] [--auto-cameras missing|always|never] [--time-budget 300]

For the web app, run:  .venv\\Scripts\\python -m server
"""
import argparse
import os
import re
import sys

from skp2render import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skp")
    ap.add_argument("--out")
    ap.add_argument("--quality", choices=pipeline.QUALITY, default="final")
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--mode", choices=["day", "night"], default="day")
    ap.add_argument("--auto-cameras", choices=["missing", "always", "never"], default="missing")
    ap.add_argument("--time-budget", type=float, default=0, help="seconds for the whole job")
    ap.add_argument("--aspect", type=float, default=16 / 9)
    a = ap.parse_args()

    name = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(a.skp))[0]).strip("_")
    out = os.path.abspath(a.out or os.path.join(pipeline.ROOT, "renders", name))
    work = os.path.join(pipeline.ROOT, "work", name)

    def on_event(kind, **d):
        if kind == "stage":
            print(f"[{d['stage']}] {d['message']}")
        elif kind == "plan":
            print(f"  {d['views']} views to render")
        elif kind == "view":
            print(f"  done: {d['name']}  ({d['seconds']:.0f}s) -> {d['file']}")
        elif kind == "log" and ("auto cameras" in d["line"] or "Error" in d["line"]):
            print("   ", d["line"])

    try:
        report = pipeline.run(a.skp, out, work, {"quality": a.quality, "scenes": a.scenes, "mode": a.mode,
                                                  "auto_cameras": a.auto_cameras, "time_budget": a.time_budget,
                                                  "aspect": a.aspect}, on_event)
    except pipeline.PipelineError as e:
        sys.exit(str(e))
    print(f"done: {len(report['views'])} views in {report['total_seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
