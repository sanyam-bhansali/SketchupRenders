"""Run the web app:  .venv\\Scripts\\python -m server  [--port 8000]"""
import argparse

import uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
a = ap.parse_args()
uvicorn.run("server.app:app", host=a.host, port=a.port)
