# SketchUp Render Studio

Upload a SketchUp `.skp` and get photoreal renders of every scene. Models without scenes get
automatic room-by-room cameras. The pipeline uses the SketchUp C API to read the model and
Blender Cycles to render.

## Run
```
.venv\Scripts\python -m server            # web app on http://127.0.0.1:8000
python render.py "model.skp" --quality final --time-budget 300    # command line
```
Web app: drag in a `.skp`, pick quality, day or night, cameras and a time limit. Progress is live,
renders show up in the gallery as each finishes, and the detected floor plan and rooms are shown.

Quality presets: `draft` 960px/32spp, `preview` 1280px/96spp, `final` 1920px/512spp, `4k`.

## Layout
```
skp2render/
  sketchup_api.py   ctypes binding to SketchUpAPI.dll (ships with SketchUp desktop)
  convert.py        .skp -> scene.json + geometry.npz + textures (instanced, metres)
  cameras.py        room detection + automatic camera placement (pure numpy)
  materials.py      SketchUp material -> PBR shader (class heuristics, bevel, bump)
  blender_build.py  Blender stage: scene, lights, sky, portals, cameras, exposure, render
  pipeline.py       convert + render with progress events (used by CLI and server)
server/
  app.py            FastAPI: POST/GET /api/jobs, file serving, static UI
  jobs.py           SQLite job store + GPU worker thread
  static/index.html upload / progress / gallery UI
render.py           CLI
tools/              SketchUp reference export + side-by-side comparison tools
data/jobs/<id>/     uploads, intermediate package and renders per job
```

## Automatic room cameras (`cameras.py`)
1. Find the floor level: the lowest horizontal level with a large share of the area and walls standing on it.
2. Slice the model at eye height (1.3 m) and at 2.2 m (above door heads, so doorways close) and rasterise the cuts on a 6 cm grid.
3. Flood-fill the free space. Regions touching the border are outside; the rest are rooms.
   Wall cavities, regions under 2 m² and empty rooms are dropped.
4. Candidate positions sit 0.35–0.9 m from the walls, with a view direction every 10°. Each is scored by
   2D ray casting on: visible area, furniture in view, **feature-wall detail** (density of cut segments),
   depth, and penalties for a wall under 1.2 m ahead or furniture right at the lens. The best view is kept,
   plus a second one looking the other way in larger rooms.
5. Rooms are named from component names (bed, sofa, WC…), then from geometry cues: a large continuous
   surface at mattress height means a bedroom, a table at 0.7–0.8 m in a large room means a living/dining room,
   a long surface at counter height means a kitchen, and a small room is a bathroom.
6. Rooms the model left open get a white ceiling at the detected wall-top height.
   `plan.png` shows the rooms and camera frustums.

Planning takes 1–2 s per apartment.

## Realism
- Physically based materials from SketchUp colour and texture, with the class inferred from the name
  (wood, stone, fabric, metal, glass…), bump from texture luminance, and 2–4 mm **rounded edges** (bevel shader).
- Enscape lights (IES spots, linear strips) are imported. Windows get **light portals** for clean daylight.
- Auto exposure, auto white balance (grey-world colour temperature estimate), AgX tone mapping, subtle bloom.
- An exterior "window pull" keeps windows bright but not blown out. The OIDN denoiser runs with accurate prefiltering.

## Speed
| Stage | Chetan (19 scenes) | 3BHK (2M tris) |
|---|---|---|
| Import (convert) | 6 s | 22 s |
| Blender setup | 2 s | 8 s |
| Per view, final 1080p on RTX 3060 Ti | ~50–70 s | |
| Per view, preview | ~9 s | ~8 s |

`--time-budget` (the time limit in the UI) spreads the remaining time across views, so a job always
finishes on time; quality scales with the budget. For the 5-minute target on large scene counts, run on a faster
cloud GPU or split views across several GPUs (future work: a multi-worker queue).

## Accuracy work (vs SketchUp's own renders of the same scenes)
Verified with `tools/export_reference.ps1` + `tools/compare.py`: framing, two-point perspective, camera
aspect, per-scene hidden objects and tags, cameras parked inside walls, colorized textures, inherited
materials, triangle winding, and normals on flat faces are all handled.

## Next steps
- Multi-GPU and cloud worker pool (the queue already separates jobs; add remote workers).
- Vision-model material classification for anonymous names such as `*57`.
- Section-plane support; replacing Enscape asset placeholders.
- Night mode: add auto ceiling lights for models without light data.
- A user-editable room name and camera picker in the UI (drag a camera on `plan.png`).
