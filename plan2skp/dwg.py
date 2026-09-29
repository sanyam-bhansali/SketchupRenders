"""DWG -> DXF conversion using AutoCAD's headless Core Console (installed with AutoCAD).

Falls back to the ODA File Converter if AutoCAD isn't available. The converted DXF is cached
next to the DWG (re-used while the DWG is unchanged).
"""
import glob
import os
import subprocess
import tempfile

ACCORE = sorted(glob.glob(r"C:\Program Files\Autodesk\AutoCAD 20*\accoreconsole.exe"))
ODA = sorted(glob.glob(r"C:\Program Files\ODA\ODAFileConverter*\ODAFileConverter.exe"))


def to_dxf(dwg_path, out_dir=None):
    dwg_path = os.path.abspath(dwg_path)
    out_dir = out_dir or os.path.dirname(dwg_path)
    dxf = os.path.join(out_dir, os.path.splitext(os.path.basename(dwg_path))[0] + ".dxf")
    if os.path.exists(dxf) and os.path.getmtime(dxf) >= os.path.getmtime(dwg_path):
        return dxf
    if ACCORE:
        script = os.path.join(tempfile.gettempdir(), "skp_to_dxf.scr")
        with open(script, "w") as f:
            f.write("_.FILEDIA\n0\n_.DXFOUT\n\"" + dxf.replace("\\", "/") + "\"\n_V\n_2018\n16\n_.QUIT\n_Y\n")
        subprocess.run([ACCORE[-1], "/i", dwg_path, "/s", script, "/l", "en-US"],
                       capture_output=True, timeout=600)
    elif ODA:
        subprocess.run([ODA[-1], os.path.dirname(dwg_path), out_dir, "ACAD2018", "DXF", "0", "1",
                        os.path.basename(dwg_path)], capture_output=True, timeout=600)
    if not os.path.exists(dxf):
        raise RuntimeError("DWG conversion failed: install AutoCAD or the ODA File Converter")
    return dxf
