"""Thin ctypes binding to the SketchUp C API (SketchUpAPI.dll).

Uses the DLL that ships with a SketchUp desktop install, so no separate SDK
download is needed. Only the functions the converter uses are wrapped.
"""
import ctypes
import os
from ctypes import byref, c_bool, c_double, c_int, c_size_t, c_void_p, c_char_p, c_int64, c_uint8

DEFAULT_SU_DIRS = [
    r"C:\Program Files\SketchUp\SketchUp 2025",
    r"C:\Program Files\SketchUp\SketchUp 2024",
    r"C:\Program Files\SketchUp\SketchUp 2023",
    r"C:\Program Files\SketchUp\SketchUp 2022",
]


class Ref(ctypes.Structure):
    _fields_ = [("ptr", c_void_p)]

    def valid(self):
        return bool(self.ptr)

    def key(self):
        return self.ptr or 0


class Point3D(ctypes.Structure):
    _fields_ = [("x", c_double), ("y", c_double), ("z", c_double)]

    def tuple(self):
        return (self.x, self.y, self.z)


class Color(ctypes.Structure):
    _fields_ = [("r", c_uint8), ("g", c_uint8), ("b", c_uint8), ("a", c_uint8)]


class Transformation(ctypes.Structure):
    _fields_ = [("values", c_double * 16)]


class SUError(RuntimeError):
    pass


class SketchUpAPI:
    def __init__(self, su_dir=None):
        su_dir = su_dir or os.environ.get("SKETCHUP_DIR") or next(
            (d for d in DEFAULT_SU_DIRS if os.path.exists(os.path.join(d, "SketchUpAPI.dll"))), None)
        if not su_dir:
            raise SUError("SketchUpAPI.dll not found; set SKETCHUP_DIR")
        os.add_dll_directory(su_dir)
        os.environ["PATH"] = su_dir + os.pathsep + os.environ.get("PATH", "")
        self.dll = ctypes.CDLL(os.path.join(su_dir, "SketchUpAPI.dll"))
        # Conversion functions (SUFaceToDrawingElement, ...) return refs by value.
        for name in ("SUFaceToDrawingElement", "SUGroupToDrawingElement", "SUComponentInstanceToDrawingElement",
                     "SUFaceToEntity", "SUGroupToEntity", "SUComponentInstanceToEntity",
                     "SUDrawingElementToEntity", "SUDrawingElementFromEntity"):
            getattr(self.dll, name).restype = Ref
        self.dll.SUInitialize()

    def __getattr__(self, name):
        fn = getattr(self.dll, name)

        def call(*args):
            r = fn(*args)
            if r != 0:
                raise SUError(f"{name} failed with SUResult {r}")
            return r

        setattr(self, name, call)
        return call

    def raw(self, name):
        """Call without raising (for optional queries)."""
        return getattr(self.dll, name)

    # --- helpers -------------------------------------------------------
    def string(self, getter, ref):
        s = Ref()
        self.SUStringCreate(byref(s))
        try:
            if getattr(self.dll, getter)(ref, byref(s)) != 0:
                return ""
            n = c_size_t()
            self.SUStringGetUTF8Length(s, byref(n))
            buf = ctypes.create_string_buffer(n.value + 1)
            self.SUStringGetUTF8(s, n.value + 1, buf, byref(n))
            return buf.value.decode("utf-8", "replace")
        finally:
            self.SUStringRelease(byref(s))

    def list(self, count_fn, get_fn, ref):
        n = c_size_t()
        self.__getattr__(count_fn)(ref, byref(n))
        if n.value == 0:
            return []
        arr = (Ref * n.value)()
        got = c_size_t()
        self.__getattr__(get_fn)(ref, n.value, arr, byref(got))
        return list(arr[: got.value])

    def open_model(self, path):
        model = Ref()
        status = c_int()
        self.SUModelCreateFromFileWithStatus(byref(model), path.encode("utf-8"), byref(status))
        return model

    def get_bool(self, fn, ref):
        v = c_bool()
        return v.value if getattr(self.dll, fn)(ref, byref(v)) == 0 else None

    def get_double(self, fn, ref):
        v = c_double()
        return v.value if getattr(self.dll, fn)(ref, byref(v)) == 0 else None


__all__ = ["SketchUpAPI", "Ref", "Point3D", "Color", "Transformation", "SUError",
           "byref", "c_bool", "c_double", "c_int", "c_size_t", "c_void_p", "c_char_p", "c_int64"]
