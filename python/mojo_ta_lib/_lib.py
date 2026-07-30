"""ctypes bindings for the compiled Mojo indicator library."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(
    os.environ.get("MOJO_TA_LIB_LIBRARY", ROOT / "dist" / "libmojo-ta-lib.so")
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mtl_sma": ([I, I, I, I], None),
    "mtl_ema": ([I, I, I, I], None),
    "mtl_wma": ([I, I, I, I], None),
    "mtl_dema": ([I, I, I, I, I, I], None),
    "mtl_tema": ([I, I, I, I, I, I, I], None),
    "mtl_trima": ([I, I, I, I], None),
    "mtl_rsi": ([I, I, I, I], None),
    "mtl_macd": ([I] * 8, None),
    "mtl_bbands": ([I, I, I, F, F, I, I, I], None),
    "mtl_trange": ([I, I, I, I, I], None),
    "mtl_atr": ([I, I, I, I, I, I, I], None),
    "mtl_momentum": ([I, I, I, I, I], None),
    "mtl_variance": ([I, I, I, F, I, I], None),
    "mtl_window": ([I, I, I, I, I], None),
    "mtl_price": ([I, I, I, I, I, I, I], None),
    "mtl_obv": ([I, I, I, I], None),
    "mtl_ad": ([I, I, I, I, I, I], None),
    "mtl_cci": ([I, I, I, I, I, I, I], None),
    "mtl_willr": ([I, I, I, I, I, I], None),
    "mtl_correl": ([I, I, I, I, I], None),
    "mtl_linearreg": ([I, I, I, I, I], None),
}


class LibraryNotBuiltError(RuntimeError):
    pass


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not LIB_PATH.is_file():
            raise LibraryNotBuiltError(
                f"{LIB_PATH} does not exist; run `pixi run build` first"
            )
        _library = ctypes.CDLL(str(LIB_PATH))
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("NumPy returned a null buffer address")
    return address
