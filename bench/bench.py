"""Benchmark Mojo kernels against TA-Lib on identical float64 arrays."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import talib

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_ta_lib as mojo_talib  # noqa: E402


def best_time(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    rng = np.random.default_rng(0)
    n = 2_000_000
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(size=n)))
    high = np.ascontiguousarray(close + rng.uniform(0.0, 3.0, n))
    low = np.ascontiguousarray(close - rng.uniform(0.0, 3.0, n))

    cases = [
        ("SMA(30), 2M", lambda: mojo_talib.SMA(close), lambda: talib.SMA(close)),
        ("EMA(30), 2M", lambda: mojo_talib.EMA(close), lambda: talib.EMA(close)),
        ("RSI(14), 2M", lambda: mojo_talib.RSI(close), lambda: talib.RSI(close)),
        (
            "MACD(12,26,9), 2M",
            lambda: mojo_talib.MACD(close),
            lambda: talib.MACD(close),
        ),
        (
            "BBANDS(20), 2M",
            lambda: mojo_talib.BBANDS(close, timeperiod=20),
            lambda: talib.BBANDS(close, timeperiod=20),
        ),
        (
            "ATR(14), 2M",
            lambda: mojo_talib.ATR(high, low, close),
            lambda: talib.ATR(high, low, close),
        ),
        (
            "MIN(30), 2M",
            lambda: mojo_talib.MIN(close),
            lambda: talib.MIN(close),
        ),
        (
            "CCI(14), 2M",
            lambda: mojo_talib.CCI(high, low, close),
            lambda: talib.CCI(high, low, close),
        ),
        (
            "LINEARREG(14), 2M",
            lambda: mojo_talib.LINEARREG(close),
            lambda: talib.LINEARREG(close),
        ),
    ]

    mojo_talib.SMA(close[:100])
    print(f"Machine: {machine_name()} ({platform.system()} {platform.machine()})")
    print(f"TA-Lib Python bindings: {talib.__version__}")
    print()
    print("| indicator | Mojo | TA-Lib | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_function, reference_function in cases:
        mojo_seconds = best_time(mojo_function)
        reference_seconds = best_time(reference_function)
        ratio = reference_seconds / mojo_seconds
        label = f"{ratio:.2f}x faster" if ratio >= 1.0 else f"{1.0 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{reference_seconds * 1e3:.2f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
