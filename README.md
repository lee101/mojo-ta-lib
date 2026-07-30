# mojo-ta-lib

`mojo-ta-lib` is a focused port of TA-Lib's technical-analysis kernels to
[Mojo](https://www.modular.com/mojo). It provides a Python package with the
same uppercase function names, argument order, defaults, tuple outputs, and
leading-`NaN` warm-up behavior as the covered part of
[TA-Lib](https://ta-lib.org/).

The project is useful today rather than a generated API shell: all 36 exposed
functions execute compiled Mojo numerics and are parity-tested against the
real `TA-Lib` Python package (0.6.4).

## Coverage

| area | implemented functions |
| --- | --- |
| Moving averages | `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA` |
| Overlap and momentum | `BBANDS`, `RSI`, `MACD`, `MOM`, `ROC`, `ROCP`, `ROCR`, `ROCR100`, `CCI`, `WILLR` |
| Volatility | `TRANGE`, `ATR`, `NATR` |
| Rolling statistics | `STDDEV`, `VAR`, `MIN`, `MAX`, `SUM`, `CORREL` |
| Linear regression | `LINEARREG`, `LINEARREG_SLOPE`, `LINEARREG_INTERCEPT`, `LINEARREG_ANGLE`, `TSF` |
| Price transforms | `AVGPRICE`, `MEDPRICE`, `TYPPRICE`, `WCLPRICE` |
| Volume | `OBV`, `AD` |

`BBANDS` currently supports TA-Lib's default `matype=0` (SMA); other MA types
raise `NotImplementedError` instead of silently returning different results.

Not covered yet are the remaining momentum indicators (including ADX and
stochastics), SAR, Hilbert-transform indicators, candlestick pattern
recognition, arithmetic operator wrappers, the abstract API, and the streaming
API. This package does not claim full TA-Lib compatibility outside the table
above.

## Install

Pixi installs the pinned Mojo nightly, Python, NumPy, pytest, and upstream
TA-Lib used by the parity suite:

```bash
pixi install
pixi run build
pixi run test
```

The build task compiles the single Mojo compilation unit to
`dist/libmojo-ta-lib.so`.

## Usage

Use `mojo_ta_lib` where the covered TA-Lib API is needed:

```python
import numpy as np
import mojo_ta_lib as talib

close = np.array(
    [101.0, 102.5, 101.8, 104.2, 105.0, 104.4, 106.1, 107.3],
    dtype=np.float64,
)

average = talib.SMA(close, timeperiod=3)
rsi = talib.RSI(close, timeperiod=3)
macd, signal, histogram = talib.MACD(
    close, fastperiod=2, slowperiod=4, signalperiod=2
)

print(average)
print(rsi)
print(histogram)
```

Run the example after `pixi run build` with `pixi run python example.py`, or
from the repository with `PYTHONPATH=python`.

## Performance

These are real best-of-five wall-clock measurements produced by
`pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86-64, using
two-million-element contiguous float64 arrays and TA-Lib Python bindings
0.6.4.

| indicator | Mojo | TA-Lib | result |
| --- | ---: | ---: | ---: |
| SMA(30), 2M | 3.82 ms | 5.18 ms | 1.35x faster |
| EMA(30), 2M | 6.06 ms | 6.61 ms | 1.09x faster |
| RSI(14), 2M | 19.82 ms | 16.23 ms | 1.22x slower |
| MACD(12,26,9), 2M | 39.17 ms | 69.16 ms | 1.77x faster |
| BBANDS(20), 2M | 29.88 ms | 37.12 ms | 1.24x faster |
| ATR(14), 2M | 13.85 ms | 30.30 ms | 2.19x faster |
| MIN(30), 2M | 10.34 ms | 15.40 ms | 1.49x faster |
| CCI(14), 2M | 29.21 ms | 61.11 ms | 2.09x faster |
| LINEARREG(14), 2M | 7.65 ms | 38.44 ms | 5.02x faster |

Mojo is faster in eight of the nine measured cases. RSI's serial recurrence
remains slower than TA-Lib and is reported as such. The
benchmark includes Python allocation and FFI overhead on both sides and prints
a fresh Markdown table when rerun.

No GPU path is included or benchmarked.

## How it works

The Python layer converts each input to a one-dimensional, C-contiguous
`float64` NumPy array only when necessary. Output arrays and scratch buffers
are allocated by NumPy. Their addresses cross `ctypes` as 64-bit integers in
one call per indicator; the Mojo C-ABI wrapper reconstructs
`UnsafePointer[Float64, AnyOrigin[mut=True]]` values and writes directly into
those buffers.

All arrays use ordinary row-major contiguous memory. The Mojo shared library
does not allocate or retain heap memory, and no per-element calls cross the
FFI. Only each indicator's short warm-up prefix is initialized to `NaN`;
valid output is written directly by Mojo. MACD reuses its output buffers for
intermediate EMAs, while CCI receives one caller-owned typical-price buffer.

CCI's typical-price transform and deviation pass use native-width float64
SIMD with scalar remainder loops. Large CCI and rolling-extrema inputs are
split into independent chunks with at most 16 workers; inputs below 262,144
elements stay serial to avoid thread-launch overhead.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task holds a machine-wide file lock to avoid overlapping factory
jobs. Tests compare every exposed indicator numerically with upstream,
including output shape and warm-up `NaN` placement.

## License

MIT
