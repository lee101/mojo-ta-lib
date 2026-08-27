"""A focused, TA-Lib-compatible API backed by Mojo kernels."""

from __future__ import annotations

import numpy as np

from ._lib import addr, lib

__version__ = "0.1.0"


def _real(value, name: str = "real") -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional array")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real values, not complex values")
    if np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize > 8:
        raise TypeError(f"{name} cannot be narrowed safely to float64")
    return np.ascontiguousarray(array, dtype=np.float64)


def _same_length(*arrays: np.ndarray) -> int:
    lengths = {array.size for array in arrays}
    if len(lengths) != 1:
        raise ValueError("input arrays must have the same length")
    return arrays[0].size


def _period(value: int, minimum: int = 2) -> int:
    value = int(value)
    if value < minimum or value > 100000:
        raise ValueError("timeperiod is out of range")
    return value


def _result(n: int, lookback: int = 0) -> np.ndarray:
    result = np.empty(n, dtype=np.float64)
    result[: min(n, lookback)] = np.nan
    return result


def SMA(real, timeperiod=30):
    real = _real(real)
    timeperiod = _period(timeperiod)
    result = _result(real.size, timeperiod - 1)
    lib().mtl_sma(addr(real), real.size, timeperiod, addr(result))
    return result


def EMA(real, timeperiod=30):
    real = _real(real)
    timeperiod = _period(timeperiod)
    result = _result(real.size, timeperiod - 1)
    lib().mtl_ema(addr(real), real.size, timeperiod, addr(result))
    return result


def WMA(real, timeperiod=30):
    real = _real(real)
    timeperiod = _period(timeperiod)
    result = _result(real.size, timeperiod - 1)
    lib().mtl_wma(addr(real), real.size, timeperiod, addr(result))
    return result


def DEMA(real, timeperiod=30):
    real = _real(real)
    timeperiod = _period(timeperiod)
    result = _result(real.size, 2 * (timeperiod - 1))
    work1 = np.empty(real.size, dtype=np.float64)
    work2 = np.empty(real.size, dtype=np.float64)
    lib().mtl_dema(
        addr(real), real.size, timeperiod, addr(work1), addr(work2), addr(result)
    )
    return result


def TEMA(real, timeperiod=30):
    real = _real(real)
    timeperiod = _period(timeperiod)
    result = _result(real.size, 3 * (timeperiod - 1))
    work1 = np.empty(real.size, dtype=np.float64)
    work2 = np.empty(real.size, dtype=np.float64)
    work3 = np.empty(real.size, dtype=np.float64)
    lib().mtl_tema(
        addr(real),
        real.size,
        timeperiod,
        addr(work1),
        addr(work2),
        addr(work3),
        addr(result),
    )
    return result


def TRIMA(real, timeperiod=30):
    real = _real(real)
    timeperiod = _period(timeperiod)
    result = _result(real.size, timeperiod - 1)
    lib().mtl_trima(addr(real), real.size, timeperiod, addr(result))
    return result


def RSI(real, timeperiod=14):
    real = _real(real)
    timeperiod = _period(timeperiod)
    result = _result(real.size, timeperiod)
    lib().mtl_rsi(addr(real), real.size, timeperiod, addr(result))
    return result


def MACD(real, fastperiod=12, slowperiod=26, signalperiod=9):
    real = _real(real)
    fastperiod = _period(fastperiod)
    slowperiod = _period(slowperiod)
    signalperiod = _period(signalperiod, minimum=1)
    lookback = max(fastperiod, slowperiod) + signalperiod - 2
    macd = _result(real.size, lookback)
    signal = _result(real.size, lookback)
    hist = _result(real.size, lookback)
    lib().mtl_macd(
        addr(real),
        real.size,
        fastperiod,
        slowperiod,
        signalperiod,
        addr(macd),
        addr(signal),
        addr(hist),
    )
    macd[: min(real.size, lookback)] = np.nan
    hist[: min(real.size, lookback)] = np.nan
    return macd, signal, hist


def BBANDS(real, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
    if int(matype) != 0:
        raise NotImplementedError("BBANDS currently supports matype=0 (SMA) only")
    real = _real(real)
    timeperiod = _period(timeperiod)
    upper = _result(real.size, timeperiod - 1)
    middle = _result(real.size, timeperiod - 1)
    lower = _result(real.size, timeperiod - 1)
    lib().mtl_bbands(
        addr(real),
        real.size,
        timeperiod,
        float(nbdevup),
        float(nbdevdn),
        addr(upper),
        addr(middle),
        addr(lower),
    )
    return upper, middle, lower


def TRANGE(high, low, close):
    high, low, close = _real(high, "high"), _real(low, "low"), _real(close, "close")
    n = _same_length(high, low, close)
    result = _result(n, 1)
    lib().mtl_trange(addr(high), addr(low), addr(close), n, addr(result))
    return result


def ATR(high, low, close, timeperiod=14):
    return _atr(high, low, close, timeperiod, False)


def NATR(high, low, close, timeperiod=14):
    return _atr(high, low, close, timeperiod, True)


def _atr(high, low, close, timeperiod, normalize):
    high, low, close = _real(high, "high"), _real(low, "low"), _real(close, "close")
    n = _same_length(high, low, close)
    timeperiod = _period(timeperiod, minimum=1)
    result = _result(n, timeperiod)
    lib().mtl_atr(
        addr(high),
        addr(low),
        addr(close),
        n,
        timeperiod,
        int(normalize),
        addr(result),
    )
    return result


def _momentum(real, timeperiod, mode):
    real = _real(real)
    timeperiod = _period(timeperiod, minimum=1)
    result = _result(real.size, timeperiod)
    lib().mtl_momentum(addr(real), real.size, timeperiod, mode, addr(result))
    return result


def MOM(real, timeperiod=10):
    return _momentum(real, timeperiod, 0)


def ROC(real, timeperiod=10):
    return _momentum(real, timeperiod, 1)


def ROCP(real, timeperiod=10):
    return _momentum(real, timeperiod, 2)


def ROCR(real, timeperiod=10):
    return _momentum(real, timeperiod, 3)


def ROCR100(real, timeperiod=10):
    return _momentum(real, timeperiod, 4)


def STDDEV(real, timeperiod=5, nbdev=1):
    return _variance(real, timeperiod, nbdev, True)


def VAR(real, timeperiod=5, nbdev=1):
    return _variance(real, timeperiod, nbdev, False)


def _variance(real, timeperiod, nbdev, take_sqrt):
    real = _real(real)
    timeperiod = _period(timeperiod)
    result = _result(real.size, timeperiod - 1)
    lib().mtl_variance(
        addr(real),
        real.size,
        timeperiod,
        float(nbdev),
        int(take_sqrt),
        addr(result),
    )
    return result


def _window(real, timeperiod, mode):
    real = _real(real)
    timeperiod = _period(timeperiod)
    result = _result(real.size, timeperiod - 1)
    lib().mtl_window(addr(real), real.size, timeperiod, mode, addr(result))
    result[: min(real.size, timeperiod - 1)] = np.nan
    return result


def MIN(real, timeperiod=30):
    return _window(real, timeperiod, 0)


def MAX(real, timeperiod=30):
    return _window(real, timeperiod, 1)


def SUM(real, timeperiod=30):
    return _window(real, timeperiod, 2)


def AVGPRICE(open, high, low, close):
    return _price(open, high, low, close, 0)


def MEDPRICE(high, low):
    return _price(high, low, high, low, 1)


def TYPPRICE(high, low, close):
    return _price(high, low, close, close, 2)


def WCLPRICE(high, low, close):
    return _price(high, low, close, close, 3)


def _price(a, b, c, d, mode):
    arrays = [_real(a), _real(b), _real(c), _real(d)]
    n = _same_length(*arrays)
    result = _result(n)
    lib().mtl_price(
        *(addr(array) for array in arrays), n, mode, addr(result)
    )
    return result


def OBV(real, volume):
    real, volume = _real(real), _real(volume, "volume")
    n = _same_length(real, volume)
    result = _result(n)
    lib().mtl_obv(addr(real), addr(volume), n, addr(result))
    return result


def AD(high, low, close, volume):
    high, low, close, volume = (
        _real(high, "high"),
        _real(low, "low"),
        _real(close, "close"),
        _real(volume, "volume"),
    )
    n = _same_length(high, low, close, volume)
    result = _result(n)
    lib().mtl_ad(
        addr(high), addr(low), addr(close), addr(volume), n, addr(result)
    )
    return result


def CCI(high, low, close, timeperiod=14):
    high, low, close = _real(high, "high"), _real(low, "low"), _real(close, "close")
    n = _same_length(high, low, close)
    timeperiod = _period(timeperiod)
    result = _result(n, timeperiod - 1)
    work = np.empty(n, dtype=np.float64)
    lib().mtl_cci(
        addr(high), addr(low), addr(close), n, timeperiod, addr(work), addr(result)
    )
    return result


def WILLR(high, low, close, timeperiod=14):
    high, low, close = _real(high, "high"), _real(low, "low"), _real(close, "close")
    n = _same_length(high, low, close)
    timeperiod = _period(timeperiod)
    result = _result(n, timeperiod - 1)
    lib().mtl_willr(
        addr(high), addr(low), addr(close), n, timeperiod, addr(result)
    )
    return result


def CORREL(real0, real1, timeperiod=30):
    real0, real1 = _real(real0, "real0"), _real(real1, "real1")
    n = _same_length(real0, real1)
    timeperiod = _period(timeperiod)
    result = _result(n, timeperiod - 1)
    lib().mtl_correl(addr(real0), addr(real1), n, timeperiod, addr(result))
    return result


def _linearreg(real, timeperiod, mode):
    real = _real(real)
    timeperiod = _period(timeperiod)
    result = _result(real.size, timeperiod - 1)
    lib().mtl_linearreg(addr(real), real.size, timeperiod, mode, addr(result))
    return result


def LINEARREG(real, timeperiod=14):
    return _linearreg(real, timeperiod, 0)


def LINEARREG_SLOPE(real, timeperiod=14):
    return _linearreg(real, timeperiod, 1)


def LINEARREG_INTERCEPT(real, timeperiod=14):
    return _linearreg(real, timeperiod, 2)


def LINEARREG_ANGLE(real, timeperiod=14):
    return _linearreg(real, timeperiod, 3)


def TSF(real, timeperiod=14):
    return _linearreg(real, timeperiod, 4)


__all__ = [
    "AD",
    "ATR",
    "AVGPRICE",
    "BBANDS",
    "CCI",
    "CORREL",
    "DEMA",
    "EMA",
    "LINEARREG",
    "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT",
    "LINEARREG_SLOPE",
    "MACD",
    "MAX",
    "MEDPRICE",
    "MIN",
    "MOM",
    "NATR",
    "OBV",
    "ROC",
    "ROCP",
    "ROCR",
    "ROCR100",
    "RSI",
    "SMA",
    "STDDEV",
    "SUM",
    "TEMA",
    "TRANGE",
    "TRIMA",
    "TSF",
    "TYPPRICE",
    "VAR",
    "WCLPRICE",
    "WILLR",
    "WMA",
]
