"""Numerical and warm-up parity with the Python TA-Lib bindings."""

import numpy as np
import pytest

talib = pytest.importorskip("talib")

import mojo_ta_lib as mojo_talib


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(20260730)
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(0.03, 1.2, 600)))
    high = np.ascontiguousarray(close + rng.uniform(0.05, 3.0, close.size))
    low = np.ascontiguousarray(close - rng.uniform(0.05, 3.0, close.size))
    open_ = np.ascontiguousarray(close + rng.normal(0.0, 0.7, close.size))
    volume = np.ascontiguousarray(rng.integers(100, 1_000_000, close.size).astype(float))
    other = np.ascontiguousarray(50.0 + np.cumsum(rng.normal(size=close.size)))
    return open_, high, low, close, volume, other


def assert_parity(ours, reference, atol=1e-10, rtol=1e-11):
    if isinstance(reference, tuple):
        assert isinstance(ours, tuple)
        assert len(ours) == len(reference)
        for actual, expected in zip(ours, reference):
            np.testing.assert_allclose(
                actual, expected, atol=atol, rtol=rtol, equal_nan=True
            )
    else:
        np.testing.assert_allclose(
            ours, reference, atol=atol, rtol=rtol, equal_nan=True
        )


@pytest.mark.parametrize("name", ["SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA"])
def test_moving_average_parity(market, name):
    close = market[3]
    kwargs = {"timeperiod": 17}
    assert_parity(
        getattr(mojo_talib, name)(close, **kwargs),
        getattr(talib, name)(close, **kwargs),
    )


@pytest.mark.parametrize("name", ["MOM", "ROC", "ROCP", "ROCR", "ROCR100"])
def test_simple_momentum_parity(market, name):
    close = market[3]
    kwargs = {"timeperiod": 11}
    assert_parity(
        getattr(mojo_talib, name)(close, **kwargs),
        getattr(talib, name)(close, **kwargs),
    )


@pytest.mark.parametrize("period", [2, 14, 31])
def test_rsi_parity(market, period):
    close = market[3]
    assert_parity(
        mojo_talib.RSI(close, timeperiod=period),
        talib.RSI(close, timeperiod=period),
    )


def test_rsi_flat_series_matches_zero_rule():
    values = np.ones(80, dtype=np.float64)
    assert_parity(mojo_talib.RSI(values), talib.RSI(values))


@pytest.mark.parametrize("name", ["TRANGE", "ATR", "NATR"])
def test_volatility_parity(market, name):
    _, high, low, close, _, _ = market
    ours = getattr(mojo_talib, name)(high, low, close)
    reference = getattr(talib, name)(high, low, close)
    assert_parity(ours, reference)


def test_macd_parity(market):
    close = market[3]
    kwargs = {"fastperiod": 8, "slowperiod": 21, "signalperiod": 5}
    assert_parity(mojo_talib.MACD(close, **kwargs), talib.MACD(close, **kwargs))


def test_macd_swapped_periods_parity(market):
    close = market[3]
    kwargs = {"fastperiod": 26, "slowperiod": 12, "signalperiod": 9}
    assert_parity(mojo_talib.MACD(close, **kwargs), talib.MACD(close, **kwargs))


def test_bbands_parity(market):
    close = market[3]
    kwargs = {"timeperiod": 20, "nbdevup": 2.5, "nbdevdn": 1.5}
    assert_parity(
        mojo_talib.BBANDS(close, **kwargs),
        talib.BBANDS(close, **kwargs),
        atol=5e-10,
    )


@pytest.mark.parametrize("name", ["STDDEV", "VAR", "MIN", "MAX", "SUM"])
def test_rolling_statistic_parity(market, name):
    close = market[3]
    kwargs = {"timeperiod": 23}
    if name == "STDDEV":
        kwargs["nbdev"] = 2.25
    assert_parity(
        getattr(mojo_talib, name)(close, **kwargs),
        getattr(talib, name)(close, **kwargs),
        atol=5e-10,
    )


def test_var_ignores_nbdev_like_talib(market):
    close = market[3]
    assert_parity(
        mojo_talib.VAR(close, timeperiod=9, nbdev=7),
        talib.VAR(close, timeperiod=9, nbdev=7),
        atol=5e-10,
    )


def test_correlation_parity(market):
    close, other = market[3], market[5]
    assert_parity(
        mojo_talib.CORREL(close, other, timeperiod=30),
        talib.CORREL(close, other, timeperiod=30),
        atol=5e-10,
    )


@pytest.mark.parametrize("name", ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"])
def test_price_transform_parity(market, name):
    open_, high, low, close, _, _ = market
    args = {
        "AVGPRICE": (open_, high, low, close),
        "MEDPRICE": (high, low),
        "TYPPRICE": (high, low, close),
        "WCLPRICE": (high, low, close),
    }[name]
    assert_parity(getattr(mojo_talib, name)(*args), getattr(talib, name)(*args))


@pytest.mark.parametrize("name", ["OBV", "AD"])
def test_volume_indicator_parity(market, name):
    _, high, low, close, volume, _ = market
    args = (close, volume) if name == "OBV" else (high, low, close, volume)
    assert_parity(getattr(mojo_talib, name)(*args), getattr(talib, name)(*args))


@pytest.mark.parametrize("name", ["CCI", "WILLR"])
def test_channel_momentum_parity(market, name):
    _, high, low, close, _, _ = market
    kwargs = {"timeperiod": 19}
    assert_parity(
        getattr(mojo_talib, name)(high, low, close, **kwargs),
        getattr(talib, name)(high, low, close, **kwargs),
    )


def test_cci_simd_tail_parity():
    rng = np.random.default_rng(771)
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(size=137)))
    high = np.ascontiguousarray(close + rng.uniform(0.0, 3.0, close.size))
    low = np.ascontiguousarray(close - rng.uniform(0.0, 3.0, close.size))
    assert_parity(
        mojo_talib.CCI(high, low, close, timeperiod=14),
        talib.CCI(high, low, close, timeperiod=14),
    )


@pytest.mark.parametrize("size", [262143, 262149])
def test_parallel_threshold_paths(size):
    rng = np.random.default_rng(size)
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(size=size)))
    high = np.ascontiguousarray(close + rng.uniform(0.0, 3.0, size))
    low = np.ascontiguousarray(close - rng.uniform(0.0, 3.0, size))
    assert_parity(
        mojo_talib.MIN(close, timeperiod=30),
        talib.MIN(close, timeperiod=30),
    )
    assert_parity(
        mojo_talib.CCI(high, low, close, timeperiod=14),
        talib.CCI(high, low, close, timeperiod=14),
    )


@pytest.mark.parametrize(
    "name",
    [
        "LINEARREG",
        "LINEARREG_SLOPE",
        "LINEARREG_INTERCEPT",
        "LINEARREG_ANGLE",
        "TSF",
    ],
)
def test_linear_regression_family_parity(market, name):
    close = market[3]
    kwargs = {"timeperiod": 18}
    assert_parity(
        getattr(mojo_talib, name)(close, **kwargs),
        getattr(talib, name)(close, **kwargs),
    )


@pytest.mark.parametrize("name", ["SMA", "EMA", "RSI", "ATR", "CCI", "LINEARREG"])
def test_short_inputs_preserve_talib_warmup(market, name):
    _, high, low, close, _, _ = market
    values = close[:8]
    if name in {"ATR", "CCI"}:
        args = (high[:8], low[:8], close[:8])
    else:
        args = (values,)
    assert_parity(getattr(mojo_talib, name)(*args), getattr(talib, name)(*args))


def test_noncontiguous_inputs_are_supported(market):
    close = market[3][::2]
    assert not close.flags.c_contiguous
    assert_parity(mojo_talib.EMA(close), talib.EMA(np.ascontiguousarray(close)))


@pytest.mark.parametrize(
    "call",
    [
        lambda: mojo_talib.SMA([]),
        lambda: mojo_talib.MACD([]),
        lambda: mojo_talib.TRANGE([], [], []),
        lambda: mojo_talib.ATR([], [], []),
        lambda: mojo_talib.AVGPRICE([], [], [], []),
        lambda: mojo_talib.OBV([], []),
        lambda: mojo_talib.AD([], [], [], []),
        lambda: mojo_talib.CCI([], [], []),
        lambda: mojo_talib.CORREL([], []),
    ],
)
def test_empty_inputs_are_safe(call):
    result = call()
    outputs = result if isinstance(result, tuple) else (result,)
    assert all(output.dtype == np.float64 and output.size == 0 for output in outputs)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_input_is_rejected_without_silent_narrowing(dtype):
    with pytest.raises(TypeError, match="complex"):
        mojo_talib.SMA(np.arange(10, dtype=dtype))


def test_extended_precision_input_is_rejected_without_silent_narrowing():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("long double is not wider than float64 on this platform")
    with pytest.raises(TypeError, match="narrowed"):
        mojo_talib.SMA(np.arange(10, dtype=np.longdouble))


def test_length_mismatch_is_rejected(market):
    _, high, low, close, _, _ = market
    with pytest.raises(ValueError, match="same length"):
        mojo_talib.ATR(high[:-1], low, close)


def test_non_sma_bbands_is_explicitly_rejected(market):
    with pytest.raises(NotImplementedError, match="matype=0"):
        mojo_talib.BBANDS(market[3], matype=1)
