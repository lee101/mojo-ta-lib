"""TA-Lib-compatible indicator kernels and their C ABI."""

from std.algorithm import parallelize
from std.math import atan, sqrt
from std.memory import stack_allocation
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 262144
comptime PARALLEL_CHUNK = 65536
comptime EXTREME_QUEUE_SIZE = 256


def ptr(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


def ema_from(src: Ptr, dst: Ptr, n: Int, first: Int, period: Int):
    var seed_end = first + period - 1
    if seed_end >= n:
        return
    var value = 0.0
    for i in range(first, seed_end + 1):
        value += src[i]
    var scale = 1.0 / Float64(period)
    value *= scale
    dst[seed_end] = value
    var alpha = 2.0 / Float64(period + 1)
    for i in range(seed_end + 1, n):
        value += alpha * (src[i] - value)
        dst[i] = value


@export("mtl_sma")
def mtl_sma(src_addr: Int, n: Int, period: Int, dst_addr: Int) abi("C"):
    var src = ptr(src_addr)
    var dst = ptr(dst_addr)
    if n < period:
        return
    var total = 0.0
    for i in range(period):
        total += src[i]
    var scale = 1.0 / Float64(period)
    dst[period - 1] = total * scale
    for i in range(period, n):
        total += src[i] - src[i - period]
        dst[i] = total * scale


@export("mtl_ema")
def mtl_ema(src_addr: Int, n: Int, period: Int, dst_addr: Int) abi("C"):
    ema_from(ptr(src_addr), ptr(dst_addr), n, 0, period)


@export("mtl_wma")
def mtl_wma(src_addr: Int, n: Int, period: Int, dst_addr: Int) abi("C"):
    var src = ptr(src_addr)
    var dst = ptr(dst_addr)
    if n < period:
        return
    var total = 0.0
    var weighted = 0.0
    for i in range(period):
        total += src[i]
        weighted += Float64(i + 1) * src[i]
    var divisor = Float64(period * (period + 1)) * 0.5
    dst[period - 1] = weighted / divisor
    for i in range(period, n):
        weighted += Float64(period) * src[i] - total
        total += src[i] - src[i - period]
        dst[i] = weighted / divisor


@export("mtl_dema")
def mtl_dema(
    src_addr: Int,
    n: Int,
    period: Int,
    work1_addr: Int,
    work2_addr: Int,
    dst_addr: Int,
) abi("C"):
    var src = ptr(src_addr)
    var e1 = ptr(work1_addr)
    var e2 = ptr(work2_addr)
    var dst = ptr(dst_addr)
    ema_from(src, e1, n, 0, period)
    ema_from(e1, e2, n, period - 1, period)
    for i in range(2 * (period - 1), n):
        dst[i] = 2.0 * e1[i] - e2[i]


@export("mtl_tema")
def mtl_tema(
    src_addr: Int,
    n: Int,
    period: Int,
    work1_addr: Int,
    work2_addr: Int,
    work3_addr: Int,
    dst_addr: Int,
) abi("C"):
    var src = ptr(src_addr)
    var e1 = ptr(work1_addr)
    var e2 = ptr(work2_addr)
    var e3 = ptr(work3_addr)
    var dst = ptr(dst_addr)
    ema_from(src, e1, n, 0, period)
    ema_from(e1, e2, n, period - 1, period)
    ema_from(e2, e3, n, 2 * (period - 1), period)
    for i in range(3 * (period - 1), n):
        dst[i] = 3.0 * e1[i] - 3.0 * e2[i] + e3[i]


@export("mtl_trima")
def mtl_trima(src_addr: Int, n: Int, period: Int, dst_addr: Int) abi("C"):
    var src = ptr(src_addr)
    var dst = ptr(dst_addr)
    if n < period:
        return
    var left = (period + 1) // 2
    var right = period // 2
    var divisor = Float64(left * (left + 1) // 2 + right * (right + 1) // 2)
    for end in range(period - 1, n):
        var start = end - period + 1
        var total = 0.0
        for j in range(left):
            total += Float64(j + 1) * src[start + j]
        for j in range(right):
            total += Float64(right - j) * src[start + left + j]
        dst[end] = total / divisor


@export("mtl_rsi")
def mtl_rsi(src_addr: Int, n: Int, period: Int, dst_addr: Int) abi("C"):
    var src = ptr(src_addr)
    var dst = ptr(dst_addr)
    if n <= period:
        return
    var gain = 0.0
    var loss = 0.0
    for i in range(1, period + 1):
        var change = src[i] - src[i - 1]
        if change > 0.0:
            gain += change
        else:
            loss -= change
    var period_value = Float64(period)
    var previous_weight = Float64(period - 1)
    gain /= period_value
    loss /= period_value
    var denom = gain + loss
    dst[period] = 100.0 * gain / denom if denom != 0.0 else 0.0
    for i in range(period + 1, n):
        var change = src[i] - src[i - 1]
        gain *= previous_weight
        loss *= previous_weight
        if change > 0.0:
            gain += change
        elif change < 0.0:
            loss -= change
        gain /= period_value
        loss /= period_value
        denom = gain + loss
        dst[i] = 100.0 * gain / denom if denom != 0.0 else 0.0


@export("mtl_macd")
def mtl_macd(
    src_addr: Int,
    n: Int,
    fast_period: Int,
    slow_period: Int,
    signal_period: Int,
    macd_addr: Int,
    signal_addr: Int,
    hist_addr: Int,
) abi("C"):
    var src = ptr(src_addr)
    var macd = ptr(macd_addr)
    var signal = ptr(signal_addr)
    var hist = ptr(hist_addr)
    var fp = fast_period
    var sp = slow_period
    if sp < fp:
        var swap = sp
        sp = fp
        fp = swap
    var first_macd = sp - 1
    var lookback = first_macd + signal_period - 1
    if n <= lookback:
        return
    var fast = hist
    var slow = macd
    ema_from(src, fast, n, sp - fp, fp)
    ema_from(src, slow, n, 0, sp)
    var sig = 0.0
    for i in range(first_macd, lookback + 1):
        sig += fast[i] - slow[i]
    sig /= Float64(signal_period)
    var alpha = 2.0 / Float64(signal_period + 1)
    for i in range(lookback, n):
        var value = fast[i] - slow[i]
        if i > lookback:
            sig += alpha * (value - sig)
        macd[i] = value
        signal[i] = sig
        hist[i] = value - sig


@export("mtl_bbands")
def mtl_bbands(
    src_addr: Int,
    n: Int,
    period: Int,
    up_dev: Float64,
    down_dev: Float64,
    upper_addr: Int,
    middle_addr: Int,
    lower_addr: Int,
) abi("C"):
    var src = ptr(src_addr)
    var upper = ptr(upper_addr)
    var middle = ptr(middle_addr)
    var lower = ptr(lower_addr)
    if n < period:
        return
    var total = 0.0
    var squares = 0.0
    for i in range(period):
        total += src[i]
        squares += src[i] * src[i]
    var scale = 1.0 / Float64(period)
    for i in range(period - 1, n):
        if i >= period:
            var old = src[i - period]
            total += src[i] - old
            squares += src[i] * src[i] - old * old
        var mean = total * scale
        var variance = squares * scale - mean * mean
        var deviation = sqrt(variance) if variance > 0.0 else 0.0
        middle[i] = mean
        upper[i] = mean + up_dev * deviation
        lower[i] = mean - down_dev * deviation


@export("mtl_trange")
def mtl_trange(
    high_addr: Int, low_addr: Int, close_addr: Int, n: Int, dst_addr: Int
) abi("C"):
    var high = ptr(high_addr)
    var low = ptr(low_addr)
    var close = ptr(close_addr)
    var dst = ptr(dst_addr)
    for i in range(1, n):
        var value = high[i] - low[i]
        var hc = high[i] - close[i - 1]
        if hc < 0.0:
            hc = -hc
        var lc = low[i] - close[i - 1]
        if lc < 0.0:
            lc = -lc
        if hc > value:
            value = hc
        if lc > value:
            value = lc
        dst[i] = value


@export("mtl_atr")
def mtl_atr(
    high_addr: Int,
    low_addr: Int,
    close_addr: Int,
    n: Int,
    period: Int,
    normalize: Int,
    dst_addr: Int,
) abi("C"):
    var high = ptr(high_addr)
    var low = ptr(low_addr)
    var close = ptr(close_addr)
    var dst = ptr(dst_addr)
    if n <= period:
        return
    var value = 0.0
    for i in range(1, period + 1):
        var tr = high[i] - low[i]
        var hc = high[i] - close[i - 1]
        if hc < 0.0:
            hc = -hc
        var lc = low[i] - close[i - 1]
        if lc < 0.0:
            lc = -lc
        if hc > tr:
            tr = hc
        if lc > tr:
            tr = lc
        value += tr
    value /= Float64(period)
    if normalize != 0:
        dst[period] = 100.0 * value / close[period] if close[period] != 0.0 else 0.0
    else:
        dst[period] = value
    for i in range(period + 1, n):
        var tr = high[i] - low[i]
        var hc = high[i] - close[i - 1]
        if hc < 0.0:
            hc = -hc
        var lc = low[i] - close[i - 1]
        if lc < 0.0:
            lc = -lc
        if hc > tr:
            tr = hc
        if lc > tr:
            tr = lc
        value = (value * Float64(period - 1) + tr) / Float64(period)
        if normalize != 0:
            dst[i] = 100.0 * value / close[i] if close[i] != 0.0 else 0.0
        else:
            dst[i] = value


@export("mtl_momentum")
def mtl_momentum(
    src_addr: Int, n: Int, period: Int, mode: Int, dst_addr: Int
) abi("C"):
    var src = ptr(src_addr)
    var dst = ptr(dst_addr)
    for i in range(period, n):
        var previous = src[i - period]
        if mode == 0:
            dst[i] = src[i] - previous
        elif mode == 1:
            dst[i] = (src[i] / previous - 1.0) * 100.0 if previous != 0.0 else 0.0
        elif mode == 2:
            dst[i] = (src[i] - previous) / previous if previous != 0.0 else 0.0
        elif mode == 3:
            dst[i] = src[i] / previous if previous != 0.0 else 0.0
        else:
            dst[i] = src[i] / previous * 100.0 if previous != 0.0 else 0.0


@export("mtl_variance")
def mtl_variance(
    src_addr: Int,
    n: Int,
    period: Int,
    deviation: Float64,
    take_sqrt: Int,
    dst_addr: Int,
) abi("C"):
    var src = ptr(src_addr)
    var dst = ptr(dst_addr)
    if n < period:
        return
    var total = 0.0
    var squares = 0.0
    for i in range(period):
        total += src[i]
        squares += src[i] * src[i]
    for i in range(period - 1, n):
        if i >= period:
            var old = src[i - period]
            total += src[i] - old
            squares += src[i] * src[i] - old * old
        var mean = total / Float64(period)
        var value = squares / Float64(period) - mean * mean
        if take_sqrt != 0:
            value = sqrt(value) * deviation if value > 0.00000000000001 else 0.0
        dst[i] = value


def window_extreme_range(
    src: Ptr,
    dst: Ptr,
    period: Int,
    mode: Int,
    first_end: Int,
    stop_end: Int,
):
    var first = first_end - period + 1
    var value = src[first]
    var value_index = first
    for i in range(first + 1, first_end + 1):
        if (mode == 0 and src[i] <= value) or (mode == 1 and src[i] >= value):
            value = src[i]
            value_index = i
    dst[first_end] = value
    for end in range(first_end + 1, stop_end):
        if (mode == 0 and src[end] <= value) or (mode == 1 and src[end] >= value):
            value = src[end]
            value_index = end
        elif value_index <= end - period:
            value_index = end - period + 1
            value = src[value_index]
            for i in range(value_index + 1, end + 1):
                if (mode == 0 and src[i] <= value) or (mode == 1 and src[i] >= value):
                    value = src[i]
                    value_index = i
        dst[end] = value


def window_deque_range(
    src: Ptr,
    dst: Ptr,
    period: Int,
    mode: Int,
    first_end: Int,
    stop_end: Int,
):
    var queue = stack_allocation[EXTREME_QUEUE_SIZE, Int]()
    var head = 0
    var tail = 0
    var first = first_end - period + 1
    for i in range(first, first_end + 1):
        while tail > head:
            var back = queue[(tail - 1) & (EXTREME_QUEUE_SIZE - 1)]
            if (mode == 0 and src[i] <= src[back]) or (
                mode == 1 and src[i] >= src[back]
            ):
                tail -= 1
            else:
                break
        queue[tail & (EXTREME_QUEUE_SIZE - 1)] = i
        tail += 1
    dst[first_end] = src[queue[head & (EXTREME_QUEUE_SIZE - 1)]]
    for end in range(first_end + 1, stop_end):
        if queue[head & (EXTREME_QUEUE_SIZE - 1)] <= end - period:
            head += 1
        while tail > head:
            var back = queue[(tail - 1) & (EXTREME_QUEUE_SIZE - 1)]
            if (mode == 0 and src[end] <= src[back]) or (
                mode == 1 and src[end] >= src[back]
            ):
                tail -= 1
            else:
                break
        queue[tail & (EXTREME_QUEUE_SIZE - 1)] = end
        tail += 1
        dst[end] = src[queue[head & (EXTREME_QUEUE_SIZE - 1)]]


@export("mtl_window")
def mtl_window(
    src_addr: Int, n: Int, period: Int, mode: Int, dst_addr: Int
) abi("C"):
    var src = ptr(src_addr)
    var dst = ptr(dst_addr)
    if mode == 2:
        if n < period:
            return
        var total = 0.0
        for i in range(period):
            total += src[i]
        dst[period - 1] = total
        for i in range(period, n):
            total += src[i] - src[i - period]
            dst[i] = total
        return
    if n < period:
        return
    if n < PARALLEL_THRESHOLD:
        if period <= EXTREME_QUEUE_SIZE:
            window_deque_range(src, dst, period, mode, period - 1, n)
        else:
            window_extreme_range(src, dst, period, mode, period - 1, n)
        return
    var valid = n - period + 1
    var tasks = (valid + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK

    @parameter
    def work(task: Int):
        var first_end = period - 1 + task * PARALLEL_CHUNK
        var stop_end = min(n, first_end + PARALLEL_CHUNK)
        if period <= EXTREME_QUEUE_SIZE:
            window_deque_range(src, dst, period, mode, first_end, stop_end)
        else:
            window_extreme_range(src, dst, period, mode, first_end, stop_end)

    parallelize[work](tasks, min(tasks, 16))


@export("mtl_price")
def mtl_price(
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    d_addr: Int,
    n: Int,
    mode: Int,
    dst_addr: Int,
) abi("C"):
    var a = ptr(a_addr)
    var b = ptr(b_addr)
    var c = ptr(c_addr)
    var d = ptr(d_addr)
    var dst = ptr(dst_addr)
    for i in range(n):
        if mode == 0:
            dst[i] = (a[i] + b[i] + c[i] + d[i]) * 0.25
        elif mode == 1:
            dst[i] = (a[i] + b[i]) * 0.5
        elif mode == 2:
            dst[i] = (a[i] + b[i] + c[i]) / 3.0
        else:
            dst[i] = (a[i] + b[i] + 2.0 * c[i]) * 0.25


@export("mtl_obv")
def mtl_obv(
    src_addr: Int, volume_addr: Int, n: Int, dst_addr: Int
) abi("C"):
    if n <= 0:
        return
    var src = ptr(src_addr)
    var volume = ptr(volume_addr)
    var dst = ptr(dst_addr)
    var value = volume[0]
    dst[0] = value
    for i in range(1, n):
        if src[i] > src[i - 1]:
            value += volume[i]
        elif src[i] < src[i - 1]:
            value -= volume[i]
        dst[i] = value


@export("mtl_ad")
def mtl_ad(
    high_addr: Int,
    low_addr: Int,
    close_addr: Int,
    volume_addr: Int,
    n: Int,
    dst_addr: Int,
) abi("C"):
    var high = ptr(high_addr)
    var low = ptr(low_addr)
    var close = ptr(close_addr)
    var volume = ptr(volume_addr)
    var dst = ptr(dst_addr)
    var value = 0.0
    for i in range(n):
        var span = high[i] - low[i]
        if span > 0.0:
            value += ((close[i] - low[i]) - (high[i] - close[i])) / span * volume[i]
        dst[i] = value


def cci_finish_range(
    typical: Ptr,
    dst: Ptr,
    period: Int,
    first_end: Int,
    stop_end: Int,
):
    comptime W = simd_width_of[DType.float64]()
    for end in range(first_end, stop_end):
        var i = end - period + 1
        var stop = end + 1
        var mean = 0.0
        for j in range(i, stop):
            mean += typical[j]
        mean /= Float64(period)
        var deviation = SIMD[DType.float64, W](0.0)
        while i + W <= stop:
            var delta = typical.load[width=W](i) - mean
            deviation += max(delta, -delta)
            i += W
        var total_deviation = deviation.reduce_add()
        while i < stop:
            var delta = typical[i] - mean
            total_deviation += delta if delta >= 0.0 else -delta
            i += 1
        var denom = 0.015 * total_deviation / Float64(period)
        dst[end] = (
            (typical[end] - mean) / denom if denom != 0.0 else 0.0
        )


@export("mtl_cci")
def mtl_cci(
    high_addr: Int,
    low_addr: Int,
    close_addr: Int,
    n: Int,
    period: Int,
    work_addr: Int,
    dst_addr: Int,
) abi("C"):
    var high = ptr(high_addr)
    var low = ptr(low_addr)
    var close = ptr(close_addr)
    var typical = ptr(work_addr)
    var dst = ptr(dst_addr)
    if n < period:
        return
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        var values = (
            high.load[width=W](i)
            + low.load[width=W](i)
            + close.load[width=W](i)
        ) / 3.0
        typical.store(i, values)
        i += W
    while i < n:
        typical[i] = (high[i] + low[i] + close[i]) / 3.0
        i += 1
    if n < PARALLEL_THRESHOLD:
        cci_finish_range(typical, dst, period, period - 1, n)
        return
    var valid = n - period + 1
    var tasks = (valid + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK

    @parameter
    def work(task: Int):
        var first_end = period - 1 + task * PARALLEL_CHUNK
        var stop_end = min(n, first_end + PARALLEL_CHUNK)
        cci_finish_range(typical, dst, period, first_end, stop_end)

    parallelize[work](tasks, min(tasks, 16))


@export("mtl_willr")
def mtl_willr(
    high_addr: Int,
    low_addr: Int,
    close_addr: Int,
    n: Int,
    period: Int,
    dst_addr: Int,
) abi("C"):
    var high = ptr(high_addr)
    var low = ptr(low_addr)
    var close = ptr(close_addr)
    var dst = ptr(dst_addr)
    for end in range(period - 1, n):
        var highest = high[end - period + 1]
        var lowest = low[end - period + 1]
        for i in range(end - period + 2, end + 1):
            if high[i] > highest:
                highest = high[i]
            if low[i] < lowest:
                lowest = low[i]
        var span = highest - lowest
        dst[end] = -100.0 * (highest - close[end]) / span if span != 0.0 else 0.0


@export("mtl_correl")
def mtl_correl(
    a_addr: Int,
    b_addr: Int,
    n: Int,
    period: Int,
    dst_addr: Int,
) abi("C"):
    var a = ptr(a_addr)
    var b = ptr(b_addr)
    var dst = ptr(dst_addr)
    if n < period:
        return
    var sx = 0.0
    var sy = 0.0
    var sxx = 0.0
    var syy = 0.0
    var sxy = 0.0
    for i in range(period):
        sx += a[i]
        sy += b[i]
        sxx += a[i] * a[i]
        syy += b[i] * b[i]
        sxy += a[i] * b[i]
    for end in range(period - 1, n):
        if end >= period:
            var old = end - period
            sx += a[end] - a[old]
            sy += b[end] - b[old]
            sxx += a[end] * a[end] - a[old] * a[old]
            syy += b[end] * b[end] - b[old] * b[old]
            sxy += a[end] * b[end] - a[old] * b[old]
        var xvar = Float64(period) * sxx - sx * sx
        var yvar = Float64(period) * syy - sy * sy
        var denom = sqrt(xvar * yvar) if xvar > 0.0 and yvar > 0.0 else 0.0
        dst[end] = (Float64(period) * sxy - sx * sy) / denom if denom != 0.0 else 0.0


@export("mtl_linearreg")
def mtl_linearreg(
    src_addr: Int,
    n: Int,
    period: Int,
    mode: Int,
    dst_addr: Int,
) abi("C"):
    var src = ptr(src_addr)
    var dst = ptr(dst_addr)
    var count = Float64(period)
    var sx = count * Float64(period - 1) * 0.5
    var sxx = count * Float64(period - 1) * Float64(2 * period - 1) / 6.0
    var denominator = count * sxx - sx * sx
    if n < period:
        return
    var sy = 0.0
    var sxy = 0.0
    for j in range(period):
        sy += src[j]
        sxy += Float64(j) * src[j]
    for end in range(period - 1, n):
        if end >= period:
            var old = src[end - period]
            var previous_sum = sy
            sy += src[end] - old
            sxy += Float64(period - 1) * src[end] - (previous_sum - old)
        var slope = (count * sxy - sx * sy) / denominator
        var intercept = (sy - slope * sx) / count
        if mode == 0:
            dst[end] = intercept + slope * Float64(period - 1)
        elif mode == 1:
            dst[end] = slope
        elif mode == 2:
            dst[end] = intercept
        elif mode == 3:
            dst[end] = atan(slope) * 57.29577951308232
        else:
            dst[end] = intercept + slope * Float64(period)
