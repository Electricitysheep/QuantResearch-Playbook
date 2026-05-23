"""WorldQuant 101 Alpha 因子工具函数

参考: 101 Formulaic Alphas by Zura Kakushadze (2016)

提供 Alpha101 计算所需的全部基础算子。
所有算子支持 Polars Series 和 NumPy array 两种输入。
"""

from __future__ import annotations

import numpy as np
import polars as pl


def to_np(s: pl.Series | np.ndarray) -> np.ndarray:
    return s.to_numpy() if isinstance(s, pl.Series) else s


def to_series(s: pl.Series | np.ndarray, name: str = "") -> pl.Series:
    if isinstance(s, pl.Series):
        return s
    return pl.Series(name, s)


# ── 时序算子 ─────────────────────────────────


def ts_sum(x: pl.Series | np.ndarray, d: int) -> pl.Series:
    """过去 d 期求和"""
    arr = to_np(x)
    result = np.full_like(arr, np.nan)
    for i in range(d - 1, len(arr)):
        result[i] = np.nansum(arr[i - d + 1 : i + 1])
    return to_series(result, x.name if isinstance(x, pl.Series) else "")


def ts_mean(x: pl.Series | np.ndarray, d: int) -> pl.Series:
    """过去 d 期均值"""
    arr = to_np(x)
    result = np.full_like(arr, np.nan)
    for i in range(d - 1, len(arr)):
        result[i] = np.nanmean(arr[i - d + 1 : i + 1])
    return to_series(result, x.name if isinstance(x, pl.Series) else "")


def ts_stddev(x: pl.Series | np.ndarray, d: int) -> pl.Series:
    """过去 d 期标准差"""
    arr = to_np(x)
    result = np.full_like(arr, np.nan)
    for i in range(d - 1, len(arr)):
        result[i] = np.nanstd(arr[i - d + 1 : i + 1])
    return to_series(result, x.name if isinstance(x, pl.Series) else "")


def ts_min(x: pl.Series | np.ndarray, d: int) -> pl.Series:
    arr = to_np(x)
    result = np.full_like(arr, np.nan)
    for i in range(d - 1, len(arr)):
        result[i] = np.nanmin(arr[i - d + 1 : i + 1])
    return to_series(result, x.name if isinstance(x, pl.Series) else "")


def ts_max(x: pl.Series | np.ndarray, d: int) -> pl.Series:
    arr = to_np(x)
    result = np.full_like(arr, np.nan)
    for i in range(d - 1, len(arr)):
        result[i] = np.nanmax(arr[i - d + 1 : i + 1])
    return to_series(result, x.name if isinstance(x, pl.Series) else "")


def ts_argmax(x: pl.Series | np.ndarray, d: int) -> pl.Series:
    """过去 d 期最大值距今天数"""
    arr = to_np(x)
    result = np.full_like(arr, np.nan)
    for i in range(d - 1, len(arr)):
        window = arr[i - d + 1 : i + 1]
        if np.all(np.isnan(window)):
            continue
        result[i] = d - 1 - np.nanargmax(window)
    return to_series(result, x.name if isinstance(x, pl.Series) else "")


def ts_argmin(x: pl.Series | np.ndarray, d: int) -> pl.Series:
    """过去 d 期最小值距今天数"""
    arr = to_np(x)
    result = np.full_like(arr, np.nan)
    for i in range(d - 1, len(arr)):
        window = arr[i - d + 1 : i + 1]
        if np.all(np.isnan(window)):
            continue
        result[i] = d - 1 - np.nanargmin(window)
    return to_series(result, x.name if isinstance(x, pl.Series) else "")


def ts_rank(x: pl.Series | np.ndarray, d: int) -> pl.Series:
    """过去 d 期百分位排名 [0, 1]"""
    arr = to_np(x)
    result = np.full_like(arr, np.nan)
    for i in range(d - 1, len(arr)):
        window = arr[i - d + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            result[i] = 0.5
        else:
            result[i] = np.mean(window[-1] >= valid)
    return to_series(result, x.name if isinstance(x, pl.Series) else "")


def ts_corr(x: pl.Series, y: pl.Series, d: int) -> pl.Series:
    """过去 d 期相关系数"""
    xn, yn = x.to_numpy(), y.to_numpy()
    result = np.full_like(xn, np.nan)
    for i in range(d, len(xn)):
        mask = ~(np.isnan(xn[i - d : i]) | np.isnan(yn[i - d : i]))
        if mask.sum() < 3:
            continue
        result[i] = np.corrcoef(xn[i - d : i][mask], yn[i - d : i][mask])[0, 1]
    return to_series(result)


def ts_covariance(x: pl.Series, y: pl.Series, d: int) -> pl.Series:
    """过去 d 期协方差"""
    xn, yn = x.to_numpy(), y.to_numpy()
    result = np.full_like(xn, np.nan)
    for i in range(d, len(xn)):
        m = ~(np.isnan(xn[i - d : i]) | np.isnan(yn[i - d : i]))
        if m.sum() < 3:
            continue
        result[i] = np.cov(xn[i - d : i][m], yn[i - d : i][m])[0, 1]
    return to_series(result)


def delta(x: pl.Series | np.ndarray, d: int) -> pl.Series:
    """x(t) - x(t-d)"""
    arr = to_np(x)
    result = np.full_like(arr, np.nan)
    result[d:] = arr[d:] - arr[:-d]
    return to_series(result, x.name if isinstance(x, pl.Series) else "")


def delay(x: pl.Series | np.ndarray, d: int) -> pl.Series:
    """滞后 d 期"""
    arr = to_np(x)
    result = np.full_like(arr, np.nan)
    result[d:] = arr[:-d]
    return to_series(result, x.name if isinstance(x, pl.Series) else "")


def signed_power(x: pl.Series | np.ndarray, a: float) -> pl.Series:
    """保持符号的幂运算: sign(x) * |x|^a"""
    arr = to_np(x)
    return to_series(np.sign(arr) * np.abs(arr) ** a)


def decay_linear(x: pl.Series | np.ndarray, d: int) -> pl.Series:
    """线性衰减加权移动平均，权重 d, d-1, ..., 1"""
    arr = to_np(x)
    weights = np.arange(d, 0, -1, dtype=float)
    weights /= weights.sum()
    result = np.full_like(arr, np.nan)
    for i in range(d - 1, len(arr)):
        window = arr[i - d + 1 : i + 1]
        if np.any(np.isnan(window)):
            continue
        result[i] = np.dot(window, weights)
    return to_series(result)


# ── 截面算子 ─────────────────────────────────


def rank(x: pl.Series | np.ndarray) -> pl.Series:
    """横截面排名 [0, 1]"""
    arr = to_np(x)
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return to_series(np.full_like(arr, 0.5))
    ranks = np.full_like(arr, np.nan)
    ranks[valid] = (np.argsort(np.argsort(arr[valid])) + 1) / valid.sum()
    return to_series(ranks)


def scale(x: pl.Series | np.ndarray) -> pl.Series:
    """归一化: sum(|x|) = 1"""
    arr = to_np(x)
    abs_sum = np.nansum(np.abs(arr))
    if abs_sum == 0:
        return to_series(arr)
    return to_series(arr / abs_sum)


# ── 辅助检查 ─────────────────────────────────


def check_alpha_input(data) -> None:
    """验证 Alpha101 输入数据完整性"""
    required = {"open", "high", "low", "close", "volume", "returns", "vwap"}
    cols = set(data.columns)
    missing = required - cols
    if missing:
        raise ValueError(f"缺少必要列: {missing}")


def prepare_alpha_data(data, adj_close: str = "close") -> dict:
    """准备 Alpha101 所需的标准数据集"""
    import polars as pl

    close = data[adj_close]
    open_p = data.get("open", close)
    high = data.get("high", close)
    low = data.get("low", close)
    volume = data.get("volume", pl.Series("vol", np.ones(len(close))))
    vwap = data.get("vwap", close)
    returns = data.get("returns", close.pct_change(1))

    return {
        "open": open_p, "high": high, "low": low,
        "close": close, "volume": volume,
        "vwap": vwap, "returns": returns,
    }
