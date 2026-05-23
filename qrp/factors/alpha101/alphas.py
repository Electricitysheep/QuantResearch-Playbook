"""WorldQuant 101 Alpha 因子实现

参考: 101 Formulaic Alphas by Zura Kakushadze, Wilmott 2016

实现方式：
  - 每个 alpha 是一个独立的函数
  - 输入为 prepared dict (含 open/high/low/close/volume/vwap/returns)
  - 返回 pl.Series

批量计算：Alpha101.compute_all(data) -> dict[str, pl.Series]
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .utils import (
    check_alpha_input,
    delay,
    delta,
    prepare_alpha_data,
    rank,
    signed_power,
    ts_argmax,
    ts_corr,
    ts_covariance,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_stddev,
    ts_sum,
)

# ── Alpha #1–#20 ──────────────────────────────────────────


def alpha001(d):
    """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
    inner = d["close"].clone()
    inner = inner.to_frame("c").with_columns(
        pl.when(pl.col("c") < 0).then(ts_stddev(d["returns"], 20)).otherwise(pl.col("c")).alias("inner")
    )["inner"]
    return rank(ts_argmax(signed_power(inner, 2.0), 5)) - 0.5


def alpha002(d):
    """-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)"""
    a = rank(delta(np.log(d["volume"].to_numpy() + 1e-10), 2))
    b = rank((d["close"] - d["open"]) / (d["open"] + 1e-10))
    result = -ts_corr(a, b, 6)
    return result.fill_nan(0).fill_null(0)


def alpha003(d):
    """-1 * correlation(rank(open), rank(volume), 10)"""
    result = -ts_corr(rank(d["open"]), rank(d["volume"]), 10)
    return result.fill_nan(0).fill_null(0)


def alpha004(d):
    """-1 * Ts_Rank(rank(low), 9)"""
    return -ts_rank(rank(d["low"]), 9)


def alpha005(d):
    """(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
    return rank(d["open"] - ts_mean(d["vwap"], 10)) * (-1 * abs(rank(d["close"] - d["vwap"])).to_numpy())


def alpha006(d):
    """-1 * correlation(open, volume, 10)"""
    return -ts_corr(d["open"], d["volume"], 10).fill_nan(0).fill_null(0)


def alpha007(d):
    """((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))"""
    adv20 = ts_mean(d["volume"], 20)
    inner = -ts_rank(abs(delta(d["close"], 7)), 60) * np.sign(delta(d["close"], 7).to_numpy())
    cond = adv20 < d["volume"]
    result = np.where(cond.to_numpy(), inner.to_numpy(), -1.0)
    return pl.Series("alpha007", result)


def alpha008(d):
    """-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))"""
    v = ts_sum(d["open"], 5) * ts_sum(d["returns"], 5)
    return -rank(v - delay(v, 10))


def alpha009(d):
    """((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))"""  # noqa: E501
    dc = delta(d["close"], 1)
    cond1 = ts_min(dc, 5) > 0
    cond2 = ts_max(dc, 5) < 0
    result = -dc.to_numpy()
    mask = cond1.to_numpy() | cond2.to_numpy()
    result[mask] = dc.to_numpy()[mask]
    return pl.Series("alpha009", result)


def alpha010(d):
    """rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"""  # noqa: E501
    dc = delta(d["close"], 1)
    cond1 = ts_min(dc, 4) > 0
    cond2 = ts_max(dc, 4) < 0
    result = -dc.to_numpy()
    mask = cond1.to_numpy() | cond2.to_numpy()
    result[mask] = dc.to_numpy()[mask]
    return rank(pl.Series("_", result))


def alpha011(d):
    """((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
    diff = d["vwap"] - d["close"]
    return (rank(ts_max(diff, 3)) + rank(ts_min(diff, 3))) * rank(delta(d["volume"], 3))


def alpha012(d):
    """(sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
    return np.sign(delta(d["volume"], 1).to_numpy()) * (-delta(d["close"], 1).to_numpy())


def alpha013(d):
    """-1 * rank(covariance(rank(close), rank(volume), 5))"""
    return -rank(ts_covariance(rank(d["close"]), rank(d["volume"]), 5))


def alpha014(d):
    """((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
    return (-rank(delta(d["returns"], 3)) * ts_corr(d["open"], d["volume"], 10)).fill_nan(0)


def alpha015(d):
    """-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)"""
    return -ts_sum(rank(ts_corr(rank(d["high"]), rank(d["volume"]), 3)), 3)


def alpha016(d):
    """-1 * rank(covariance(rank(high), rank(volume), 5))"""
    return -rank(ts_covariance(rank(d["high"]), rank(d["volume"]), 5))


def alpha017(d):
    """((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)))"""
    return (-rank(ts_rank(d["close"], 10)) * rank(delta(delta(d["close"], 1), 1)))


def alpha018(d):
    """-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10)))"""
    abs_diff = abs(d["close"] - d["open"])
    return -rank(ts_stddev(abs_diff, 5) + (d["close"] - d["open"]) + ts_corr(d["close"], d["open"], 10).fill_nan(0))


def alpha019(d):
    """((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    return (-np.sign(((d["close"] - delay(d["close"], 7)) + delta(d["close"], 7)).to_numpy())
            * (1 + rank(1 + ts_sum(d["returns"], 250))).to_numpy())


def alpha020(d):
    """(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))"""
    return (-rank(d["open"] - delay(d["high"], 1)) * rank(d["open"] - delay(d["close"], 1))
            * rank(d["open"] - delay(d["low"], 1)))


# ── Alpha 注册表 ─────────────────────────────────


ALPHAS = {
    1: alpha001, 2: alpha002, 3: alpha003, 4: alpha004, 5: alpha005,
    6: alpha006, 7: alpha007, 8: alpha008, 9: alpha009, 10: alpha010,
    11: alpha011, 12: alpha012, 13: alpha013, 14: alpha014, 15: alpha015,
    16: alpha016, 17: alpha017, 18: alpha018, 19: alpha019, 20: alpha020,
}

ALPHA_NAMES = {f"alpha{n:03d}": n for n in ALPHAS}


class Alpha101:
    """WorldQuant 101 Alpha 因子计算器"""

    def __init__(self):
        self._results: dict[str, pl.Series] = {}

    def compute(self, n: int, data: pl.DataFrame) -> pl.Series:
        """计算单个 Alpha"""
        check_alpha_input(data)
        d = prepare_alpha_data(data)
        fn = ALPHAS.get(n)
        if fn is None:
            raise ValueError(f"Alpha #{n} 未实现 (已实现: {sorted(ALPHAS)})")
        result = fn(d)
        self._results[f"alpha{n:03d}"] = result
        return result

    def compute_all(self, data: pl.DataFrame) -> dict[str, pl.Series]:
        """计算所有已实现的 Alpha"""
        d = prepare_alpha_data(data)
        results = {}
        for n, fn in ALPHAS.items():
            try:
                results[f"alpha{n:03d}"] = fn(d)
            except Exception:
                results[f"alpha{n:03d}"] = pl.Series(f"alpha{n:03d}", [np.nan] * len(data))
        self._results = results
        return results

    @property
    def results(self) -> dict[str, pl.Series]:
        return self._results

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(self._results)


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    data = pl.DataFrame({
        "open": 10 + np.random.randn(n).cumsum() * 0.5,
        "high": 10 + np.random.randn(n).cumsum() * 0.5 + 0.3,
        "low": 10 + np.random.randn(n).cumsum() * 0.5 - 0.3,
        "close": 10 + np.random.randn(n).cumsum() * 0.5,
        "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
        "vwap": 10 + np.random.randn(n).cumsum() * 0.4,
        "returns": 10 + np.random.randn(n).cumsum() * 0.5,
    })
    data = data.with_columns(pl.col("close").pct_change(1).alias("returns"))

    alpha = Alpha101()
    results = alpha.compute_all(data)

    print(f"已实现 {len(results)} 个 Alpha 因子:")
    ic_values = []
    for name, val in sorted(results.items()):
        c = np.corrcoef(val.fill_null(0).to_numpy(), data["returns"].fill_null(0).to_numpy())[0, 1]
        if not np.isnan(c):
            ic_values.append((name, c))

    ic_values.sort(key=lambda x: -abs(x[1]))
    for name, ic in ic_values[:10]:
        print(f"  {name}: IC={ic:.4f}")
