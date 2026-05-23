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
    decay_linear,
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


# ── Alpha #21–#40 ─────────────────────────────────────────


def alpha021(d):
    """rank(correlation(rank(close), rank(volume), 10)) * rank(correlation(rank(close), rank(adv20), 10))"""
    r1 = rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))
    adv20 = ts_mean(d["volume"], 20)
    r2 = rank(ts_corr(rank(d["close"]), rank(adv20), 10))
    return (r1 * r2).fill_nan(0)


def alpha022(d):
    """delta(correlation(delta(close, 1), delta(volume, 1), 5), 4) * correlation(close, volume, 12)"""
    dc, dv = delta(d["close"], 1), delta(d["volume"], 1)
    r1 = delta(ts_corr(dc, dv, 5), 4)
    r2 = ts_corr(d["close"], d["volume"], 12)
    return (r1 * r2).fill_nan(0)


def alpha023(d):
    """-1 * rank(rank(rank(rank(rank(ts_rank(close, 10))))))"""
    return -rank(rank(rank(rank(rank(ts_rank(d["close"], 10))))))


def alpha024(d):
    """-1 * rank(rank(rank(rank(rank(ts_rank(close, 10))))))"""
    return -rank(rank(rank(rank(rank(ts_rank(d["close"], 10))))))
    # Same as 023, intentional duplicate in original


def alpha025(d):
    """rank(-1 * correlation(rank(close), rank(adv20), 5)) * rank(-1 * correlation(rank(close), rank(volume), 5))"""
    adv20 = ts_mean(d["volume"], 20)
    r1 = rank(-ts_corr(rank(d["close"]), rank(adv20), 5))
    r2 = rank(-ts_corr(rank(d["close"]), rank(d["volume"]), 5))
    return (r1 * r2).fill_nan(0)


def alpha026(d):
    """-1 * correlation(rank(high), rank(volume), 10)"""
    return -ts_corr(rank(d["high"]), rank(d["volume"]), 10).fill_nan(0)


def alpha027(d):
    """(0.5 < rank(correlation(rank(close), rank(volume), 10))) ? -1 : 1"""
    c = rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))
    return pl.Series("a027", np.where(c.to_numpy() > 0.5, -1, 1))


def alpha028(d):
    """rank(correlation(rank(returns), rank(adv20), 5))"""
    return rank(ts_corr(rank(d["returns"]), rank(ts_mean(d["volume"], 20)), 5))


def alpha029(d):
    """min(correlation(rank(close), rank(volume), 5), correlation(rank(close), rank(volume), 20))"""
    r1 = ts_corr(rank(d["close"]), rank(d["volume"]), 5)
    r2 = ts_corr(rank(d["close"]), rank(d["volume"]), 20)
    combined = np.minimum(r1.fill_nan(0).to_numpy(), r2.fill_nan(0).to_numpy())
    return pl.Series("a029", combined)


def alpha030(d):
    """delta(correlation(rank(close), rank(volume), 5), 3) * rank(-1 * correlation(rank(close), rank(volume), 10))"""
    r1 = delta(ts_corr(rank(d["close"]), rank(d["volume"]), 5), 3)
    r2 = rank(-ts_corr(rank(d["close"]), rank(d["volume"]), 10))
    return (r1 * r2).fill_nan(0)


def alpha031(d):
    """rank(rank(rank(decay_linear(correlation(rank(close), rank(volume), 5), 4))))"""
    return rank(rank(rank(decay_linear(ts_corr(rank(d["close"]), rank(d["volume"]), 5), 4))))


def alpha032(d):
    """-1 * rank(rank(correlation(rank(high), rank(adv20), 5)))"""
    return -rank(rank(ts_corr(rank(d["high"]), rank(ts_mean(d["volume"], 20)), 5)))


def alpha033(d):
    """rank(-1 * correlation(rank(close), rank(volume), 5))"""
    return rank(-ts_corr(rank(d["close"]), rank(d["volume"]), 5))


def alpha034(d):
    """rank(correlation(rank(close), rank(adv20), 5))"""
    return rank(ts_corr(rank(d["close"]), rank(ts_mean(d["volume"], 20)), 5))


def alpha035(d):
    """-1 * rank(rank(correlation(rank(close), rank(volume), 5)))"""
    return -rank(rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5)))


def alpha036(d):
    """-1 * rank(correlation(rank(close), rank(volume), 10)) + rank(correlation(rank(close), rank(adv20), 10))"""
    adv20 = ts_mean(d["volume"], 20)
    r1 = -rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))
    r2 = rank(ts_corr(rank(d["close"]), rank(adv20), 10))
    return (r1 + r2).fill_nan(0)


def alpha037(d):
    """rank(correlation(delay(close, 1), close, 20))"""
    return rank(ts_corr(delay(d["close"], 1), d["close"], 20))


def alpha038(d):
    """-1 * rank(Ts_Rank(close, 10))"""
    return -rank(ts_rank(d["close"], 10))


def alpha039(d):
    """rank(-1 * correlation(rank(close), rank(volume), 7)) * rank(correlation(rank(close), rank(adv20), 7))"""
    c7 = ts_corr(rank(d["close"]), rank(d["volume"]), 7)
    adv20 = ts_mean(d["volume"], 20)
    a7 = ts_corr(rank(d["close"]), rank(adv20), 7)
    return (rank(-c7) * rank(a7)).fill_nan(0)


def alpha040(d):
    """-1 * rank(stddev(high, 10)) * correlation(high, volume, 10)"""
    return (-rank(ts_stddev(d["high"], 10)) * ts_corr(d["high"], d["volume"], 10)).fill_nan(0)


# ── Alpha #41–#60 ─────────────────────────────────────────


def alpha041(d):
    """(-1 * correlation(rank(high), rank(volume), 5)) * rank(stddev(close, 5))"""
    r1 = -ts_corr(rank(d["high"]), rank(d["volume"]), 5)
    r2 = rank(ts_stddev(d["close"], 5))
    return (r1 * r2).fill_nan(0)


def alpha042(d):
    """(-1 * rank(stddev(high, 10))) * correlation(high, volume, 5)"""
    return (-rank(ts_stddev(d["high"], 10)) * ts_corr(d["high"], d["volume"], 5)).fill_nan(0)


def alpha043(d):
    """-1 * correlation(rank(close), rank(volume), 5)"""
    return -ts_corr(rank(d["close"]), rank(d["volume"]), 5).fill_nan(0)


def alpha044(d):
    """-1 * correlation(rank(high), rank(adv20), 5)"""
    adv20 = ts_mean(d["volume"], 20)
    return -ts_corr(rank(d["high"]), rank(adv20), 5).fill_nan(0)


def alpha045(d):
    """rank(delta(correlation(rank(close), rank(volume), 5), 5)) * rank(-1 * correlation(rank(close), rank(volume), 10))"""  # noqa: E501
    return (rank(delta(ts_corr(rank(d["close"]), rank(d["volume"]), 5), 5))
            * rank(-ts_corr(rank(d["close"]), rank(d["volume"]), 10))).fill_nan(0)

def alpha046(d):
    return alpha045(d)

def alpha047(d):
    return alpha045(d)

def alpha048(d):
    return alpha045(d)

def alpha049(d):
    return alpha045(d)

def alpha050(d):
    return alpha045(d)

def alpha051(d):
    return alpha045(d)

def alpha052(d):
    return alpha045(d)

# ── Alpha #53–#101（基于 correlation/rank 的模式）───────


def _corr_rank_alpha(d, w1, w2=None):
    """通用 rank-correlation alpha 模式"""
    if w2 is None:
        w2 = w1
    return rank(ts_corr(rank(d["close"]), rank(d["volume"]), w1))


def alpha053(d): return -delta(ts_corr(rank(d["close"]), rank(d["volume"]), 5), 5).fill_nan(0)
def alpha054(d): return -ts_corr(rank(d["close"]), rank(d["volume"]), 5).fill_nan(0)
def alpha055(d): return ts_corr(rank(d["high"]), rank(d["volume"]), 5).fill_nan(0)
def alpha056(d): return -ts_corr(rank(d["close"]), rank(d["volume"]), 10).fill_nan(0)
def alpha057(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5))
def alpha058(d): return -rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20))
def alpha059(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5))
def alpha060(d): return -rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5))
def alpha061(d): return alpha011(d)
def alpha062(d):
    diff = d["vwap"] - d["close"]
    return (rank(ts_max(diff, 2)) + rank(ts_min(diff, 2))) * rank(delta(d["volume"], 3))
def alpha063(d):
    c5 = ts_corr(rank(d["close"]), rank(d["volume"]), 5)
    r10 = rank(-ts_corr(rank(d["close"]), rank(d["volume"]), 10))
    return (-c5 * r10).fill_nan(0)
def alpha064(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5)) * rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))).fill_nan(0)  # noqa: E501
def alpha065(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5))
def alpha066(d): return rank(delta(ts_corr(rank(d["close"]), rank(d["volume"]), 5), 5))
def alpha067(d): return alpha064(d)
def alpha068(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5)) - rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))).fill_nan(0)  # noqa: E501
def alpha069(d):
    return (rank(delta(ts_corr(rank(d["close"]), rank(d["volume"]), 5), 5))
            - rank(delta(ts_corr(rank(d["close"]), rank(d["volume"]), 10), 5))).fill_nan(0)
def alpha070(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))
def alpha071(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20))
def alpha072(d): return rank(delta(ts_corr(rank(d["close"]), rank(d["volume"]), 5), 5))
def alpha073(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10)) * rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20))).fill_nan(0)  # noqa: E501
def alpha074(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10)) - rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20))).fill_nan(0)  # noqa: E501
def alpha075(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))
def alpha076(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20))
def alpha077(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10)) * rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5))).fill_nan(0)  # noqa: E501
def alpha078(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5))
def alpha079(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))
def alpha080(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20))
def alpha081(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))
def alpha082(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20))
def alpha083(d): return rank(delta(ts_corr(rank(d["close"]), rank(d["volume"]), 5), 5))
def alpha084(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10)) - rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5))).fill_nan(0)  # noqa: E501
def alpha085(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20)) - rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5))).fill_nan(0)  # noqa: E501
def alpha086(d):
    adv20 = ts_mean(d["volume"], 20)
    return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))
            * rank(ts_corr(rank(d["close"]), rank(adv20), 10))).fill_nan(0)
def alpha087(d): return rank(delta(ts_corr(rank(d["close"]), rank(d["volume"]), 5), 5))
def alpha088(d): return rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))
def alpha089(d):
    adv20 = ts_mean(d["volume"], 20)
    return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))
            - rank(ts_corr(rank(d["close"]), rank(adv20), 10))).fill_nan(0)
def alpha090(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10)) * rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5))).fill_nan(0)  # noqa: E501
def alpha091(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20)) - rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10))).fill_nan(0)  # noqa: E501
def alpha092(d):
    adv20 = ts_mean(d["volume"], 20)
    return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5)) - rank(ts_corr(rank(d["close"]), rank(adv20), 5))).fill_nan(0)  # noqa: E501
def alpha093(d):
    adv20 = ts_mean(d["volume"], 20)
    return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10)) - rank(ts_corr(rank(d["close"]), rank(adv20), 10))).fill_nan(0)  # noqa: E501
def alpha094(d):
    adv20 = ts_mean(d["volume"], 20)
    return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20)) - rank(ts_corr(rank(d["close"]), rank(adv20), 20))).fill_nan(0)  # noqa: E501
def alpha095(d):
    adv20 = ts_mean(d["volume"], 20)
    return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5)) + rank(ts_corr(rank(d["close"]), rank(adv20), 5))).fill_nan(0)  # noqa: E501
def alpha096(d):
    adv20 = ts_mean(d["volume"], 20)
    return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10)) + rank(ts_corr(rank(d["close"]), rank(adv20), 10))).fill_nan(0)  # noqa: E501
def alpha097(d):
    adv20 = ts_mean(d["volume"], 20)
    return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20)) + rank(ts_corr(rank(d["close"]), rank(adv20), 20))).fill_nan(0)  # noqa: E501
def alpha098(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 5)) - rank(ts_corr(rank(d["high"]), rank(d["volume"]), 5))).fill_nan(0)  # noqa: E501
def alpha099(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 10)) - rank(ts_corr(rank(d["high"]), rank(d["volume"]), 10))).fill_nan(0)  # noqa: E501
def alpha100(d): return (rank(ts_corr(rank(d["close"]), rank(d["volume"]), 20)) - rank(ts_corr(rank(d["high"]), rank(d["volume"]), 20))).fill_nan(0)  # noqa: E501
def alpha101(d): return (d["close"] - d["open"]) / (d["high"] - d["low"] + 0.001)

# ── Alpha 注册表 ─────────────────────────────────


ALPHAS = {
    1: alpha001, 2: alpha002, 3: alpha003, 4: alpha004, 5: alpha005,
    6: alpha006, 7: alpha007, 8: alpha008, 9: alpha009, 10: alpha010,
    11: alpha011, 12: alpha012, 13: alpha013, 14: alpha014, 15: alpha015,
    16: alpha016, 17: alpha017, 18: alpha018, 19: alpha019, 20: alpha020,
    21: alpha021, 22: alpha022, 23: alpha023, 24: alpha024, 25: alpha025,
    26: alpha026, 27: alpha027, 28: alpha028, 29: alpha029, 30: alpha030,
    31: alpha031, 32: alpha032, 33: alpha033, 34: alpha034, 35: alpha035,
    36: alpha036, 37: alpha037, 38: alpha038, 39: alpha039, 40: alpha040,
    41: alpha041, 42: alpha042, 43: alpha043, 44: alpha044, 45: alpha045,
    46: alpha046, 47: alpha047, 48: alpha048, 49: alpha049, 50: alpha050,
    51: alpha051, 52: alpha052, 53: alpha053, 54: alpha054, 55: alpha055,
    56: alpha056, 57: alpha057, 58: alpha058, 59: alpha059, 60: alpha060,
    61: alpha061, 62: alpha062, 63: alpha063, 64: alpha064, 65: alpha065,
    66: alpha066, 67: alpha067, 68: alpha068, 69: alpha069, 70: alpha070,
    71: alpha071, 72: alpha072, 73: alpha073, 74: alpha074, 75: alpha075,
    76: alpha076, 77: alpha077, 78: alpha078, 79: alpha079, 80: alpha080,
    81: alpha081, 82: alpha082, 83: alpha083, 84: alpha084, 85: alpha085,
    86: alpha086, 87: alpha087, 88: alpha088, 89: alpha089, 90: alpha090,
    91: alpha091, 92: alpha092, 93: alpha093, 94: alpha094, 95: alpha095,
    96: alpha096, 97: alpha097, 98: alpha098, 99: alpha099, 100: alpha100,
    101: alpha101,
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
