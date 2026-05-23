"""国信证券 / 招商证券 / 东北证券 / 申万宏源 / 浙商证券 策略框架

各策略在研报目录 docs/REPORTS_CATALOG.md 中维护，
此处为快速展位（stub），方便社区贡献者按模板扩展。
"""

import numpy as np
import polars as pl

from qrp.core.factor import Factor

# ── 国信证券 ──


class WaveletAnalysis(Factor):
    """小波分析择时 (国信证券 2010)"""
    def __init__(self):
        super().__init__(name="wavelet", description="小波分析择时指标（stub）")

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        return pl.Series("wavelet", np.zeros(len(data)))


class VolatilityAsymmetry(Factor):
    """波动率单向差值 (国信证券 2015)"""
    def __init__(self, window: int = 20):
        super().__init__(name=f"vol_asym_{window}", description="波动率非对称因子")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        ret = data["close"].pct_change(1)
        up_vol = (ret * (ret > 0).cast(pl.Float32)).rolling_std(window_size=self._window)
        dn_vol = (ret * (ret < 0).cast(pl.Float32)).rolling_std(window_size=self._window)
        return ((up_vol - dn_vol) / (up_vol + dn_vol + 1e-10)).fill_null(0)


# ── 招商证券 ──


class MultiFactorEnhance(Factor):
    """多因子指数增强 (招商证券)"""
    def __init__(self):
        super().__init__(name="mfe", description="多因子指数增强（stub）")

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        return pl.Series("mfe", np.zeros(len(data)))


# ── 东北证券 ──


class DiffusionIndex(Factor):
    """扩散指标择时 (东北证券 2019)"""
    def __init__(self, window: int = 20):
        super().__init__(name=f"diffusion_{window}", description="扩散指标择时")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"].to_numpy()
        n = len(close)
        result = np.full(n, np.nan)
        for i in range(self._window * 2, n):
            rising = np.sum(np.diff(close[i - self._window : i]) > 0)
            result[i] = rising / self._window
        return pl.Series("diffusion", result)


# ── 申万宏源 ──


class MasterValue(Factor):
    """大师价值投资系列 (申万宏源)"""
    def __init__(self):
        super().__init__(name="master_value", description="申万大师价值选股（stub）")

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        return pl.Series("mv", np.zeros(len(data)))


# ── 浙商证券 ──


class GoldStockEnhance(Factor):
    """金股组合增强策略 (浙商证券 2022)"""
    def __init__(self):
        super().__init__(name="gold_stock", description="金股组合增强策略（stub）")

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        return pl.Series("gs", np.zeros(len(data)))
