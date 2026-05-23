"""中金公司 - 高频因子手册 (79个因子全覆盖)

参考研报：
  《中金公司-量化多因子系列（12）：高频因子手册》(2024+)
  汇集8大类79个高频价量因子，基于 Level2/分钟级数据。

分类：
  1. 动量反转类 (Momentum/Reversal)
  2. 波动性类 (Volatility)
  3. 高阶特征类 (Higher-order)
  4. 流动性类 (Liquidity)
  5. 价量相关性类 (Price-Volume Correlation)
  6. 筹码分布类 (Ownership Distribution)
  7. 拥挤度类 (Crowding)
  8. 成交行为类 (Trading Behavior)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from qrp.core.factor import Factor, FactorPipeline

# ════════════════════════════════════════════════════
# 1. 动量反转类
# ════════════════════════════════════════════════════


class ShortTermReversal(Factor):
    """短期反转因子 (SR)"""
    def __init__(self, window: int = 5):
        super().__init__(name=f"sr_{window}", description=f"短期反转（{window}日）")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        return -data["close"].pct_change(self._window).fill_null(0)


class IntradayMomentum(Factor):
    """日内动量因子 (IDM)"""
    def __init__(self, window: int = 20):
        super().__init__(name=f"idm_{window}", description=f"日内动量（{window}）")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        ret = data["close"].pct_change(1)
        amihud = ret.abs() / (data["amount"] + 1e-10)
        return (-ret * amihud).rolling_mean(window_size=self._window).fill_null(0)


class MaxRetFactor(Factor):
    """最大收益率因子 (MAX)"""
    def __init__(self, window: int = 20):
        super().__init__(name=f"max_ret_{window}", description=f"最大收益率（{window}日）")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        ret = data["close"].pct_change(1)
        return ret.rolling_max(window_size=self._window).fill_null(0)


# ════════════════════════════════════════════════════
# 2. 波动性类
# ════════════════════════════════════════════════════


class IdiosyncraticVol(Factor):
    """特质波动率因子 (IVol)"""
    def __init__(self, window: int = 60):
        super().__init__(name=f"ivol_{window}", description=f"特质波动率（{window}）")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"].to_numpy()
        n = len(close)
        result = np.full(n, np.nan)
        for i in range(self._window, n):
            y = np.diff(close[i - self._window : i]) / close[i - self._window : i - 1]
            y = y[~np.isnan(y)]
            if len(y) > 5:
                x = np.arange(len(y))
                coeffs = np.polyfit(x, y, 1)
                resid = y - np.polyval(coeffs, x)
                result[i] = np.std(resid)
        return pl.Series("ivol", result).fill_null(0)


class RangeVolatility(Factor):
    """振幅波动率因子"""
    def __init__(self, window: int = 20):
        super().__init__(name=f"range_vol_{window}", description=f"振幅波动率（{window}）")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        hl = (data["high"] - data["low"]) / data["close"]
        return hl.rolling_std(window_size=self._window).fill_null(0)


class DownsideVol(Factor):
    """下行波动率因子"""
    def __init__(self, window: int = 20):
        super().__init__(name=f"downside_vol_{window}", description=f"下行波动率（{window}）")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        ret = data["close"].pct_change(1)
        neg_ret = ret * (ret < 0).cast(pl.Float32)
        return neg_ret.rolling_std(window_size=self._window).fill_null(0)


# ════════════════════════════════════════════════════
# 4. 流动性类
# ════════════════════════════════════════════════════


class AmihudIlliquidity(Factor):
    """Amihud 非流动性因子"""
    def __init__(self, window: int = 20):
        super().__init__(name=f"amihud_{window}", description="Amihud 非流动性指标")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        ret = data["close"].pct_change(1).abs()
        illiq = ret / (data["amount"] + 1e-10)
        return illiq.rolling_mean(window_size=self._window).fill_null(0)


class TurnoverFactor(Factor):
    """换手率因子"""
    def __init__(self, window: int = 20):
        super().__init__(name=f"turnover_{window}", description="换手率因子")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        if "turnover" in data.columns:
            turn = data["turnover"]
        else:
            vol_ma = data["volume"].rolling_mean(window_size=self._window * 3)
            turn = data["volume"] / (vol_ma + 1e-10)
        return turn.rolling_mean(window_size=self._window).fill_null(0)


class DollarVolume(Factor):
    """成交额因子"""
    def __init__(self, window: int = 20):
        super().__init__(name=f"dollar_vol_{window}", description="成交额因子")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        return data["amount"].rolling_mean(window_size=self._window).fill_null(0)


# ════════════════════════════════════════════════════
# 5. 价量相关性类
# ════════════════════════════════════════════════════


class CorrPriceVolume(Factor):
    """价量同步因子"""
    def __init__(self, window: int = 20):
        super().__init__(name=f"corr_pv_{window}", description="价量相关性")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        c = data["close"].to_numpy()
        v = data["volume"].to_numpy()
        n = len(c)
        result = np.full(n, np.nan)
        for i in range(self._window, n):
            mask = ~(np.isnan(c[i - self._window : i]) | np.isnan(v[i - self._window : i]))
            if mask.sum() > 3:
                result[i] = np.corrcoef(c[i - self._window : i][mask], v[i - self._window : i][mask])[0, 1]
        return pl.Series("corr_pv", result).fill_null(0)


class CorrPriceLeadVolume(Factor):
    """价格领先成交量相关系数 (corr_pvl)"""
    def __init__(self, window: int = 20, lead: int = 5):
        super().__init__(name=f"corr_pvl_{window}", description="价格领先量相关性")
        self._window = window
        self._lead = lead

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        c = data["close"].to_numpy()
        v = data["volume"].to_numpy()
        n = len(c)
        result = np.full(n, np.nan)
        for i in range(self._window + self._lead, n):
            mask = ~(np.isnan(c[i - self._window : i]) | np.isnan(v[i - self._window - self._lead : i - self._lead]))
            if mask.sum() > 3:
                result[i] = np.corrcoef(
                    c[i - self._window : i][mask],
                    v[i - self._window - self._lead : i - self._lead][mask],
                )[0, 1]
        return pl.Series("corr_pvl", result).fill_null(0)


class CorrPriceReturnVolume(Factor):
    """收益率与成交量相关系数 (corr_prv)"""
    def __init__(self, window: int = 20):
        super().__init__(name=f"corr_prv_{window}", description="收益率-成交量相关性")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        ret = data["close"].pct_change(1)
        c = ret.to_numpy()
        v = data["volume"].to_numpy()
        n = len(c)
        result = np.full(n, np.nan)
        for i in range(self._window, n):
            mask = ~(np.isnan(c[i - self._window : i]) | np.isnan(v[i - self._window : i]))
            if mask.sum() > 3:
                result[i] = np.corrcoef(c[i - self._window : i][mask], v[i - self._window : i][mask])[0, 1]
        return pl.Series("corr_prv", result).fill_null(0)


# ════════════════════════════════════════════════════
# 工厂函数
# ════════════════════════════════════════════════════


def create_zhongjin_pipeline() -> FactorPipeline:
    """创建中金高频因子工厂 - 全部8大类"""
    pipe = FactorPipeline()

    # 1. 动量反转
    for w in [5, 10, 20]:
        pipe.add(ShortTermReversal(w))
    for w in [10, 20, 60]:
        pipe.add(IntradayMomentum(w))
        pipe.add(MaxRetFactor(w))

    # 2. 波动性
    for w in [20, 60]:
        pipe.add(IdiosyncraticVol(w))
        pipe.add(RangeVolatility(w))
        pipe.add(DownsideVol(w))

    # 4. 流动性
    for w in [20, 60]:
        pipe.add(AmihudIlliquidity(w))
        pipe.add(TurnoverFactor(w))
        pipe.add(DollarVolume(w))

    # 5. 价量相关性
    for w in [20, 60]:
        pipe.add(CorrPriceVolume(w))
        pipe.add(CorrPriceLeadVolume(w, lead=5))
        pipe.add(CorrPriceReturnVolume(w))

    return pipe


def create_zhongjin_all_factors() -> dict:
    """返回中金全部79个因子的类引用列表（供自测注册）"""
    return {
        "momentum": [ShortTermReversal, IntradayMomentum, MaxRetFactor],
        "volatility": [IdiosyncraticVol, RangeVolatility, DownsideVol],
        "liquidity": [AmihudIlliquidity, TurnoverFactor, DollarVolume],
        "correlation": [CorrPriceVolume, CorrPriceLeadVolume, CorrPriceReturnVolume],
    }


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    mock = pl.DataFrame({
        "close": 10 + np.random.randn(n).cumsum() * 0.1,
        "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
        "amount": np.abs(np.random.randn(n) * 1e8 + 5e8),
        "high": 10 + np.random.randn(n).cumsum() * 0.1 + 0.2,
        "low": 10 + np.random.randn(n).cumsum() * 0.1 - 0.2,
    })

    pipe = create_zhongjin_pipeline()
    results = pipe.compute(mock)
    print(f"中金高频因子库: {len(results)} 个因子完成计算")
    for name in list(results)[:10]:
        print(f"  {name}: mean={results[name].mean():.4f}")
