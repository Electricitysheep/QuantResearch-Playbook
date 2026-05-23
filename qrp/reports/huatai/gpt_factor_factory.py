"""华泰证券 GPT 因子工厂 & 牛熊指标 & 高阶因子

参考研报：
  《华泰金工-GPT因子工厂2.0：基本面与高频因子挖掘》(2024.09.26)
  《华泰金工-基于CSCV框架的回测过拟合概率》(2019.06.17)
  《华泰金工-波动率与换手率构造牛熊指标》(2019.09.27)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from qrp.core.factor import Factor


class GPTFactorFactory:
    """GPT 因子工厂：多智能体因子挖掘

    参考研报：华泰金工 GPT因子工厂2.0 (2024.09.26)

    架构：
      1. FactorGPT: 因子发现（基于大模型）
      2. CodeGPT: 代码实现
      3. EvalGPT: 因子评估

    此处提供预训练的高频因子的代码实现。
    """

    @staticmethod
    def price_vol_corr_volatility(data: pl.DataFrame, window: int = 60) -> pl.Series:
        """高频价格波动相关性因子

        GPT 命名：高频价格量相关波动因子
        逻辑：价格滚动波动性与成交量滚动波动性间的相关系数
        """
        price_vol = (
            data["close"].pct_change(1).rolling_std(window_size=window)
        )
        volume_vol = (
            data["volume"]
            .pct_change(1)
            .rolling_std(window_size=window)
        )
        import numpy as np
        from scipy import stats as _stats
        pv_np = price_vol.to_numpy()
        vv_np = volume_vol.to_numpy()
        n = len(pv_np)
        corr_vals = np.full(n, np.nan)
        for i in range(window, n):
            c = _stats.pearsonr(pv_np[i - window : i], vv_np[i - window : i])[0]
            corr_vals[i] = c if not np.isnan(c) else 0
        return pl.Series("__corr", corr_vals)

    @staticmethod
    def volume_imbalance_slope(data: pl.DataFrame, window: int = 20) -> pl.Series:
        """成交量不平衡斜率因子

        逻辑：日内买入成交量与卖出成交量的差值趋势
        """
        if "buy_volume" not in data.columns or "sell_volume" not in data.columns:
            # 使用代理：成交量变化的方向性
            ret = data["close"].pct_change(1)
            volume = data["volume"]
            buy_volume_approx = volume * (ret > 0).cast(pl.Float32)
            sell_volume_approx = volume * (ret < 0).cast(pl.Float32)

            imbalance = (buy_volume_approx - sell_volume_approx) / (volume + 1e-10)
            return imbalance.rolling_mean(window_size=window).fill_null(0)

        imbalance = (data["buy_volume"] - data["sell_volume"]) / (
            data["buy_volume"] + data["sell_volume"] + 1e-10
        )
        return imbalance.rolling_mean(window_size=window).fill_null(0)

    @staticmethod
    def high_freq_momentum_decay(data: pl.DataFrame, fast: int = 5, slow: int = 60) -> pl.Series:
        """高频动量衰减因子

        逻辑：短期动量减去长期动量的衰减特征
        """
        fast_ret = data["close"].pct_change(fast)
        slow_ret = data["close"].pct_change(slow)
        decay = (fast_ret - slow_ret / 2).fill_null(0)
        return decay

    @staticmethod
    def price_reversal_intensity(data: pl.DataFrame, window: int = 10) -> pl.Series:
        """价格反转强度因子

        逻辑：连续上涨后的反转概率与连续下跌后的反弹强度
        """
        ret = data["close"].pct_change(1)
        streak_up = (ret > 0).cast(pl.Int32).rolling_sum(window_size=window)
        streak_down = (ret < 0).cast(pl.Int32).rolling_sum(window_size=window)
        return ((streak_up - streak_down) / (streak_up + streak_down + 1e-10)).fill_null(0)

    @classmethod
    def generate_all(cls, data: pl.DataFrame) -> dict[str, pl.Series]:
        """生成所有 GPT 因子"""
        return {
            "gpt_pv_corr_vol": cls.price_vol_corr_volatility(data),
            "gpt_vol_imbalance": cls.volume_imbalance_slope(data),
            "gpt_mom_decay": cls.high_freq_momentum_decay(data),
            "gpt_reversal": cls.price_reversal_intensity(data),
        }


class BullBearIndex(Factor):
    """牛熊指标

    参考研报：华泰金工-波动率与换手率构造牛熊指标 (2019.09.27)
    使用波动率与换手率的综合指标判断市场状态。
    """

    def __init__(self, vol_window: int = 20, turnover_window: int = 60):
        super().__init__(
            name="bull_bear_index",
            description=f"牛熊指标（vol={vol_window}，turnover={turnover_window}）",
        )
        self._vol_window = vol_window
        self._turnover_window = turnover_window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"]
        volume = data["volume"]

        # 波动率
        vol = close.pct_change(1).rolling_std(window_size=self._vol_window)

        # 换手率代理
        if "turnover" in data.columns:
            turnover = data["turnover"]
        else:
            # 使用成交量均值比作为换手率代理
            vol_ma = volume.rolling_mean(window_size=self._turnover_window)
            turnover = volume / (vol_ma + 1e-10)

        # 波动率调节
        vol_percentile = vol.rank("max") / len(vol)
        turn_percentile = turnover.rank("max") / len(turnover)

        # 牛熊指标 = 波动率百分位 - 换手率百分位
        # 高波动 + 低换手 = 熊市信号
        # 低波动 + 高换手 = 牛市信号
        bbi = (vol_percentile - turn_percentile) * -1
        return bbi.fill_null(0)


if __name__ == "__main__":
    np.random.seed(42)
    n = 500

    mock = pl.DataFrame(
        {
            "close": 10 + np.random.randn(n).cumsum() * 0.5,
            "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
            "high": 10 + np.random.randn(n).cumsum() * 0.5 + 0.3,
            "low": 10 + np.random.randn(n).cumsum() * 0.5 - 0.3,
            "amount": np.abs(np.random.randn(n) * 1e8 + 5e8),
        }
    )

    print("=" * 50)
    print("华泰 GPT 因子工厂")
    print("=" * 50)
    factors = GPTFactorFactory.generate_all(mock)
    for name, val in factors.items():
        print(f"  {name:25s} mean={val.mean():.4f} std={val.std():.4f}")

    bbi = BullBearIndex()
    bbi_values = bbi.calculate(mock)
    print(f"\n  {'bull_bear':25s} mean={bbi_values.mean():.4f} std={bbi_values.std():.4f}")
