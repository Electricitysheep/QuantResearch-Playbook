"""开源证券 聪明钱因子模型

参考研报：
  《开源证券-市场微观结构研究系列（3）：聪明钱因子模型的2.0版本》(2020.02.09)
  《开源证券-市场微观结构研究系列（1）：A股反转之力的微观来源》(2019.12.23)
  《开源证券-市场微观结构研究系列（5）：APM因子模型的进阶版》(2020.03.07)
  《开源证券-市场微观结构研究系列（7）：振幅因子的隐藏结构》(2020.05.16)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from qrp.core.factor import Factor


class SmartMoneyFactor(Factor):
    """聪明钱因子

    核心逻辑：聪明钱（机构）的交易行为具有信息优势，
    通过识别大单交易、主动买卖方向来追踪聪明钱流向。
    """

    def __init__(self, window: int = 20):
        super().__init__(
            name=f"smart_money_{window}",
            description=f"聪明钱因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"]
        volume = data["volume"]

        # 识别大单：成交量异常放大
        vol_ma = volume.rolling_mean(window_size=self._window)
        vol_std = volume.rolling_std(window_size=self._window)
        large_trade = (volume - vol_ma) / (vol_std + 1e-10)

        # 价格方向
        ret = close.pct_change(1)
        price_direction = ret.sign().cast(pl.Float32)

        # 聪明钱 = 大单 × 价格方向（主动买卖识别）
        smart_flow = large_trade * price_direction

        # 滚动聚合
        result = smart_flow.rolling_mean(window_size=self._window)
        return result.fill_null(0)


class APMFloorFactor(Factor):
    """APM 因子（改进版）

    参考研报：APM因子模型的进阶版 (2020.03.07)
    APM = Amihud 非流动性指标 + 价格冲击系数
    """

    def __init__(self, window: int = 20):
        super().__init__(
            name=f"apm_{window}",
            description=f"APM 非流动性因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        ret = data["close"].pct_change(1).abs()
        volume = data["volume"]
        amount = data["amount"]

        # Amihud 非流动性指标 = |r| / 成交额
        amihud = ret / (amount + 1e-10)

        # 价格冲击系数 = |r| / 换手率
        turnover = volume / (volume.rolling_mean(window_size=self._window) + 1e-10)
        impact = ret / (turnover + 1e-10)

        # APM = 非流动性 × 价格冲击（滚动均值）
        apm = (amihud * impact).rolling_mean(window_size=self._window)
        return apm.fill_null(0)

    @classmethod
    def apm_v2_improved(cls, window: int = 20) -> APMFloorFactor:
        """APM 2.0：使用修正的非流动性度量"""
        return cls(window=window)


class AmplitudeFactor(Factor):
    """振幅因子

    参考研报：振幅因子的隐藏结构 (2020.05.16)
    日内振幅包含了市场微观结构信息。
    振幅因子 = 日内振幅的滚动调整值
    """

    def __init__(self, window: int = 20):
        super().__init__(
            name=f"amplitude_{window}",
            description=f"振幅因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # 日内振幅
        amplitude = (high - low) / close

        # 振幅的隐藏结构：振幅与价格的负相关性
        import numpy as np
        from scipy import stats as _stats
        amp_np = amplitude.to_numpy()
        ret_np = close.pct_change(1).to_numpy()
        n = len(amp_np)
        corr_vals = np.full(n, np.nan)
        for i in range(self._window, n):
            c = _stats.pearsonr(amp_np[i - self._window : i], ret_np[i - self._window : i])[0]
            corr_vals[i] = c if not np.isnan(c) else 0
        corr_amp_close = pl.Series("__corr", corr_vals)

        # 调整振幅 = 振幅 × (1 - 价幅相关性)
        adjusted = amplitude * (1 - corr_amp_close.fill_null(0))

        return adjusted.fill_null(0)


class MomentumFactor(Factor):
    """A 股动量因子

    参考研报：《A股市场中如何构造动量因子？》(2020.07.21)
    A 股动量效应与美股不同，需要考虑反转特征。
    """

    def __init__(self, window: int = 60, skip_days: int = 21):
        super().__init__(
            name=f"momentum_cn_{window}",
            description=f"A 股动量因子（窗口={window}，跳过={skip_days}）",
        )
        self._window = window
        self._skip_days = skip_days

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"]
        n = len(close)

        # 经典动量 = 过去 N 日收益（跳过最近 M 日）
        ret_window = close.pct_change(self._window)
        ret_skip = close.pct_change(self._skip_days)

        # A 股特有的反转调整
        result = ret_window - 0.5 * ret_skip
        return result.fill_null(0)


if __name__ == "__main__":
    np.random.seed(42)
    n = 500

    mock = pl.DataFrame(
        {
            "close": 10 + np.random.randn(n).cumsum() * 0.5,
            "high": 10 + np.random.randn(n).cumsum() * 0.5 + 0.3,
            "low": 10 + np.random.randn(n).cumsum() * 0.5 - 0.3,
            "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
            "amount": np.abs(np.random.randn(n) * 1e8 + 5e8),
        }
    )

    factors = [
        SmartMoneyFactor(),
        APMFloorFactor(),
        AmplitudeFactor(),
        MomentumFactor(),
    ]

    print("=" * 50)
    print("开源证券 聪明钱因子系列")
    print("=" * 50)
    for fac in factors:
        values = fac.calculate(mock)
        print(f"{fac.name:25s} mean={values.mean():.4f} std={values.std():.4f}")
