"""东吴证券 技术因子系列

参考研报：
  《东吴证券"技术分析拥抱选股因子"系列研究（二）：上下影线，蜡烛好还是威廉好》(2020.06.19)
  《东吴证券"波动率选股因子"系列研究（一）：寻找特质波动率中的纯真信息》(2020.05.28)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from qrp.core.factor import Factor


class ShadowLineFactor(Factor):
    """上下影线因子

    参考研报：东吴证券 上下影线因子
    上影线 = 最高价 - max(开盘价, 收盘价)
    下影线 = min(开盘价, 收盘价) - 最低价
    上下影线比率反映买卖力量对比。
    """

    def __init__(self, window: int = 20):
        super().__init__(
            name=f"shadow_line_{window}",
            description=f"上下影线因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        open_np = data["open"].to_numpy()
        high_np = data["high"].to_numpy()
        low_np = data["low"].to_numpy()
        close_np = data["close"].to_numpy()

        # 上影线 = 最高价 - max(开盘价, 收盘价)
        upper_shadow = high_np - np.maximum(open_np, close_np)
        # 下影线 = min(开盘价, 收盘价) - 最低价
        lower_shadow = np.minimum(open_np, close_np) - low_np

        # 总振幅
        total_range = high_np - low_np

        # 上下影线比率
        upper_ratio = upper_shadow / (total_range + 1e-10)
        lower_ratio = lower_shadow / (total_range + 1e-10)

        # 净影线方向（滚动均值）
        net = upper_ratio - lower_ratio
        result = np.full_like(net, np.nan)
        for i in range(self._window, len(net)):
            result[i] = np.mean(net[i - self._window : i])

        return pl.Series("shadow_line", result)


class CandlePowerFactor(Factor):
    """蜡烛力量因子

    基于威廉指标的改进，结合 K 线实体与影线的相对关系。
    """

    def __init__(self, window: int = 14):
        super().__init__(
            name=f"candle_power_{window}",
            description=f"蜡烛力量因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        open_p = data["open"]
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # 实体大小
        body = (close - open_p).abs()
        body_ratio = body / (high - low + 1e-10)

        # 威廉指标
        highest_high = high.rolling_max(window_size=self._window)
        lowest_low = low.rolling_min(window_size=self._window)
        williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10)

        # 蜡烛力量 = 实体比率 × (0.5 - 威廉指标)
        power = body_ratio * (0.5 - williams_r.fill_null(0.5))
        return power.fill_null(0)


class IdiosyncraticVolatilityFactor(Factor):
    """特质波动率因子

    参考研报：东吴证券 特质波动率因子
    特质波动率 = 剔除市场因子后的残差波动率
    低特质波动率异象：低波动股票获得更高收益。
    """

    def __init__(self, window: int = 60):
        super().__init__(
            name=f"iv_vol_{window}",
            description=f"特质波动率因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"]
        n = len(close)
        result = np.full(n, np.nan)

        returns = close.pct_change(1).to_numpy()

        for i in range(self._window, n):
            y = returns[i - self._window : i]
            x = np.arange(len(y))

            # 去趋势
            coeffs = np.polyfit(x, y, 1)
            trend = np.polyval(coeffs, x)
            residual = y - trend

            # 特质波动率 = 残差标准差
            result[i] = np.std(residual)

        return pl.Series("iv_vol", result)


class SerialCorrelationFactor(Factor):
    """偏自相关因子 (CPV 移位版)

    参考研报：CPV因子移位版，价量自相关性中蕴藏的选股信息 (2021.03.01)
    计算价量序列的自相关性，捕捉信息扩散的持续性。
    """

    def __init__(self, lag: int = 5, window: int = 60):
        super().__init__(
            name=f"serial_corr_{lag}_{window}",
            description=f"价量自相关性因子（lag={lag}，window={window}）",
        )
        self._lag = lag
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"]
        volume = data["volume"]

        # 价格自相关性
        close_autocorr = data.select(
            pl.rolling_corr("close", pl.col("close").shift(self._lag), window_size=self._window).alias("__ac1")
        )["__ac1"]

        # 成交量自相关性
        vol_autocorr = data.select(
            pl.rolling_corr("volume", pl.col("volume").shift(self._lag), window_size=self._window).alias("__ac2")
        )["__ac2"]

        # 价量交叉自相关性
        cross_corr = data.select(
            pl.rolling_corr("close", pl.col("volume").shift(self._lag), window_size=self._window).alias("__ac3")
        )["__ac3"]

        # 综合自相关信号
        combined = (close_autocorr + vol_autocorr + cross_corr) / 3.0
        return combined.fill_null(0)


if __name__ == "__main__":
    np.random.seed(42)
    n = 500

    mock = pl.DataFrame(
        {
            "open": 10 + np.random.randn(n).cumsum() * 0.5,
            "high": 10 + np.random.randn(n).cumsum() * 0.5 + 0.4,
            "low": 10 + np.random.randn(n).cumsum() * 0.5 - 0.4,
            "close": 10 + np.random.randn(n).cumsum() * 0.5,
            "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
        }
    )

    factors = [
        ShadowLineFactor(),
        CandlePowerFactor(),
        IdiosyncraticVolatilityFactor(),
        SerialCorrelationFactor(),
    ]

    print("=" * 50)
    print("东吴证券 技术因子系列")
    print("=" * 50)
    for fac in factors:
        values = fac.calculate(mock)
        print(f"{fac.name:30s} mean={values.mean():.4f} std={values.std():.4f}")
