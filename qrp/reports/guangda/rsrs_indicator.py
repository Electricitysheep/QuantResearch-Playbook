"""光大证券 RSRS 阻力支撑相对强度择时指标

参考研报：
  《光大证券-择时系列报告之一：基于阻力支撑相对强度（RSRS）的市场择时》(2017.05.01)
  《光大证券-RSRS择时：回顾与改进》(2019.11.17)
  《中金公司-量化择时系列（1）：金融工程视角下的技术择时艺术》(2021.01.21) - QRS

核心逻辑：
  RSRS = 日内最高价与最低价的线性回归斜率标准化值
  斜率大 → 阻力支撑强度大 → 趋势信号
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from qrp.core.factor import Factor


class RSRSIndicator(Factor):
    """RSRS 阻力支撑相对强度指标

    使用每日最高价和最低价进行滚动线性回归，
    以回归斜率作为阻力支撑相对强度的度量。
    """

    def __init__(self, window: int = 18, ret_threshold: float = 0.7):
        super().__init__(
            name=f"rsrs_{window}",
            description=f"RSRS 阻力支撑相对强度（窗口={window}）",
        )
        self._window = window
        self._ret_threshold = ret_threshold

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        high = data["high"].to_numpy()
        low = data["low"].to_numpy()
        n = len(high)
        result = np.full(n, np.nan)

        for i in range(self._window, n):
            y = high[i - self._window : i]
            x = low[i - self._window : i]

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # 标准化斜率
            result[i] = (slope - np.nanmean(result[max(0, i - 100) : i] if i > 100 else result[self._window : i])) / (
                np.nanstd(result[max(0, i - 100) : i] if i > 100 else result[self._window : i]) + 1e-10
            )

        return pl.Series("rsrs", result)

    def get_signal(self, data: pl.DataFrame) -> pl.Series:
        """生成交易信号

        正信号 → 买入
        负信号 → 卖出/空仓
        """
        rsrs = self.calculate(data)
        signal = rsrs.clone()

        # 正信号买入，负信号卖出
        result = pl.Series("signal", np.zeros(len(signal)))
        rsrs_np = signal.to_numpy()
        for i in range(len(rsrs_np)):
            if rsrs_np[i] > self._ret_threshold:
                result[i] = 1
            elif rsrs_np[i] < -self._ret_threshold:
                result[i] = -1

        return result


class QRSIndicator(Factor):
    """QRS 择时指标（中金公司改进版）

    参考研报：
      《中金公司-量化择时系列（1）：金融工程视角下的技术择时艺术》(2021.01.21)

    在 RSRS 基础上改进，引入分位数回归和右偏态调整。
    """

    def __init__(self, window: int = 18):
        super().__init__(
            name=f"qrs_{window}",
            description=f"QRS 改进型阻力支撑指标（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        high = data["high"].to_numpy()
        low = data["low"].to_numpy()
        n = len(high)
        result = np.full(n, np.nan)

        for i in range(self._window, n):
            y = high[i - self._window : i]
            x = low[i - self._window : i]

            # OLS 回归
            slope, intercept, _, _, _ = stats.linregress(x, y)

            # 右偏态调整（QRS 改进核心）
            skew = stats.skew(y - slope * x - intercept)
            adjusted_slope = slope * (1 + 0.1 * skew)

            # 滚动标准化
            lookback = min(300, max(i, self._window + 1))
            hist_slopes = []
            for j in range(self._window, lookback):
                if j >= self._window and j < i:
                    yh = high[j - self._window : j]
                    xl = low[j - self._window : j]
                    s, _, _, _, _ = stats.linregress(xl, yh)
                    hist_slopes.append(s)

            if len(hist_slopes) > 10:
                mean_s = np.mean(hist_slopes)
                std_s = np.std(hist_slopes)
                result[i] = (adjusted_slope - mean_s) / (std_s + 1e-10)
            else:
                result[i] = 0

        return pl.Series("qrs", result)


if __name__ == "__main__":
    # 演示
    import numpy as np

    np.random.seed(42)
    n = 500

    mock_data = pl.DataFrame(
        {
            "high": 10 + np.random.randn(n).cumsum() * 0.5 + 0.3,
            "low": 10 + np.random.randn(n).cumsum() * 0.5 - 0.3,
            "close": 10 + np.random.randn(n).cumsum() * 0.5,
        }
    )

    rsrs = RSRSIndicator()
    values = rsrs.calculate(mock_data)
    signal = rsrs.get_signal(mock_data)

    print("=" * 50)
    print("RSRS 阻力支撑相对强度")
    print("=" * 50)
    print(f"RSRS 值 (最后10个): {values.tail(10).to_numpy()}")
    buys = (signal == 1).sum()
    sells = (signal == -1).sum()
    print(f"买入信号: {buys}, 卖出信号: {sells}")

    qrs = QRSIndicator()
    qrs_values = qrs.calculate(mock_data)
    print(f"\nQRS 值 (最后10个): {qrs_values.tail(10).to_numpy()}")
