"""国泰君安 - 多因子策略

参考研报：
  《国泰君安-数量化专题之一百二十二：基于CCK模型的股票市场羊群效应研究》(2018.11.28)
"""

import numpy as np
import polars as pl

from qrp.core.factor import Factor


class CCKHerdEffect(Factor):
    """CCK 羊群效应因子"""

    def __init__(self, window: int = 60):
        super().__init__(name=f"cck_{window}",
                         description=f"CCK 羊群效应因子（窗口={window}）")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        ret = data["close"].pct_change(1).to_numpy()
        n = len(ret)
        result = np.full(n, np.nan)
        # CCK: 检测收益率横截面分散度与市场收益率的非线性关系
        for i in range(self._window, n):
            w = ret[max(0, i - self._window) : i]
            w = w[~np.isnan(w)]
            if len(w) > 10:
                market_ret = np.mean(w)
                csd = np.std(w)
                result[i] = csd / (abs(market_ret) + 1e-10)
        return pl.Series("cck", result)


class AnalystRevisionFactor(Factor):
    """分析师预期修正因子"""

    def __init__(self, window: int = 20):
        super().__init__(name=f"analyst_rev_{window}",
                         description=f"分析师预期修正因子（窗口={window}）")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"]
        volume = data["volume"]
        ret = close.pct_change(1)

        rev_signal = ret.rolling_mean(window_size=self._window)
        vol_conf = volume / (volume.rolling_mean(window_size=self._window * 3) + 1e-10)
        return (rev_signal * vol_conf).fill_null(0)
