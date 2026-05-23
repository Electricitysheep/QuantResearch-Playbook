"""东方证券 - 量价买卖压力因子

参考研报：《东方证券-因子选股系列研究六十：基于量价关系度量股票的买卖压力》(2019.10.29)

核心逻辑：通过量价关系度量市场买卖压力，识别机构行为。
"""

import polars as pl

from qrp.core.factor import Factor


class BuySellPressureFactor(Factor):
    """买卖压力因子"""

    def __init__(self, window: int = 20):
        super().__init__(name=f"bs_pressure_{window}",
                         description=f"量价买卖压力因子（窗口={window}）")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        ret = data["close"].pct_change(1)
        volume = data["volume"]
        amount = data["amount"]

        buy_pressure = (ret > 0).cast(pl.Float32) * volume / (amount + 1e-10)
        sell_pressure = (ret < 0).cast(pl.Float32) * volume / (amount + 1e-10)
        net_pressure = (buy_pressure - sell_pressure).rolling_mean(window_size=self._window)
        return net_pressure.fill_null(0)


class VolumePriceTrendFactor(Factor):
    """量价趋势因子"""

    def __init__(self, window: int = 20):
        super().__init__(name=f"vpt_{window}",
                         description=f"量价趋势因子（窗口={window}）")
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        ret = data["close"].pct_change(1)
        volume = data["volume"]
        vpt = (ret * volume).rolling_sum(window_size=self._window)
        return vpt.fill_null(0)
