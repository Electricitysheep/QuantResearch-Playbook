"""东吴证券 CPV 期货版 - DOV 测谎机与最新因子

参考研报：
  《东吴证券-CPV因子期货版1.0》(2020.06.18)
  《东吴证券-CPV因子期货版2.0 - 样本内外的动量反转》(2023.02.13)
  《东吴证券-CPV因子期货版3.0 - CPV测谎机》(2024.09.02)

核心逻辑：
  CPV 期货版将 CPV 因子从选股扩展到期货CTA领域。
  CPV3.0 引入 DOV（成交量异常指标）来判别 PV 信号有效性。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from qrp.core.factor import Factor


class CPVFuturesV1(Factor):
    """CPV 因子期货版 1.0

    基于修正后的持仓量序列与收盘价计算价量相关系数。
    """

    def __init__(self, window: int = 30):
        super().__init__(
            name=f"cpv_futures_v1_{window}",
            description=f"CPV 因子期货版1.0（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"]
        if "open_interest" in data.columns:
            oi = data["open_interest"]
        else:
            oi = data["volume"].cum_sum()

        # 修正持仓量：去除 T+0 交易者噪声
        oi_adj = oi - oi.shift(1).fill_null(0)
        oi_smooth = oi_adj.rolling_mean(window_size=min(5, self._window)).fill_null(0)

        # 计算修正后的价量相关性
        import numpy as np
        from scipy import stats as _s
        cn = close.to_numpy()
        oin = oi_smooth.to_numpy()
        n = len(cn)
        cpv_arr = np.full(n, np.nan)
        for i in range(self._window, n):
            c = _s.pearsonr(cn[i-self._window:i], oin[i-self._window:i])[0]
            cpv_arr[i] = c if not np.isnan(c) else 0
        return pl.Series("cpv", cpv_arr)


class CPVFuturesV3(Factor):
    """CPV 因子期货版 3.0 - CPV 测谎机

    引入 DOV 指标：
    DOV = |ΔOI| / Volume
    DOV 高 → 机构主导 → PV 信号可靠
    DOV 低 → 散户主导 → PV 信号不可靠（反转）
    """

    def __init__(self, window: int = 30, dov_threshold: float = 0.3):
        super().__init__(
            name=f"cpv_futures_v3_{window}",
            description=f"CPV 因子期货版3.0 测谎机（窗口={window}）",
        )
        self._window = window
        self._dov_threshold = dov_threshold

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"]
        volume = data["volume"]

        # DOV 指标
        if "open_interest" in data.columns:
            oi = data["open_interest"]
        else:
            oi = volume.rolling_sum(window_size=self._window)
        oi_change = oi.diff().abs()
        dov = oi_change / (volume + 1e-10)

        # PV 信号（基础价量相关性）
        import numpy as np
        from scipy import stats as _s
        cn = close.to_numpy()
        vn = volume.to_numpy()
        n = len(cn)
        pv_arr = np.full(n, np.nan)
        for i in range(self._window, n):
            c = _s.pearsonr(cn[i-self._window:i], vn[i-self._window:i])[0]
            pv_arr[i] = c if not np.isnan(c) else 0
        pv = pl.Series("pv", pv_arr)

        # 用 DOV 对 PV 信号进行"测谎"调整
        dov_mask = (dov > self._dov_threshold).cast(pl.Float32)
        adjusted = pv * dov_mask
        return adjusted.fill_null(0)


class MarketMicrostructureFactor(Factor):
    """市场微观结构因子

    参考东吴证券系列研报，综合 CPV + DOV + 动量反转的复合信号。
    """

    def __init__(self, cpv_window: int = 30, momentum_window: int = 20):
        super().__init__(
            name=f"microstructure_{cpv_window}_{momentum_window}",
            description="市场微观结构复合因子",
        )
        self._cpv_window = cpv_window
        self._mom_window = momentum_window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"]
        volume = data["volume"]

        ret = close.pct_change(1)
        import numpy as np
        from scipy import stats as _s
        cn = close.to_numpy()
        vn = volume.to_numpy()
        n = len(cn)
        cv_arr = np.full(n, np.nan)
        for i in range(self._cpv_window, n):
            c = _s.pearsonr(cn[i-self._cpv_window:i], vn[i-self._cpv_window:i])[0]
            cv_arr[i] = c if not np.isnan(c) else 0
        cpv = pl.Series("cpv", cv_arr)
        momentum = ret.rolling_mean(window_size=self._mom_window).fill_null(0)

        # 动量 + 价量相关性复合
        composite = cpv * momentum.sign().cast(pl.Float32)
        return composite.fill_null(0)


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    mock = pl.DataFrame({
        "close": 10 + np.random.randn(n).cumsum() * 0.1,
        "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
        "open_interest": np.abs(np.random.randn(n) * 1e5 + 1e6),
    })
    for fac in [CPVFuturesV1(), CPVFuturesV3(), MarketMicrostructureFactor()]:
        v = fac.calculate(mock)
        print(f"{fac.name:35s} mean={v.mean():.4f}")
