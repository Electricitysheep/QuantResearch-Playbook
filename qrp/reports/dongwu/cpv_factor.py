"""东吴证券 CPV 价量相关性因子

参考研报：
  东吴证券《"技术分析拥抱选股因子"系列研究（一）：高频价量相关性，意想不到的选股因子》(2020.02.23)
  东吴证券《CPV分时版》(2024.12.29)

核心逻辑：
  CPV = 日内分钟级价格与成交量的滚动相关系数
  量价负相关 → 主力吸筹 → 预期正收益
  量价正相关 → 主力出货 → 预期负收益
"""

from __future__ import annotations

from typing import Any

import polars as pl

from qrp.core.analysis import FactorAnalyzer
from qrp.core.factor import Factor


class CPVFactor(Factor):
    """CPV 因子：高频价量相关性

    计算分钟级收盘价与成交量的滚动相关系数。
    支持全时段和分时段两种计算方式。
    """

    def __init__(self, window: int = 60):
        super().__init__(
            name=f"cpv_{window}",
            description=f"CPV 因子：{window}分钟价量滚动相关性",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        """计算 CPV 因子"""
        return data.with_columns(
            pl.rolling_corr("close", "volume", window_size=self._window).alias("__cpv")
        )["__cpv"]

    @classmethod
    def from_report_v1(cls) -> CPVFactor:
        """原始版本：全天240分钟"""
        return cls(window=240)

    @classmethod
    def from_report_v2(cls) -> CPVFactor:
        """改进版本：60分钟窗口"""
        return cls(window=60)


class CPVTimeSegmentFactor(Factor):
    """CPV 分时版因子

    参考研报：东吴证券 CPV 分时版 (2024.12.29)
    将一天交易时间等分成 8 个 30 分钟时段，
    计算各时段价量相关性的标准差。
    最后30分钟版本表现最佳（PV_corr_std_1430）。
    """

    SEGMENTS = [
        ("0930_1000", 0, 30),
        ("1000_1030", 30, 60),
        ("1030_1100", 60, 90),
        ("1100_1130", 90, 120),
        ("1300_1330", 120, 150),
        ("1330_1400", 150, 180),
        ("1400_1430", 180, 210),
        ("1430_1500", 210, 240),
    ]

    def __init__(self, use_last_only: bool = False):
        super().__init__(
            name="cpv_std_8" if not use_last_only else "cpv_std_1430",
            description="CPV 分时版：8时段价量相关性标准差"
            if not use_last_only
            else "CPV 分时版：最后30分钟价量相关性",
        )
        self._use_last_only = use_last_only

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        n = len(data)
        result = pl.Series("cpv_seg", [float("nan")] * n)

        if self._use_last_only:
            # 仅计算最后30分钟
            seg_start, seg_end = 210, min(240, n)
            if seg_end > seg_start:
                seg_data = data.slice(seg_start, seg_end - seg_start)
                if len(seg_data) > 10:
                    cpv = seg_data.with_columns(
                        pl.rolling_corr("close", "volume", window_size=min(30, len(seg_data) // 2)).alias("__cpv_seg")
                    )["__cpv_seg"]
                    for i in range(seg_start, min(seg_end, n)):
                        idx = i - seg_start
                        if idx < len(cpv):
                            result[i] = cpv[idx]
        else:
            # 计算8时段标准差
            seg_vals = []
            for seg_name, start, end in self.SEGMENTS:
                if end > n:
                    continue
                seg_data = data.slice(start, end - start)
                if len(seg_data) > 10:
                    from scipy import stats as _stats
                    corr = _stats.pearsonr(
                        seg_data["close"].to_numpy(),
                        seg_data["volume"].to_numpy(),
                    )[0]
                    seg_vals.append(corr if corr is not None else 0)
                else:
                    seg_vals.append(0)

            if len(seg_vals) >= 4:
                import numpy as np
                std_val = float(np.std(seg_vals))
                for i in range(n):
                    result[i] = std_val

        return result


class RPVFactor(Factor):
    """RPV 因子：新价量相关性因子

    参考研报：东吴证券《新价量相关性因子绩效月报》(2025.03)
    对日内与隔夜信息叠加，通过划分价量四象限，
    利用动量效应切割，以相关性的形式加入成交量信息。
    """

    def __init__(self):
        super().__init__(
            name="rpv",
            description="RPV 新价量相关性因子",
        )

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"]
        volume = data["volume"]
        ret_1 = close.pct_change(1)
        vol_change = volume.pct_change(1)

        # 价量四象限
        pos_price = ret_1 > 0
        pos_vol = vol_change > 0

        # 价量同向（象限I和III）：动量效应
        same_direction = pos_price == pos_vol
        # 价量反向（象限II和IV）：反转效应
        opposite_direction = pos_price != pos_vol

        # 计算相关性信号
        import numpy as np
        from scipy import stats

        close_np = close.to_numpy()
        vol_np = volume.to_numpy()
        n = len(close_np)
        corr_vals = np.full(n, np.nan)
        for i in range(60, n):
            c = stats.pearsonr(close_np[i - 60 : i], vol_np[i - 60 : i])[0]
            corr_vals[i] = c if not np.isnan(c) else 0
        corr_60 = pl.Series("__corr", corr_vals)

        # 叠加动量信息
        result = corr_60.clone()
        signal = same_direction.cast(pl.Float32) * corr_60
        result = signal.fill_null(0)

        return result


class SRVFactor(Factor):
    """SRV 聪明版日频价量相关性因子

    参考研报：东吴证券 SRV 因子 (2025.03)
    日内拆分为上午和下午，计算下午"聪明"时段价量相关性。
    隔夜部分使用昨日最后半小时换手率。
    """

    def __init__(self):
        super().__init__(
            name="srv",
            description="聪明版日频价量相关性因子",
        )

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        close = data["close"]
        volume = data["volume"]
        n = len(data)

        # 简化实现：使用分时段相关性的加权组合
        import numpy as np
        from scipy import stats as _stats
        _cn = close.to_numpy()
        _vn = volume.to_numpy()
        _len = len(_cn)
        _cv = np.full(_len, np.nan)
        for _i in range(60, _len):
            _c = _stats.pearsonr(_cn[_i-60:_i], _vn[_i-60:_i])[0]
            _cv[_i] = _c if not np.isnan(_c) else 0
        all_corr = pl.Series("_ac", _cv)

        # 隔夜动量
        overnight_ret = close.pct_change(1)

        # SRV = 量价相关性 × 隔夜动量调整
        result = all_corr * overnight_ret.sign().cast(pl.Float32)
        return result


# CPV 全系列函数


def run_cpv_analysis(
    data: pl.DataFrame, window: int = 60
) -> dict[str, Any]:
    """运行完整的 CPV 因子分析"""
    factor = CPVFactor(window)
    values = factor.calculate(data)

    analyzer = FactorAnalyzer(data, values)
    ic = analyzer.compute_ic()

    quantiles = analyzer.quantile_returns()
    ls_return = analyzer.long_short_return()

    return {
        "factor": factor,
        "values": values,
        "ic_metrics": ic,
        "quantile_returns": quantiles,
        "long_short_return": ls_return,
    }


def run_cpv_advanced(data: pl.DataFrame) -> dict[str, Any]:
    """运行 CPV 全系列因子分析"""
    factors = {
        "cpv_240": CPVFactor.from_report_v1(),
        "cpv_60": CPVFactor.from_report_v2(),
        "cpv_std_8": CPVTimeSegmentFactor(use_last_only=False),
        "cpv_std_1430": CPVTimeSegmentFactor(use_last_only=True),
        "rpv": RPVFactor(),
        "srv": SRVFactor(),
    }

    results = {}
    for name, fac in factors.items():
        values = fac.calculate(data)
        analyzer = FactorAnalyzer(data, values)
        ic = analyzer.compute_ic(forward_periods=1)
        results[name] = {
            "ic_mean": ic.ic_mean,
            "icir": ic.icir,
        }

    return results


if __name__ == "__main__":
    # 演示：使用模拟数据运行 CPV 分析
    import numpy as np

    np.random.seed(42)
    n = 1000
    base = 10.0

    mock_data = pl.DataFrame(
        {
            "close": base + np.random.randn(n).cumsum() * 0.1,
            "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
            "amount": np.abs(np.random.randn(n) * 1e8 + 5e8),
            "high": base + np.random.randn(n).cumsum() * 0.1 + 0.2,
            "low": base + np.random.randn(n).cumsum() * 0.1 - 0.2,
        }
    )

    results = run_cpv_analysis(mock_data)
    print("=" * 50)
    print("CPV 因子分析结果")
    print("=" * 50)
    print(f"IC 均值: {results['ic_metrics'].ic_mean:.4f}")
    print(f"ICIR: {results['ic_metrics'].icir:.2f}")
    print(f"RankIC 均值: {results['ic_metrics'].rank_ic_mean:.4f}")
    print(f"多空收益: {results['long_short_return']:.4%}")
    print(f"\n分层收益: {results['quantile_returns']}")

    print("\n" + "=" * 50)
    print("CPV 全系列因子对比")
    print("=" * 50)
    advanced = run_cpv_advanced(mock_data)
    for name, metrics in advanced.items():
        print(f"  {name:15s} IC={metrics['ic_mean']:.4f}  ICIR={metrics['icir']:.2f}")
