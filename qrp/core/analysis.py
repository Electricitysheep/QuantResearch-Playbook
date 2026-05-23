"""因子分析与评估套件

包含 IC 计算、分组收益、多空收益等标准因子评估方法。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import stats


@dataclass
class ICMetrics:
    """IC 指标"""

    ic_mean: float
    ic_std: float
    icir: float
    ic_positive_ratio: float
    rank_ic_mean: float
    rank_icir: float
    ic_series: pl.Series

    def summary(self) -> dict[str, str]:
        return {
            "IC 均值": f"{self.ic_mean:.4f}",
            "IC 标准差": f"{self.ic_std:.4f}",
            "ICIR": f"{self.icir:.2f}",
            "IC 正值比例": f"{self.ic_positive_ratio:.1%}",
            "RankIC 均值": f"{self.rank_ic_mean:.4f}",
            "RankICIR": f"{self.rank_icir:.2f}",
        }


@dataclass
class FactorReport:
    """因子评估完整报告"""

    ic_metrics: ICMetrics
    factor_name: str
    factor_values: pl.Series
    forward_returns: pl.Series

    def summary(self) -> dict[str, Any]:
        return {
            "因子名称": self.factor_name,
            **self.ic_metrics.summary(),
        }


class FactorAnalyzer:
    """因子分析器

    提供因子 IC 分析、分层回测、多空收益等评估功能。
    """

    def __init__(
        self,
        data: pl.DataFrame,
        factor_values: pl.Series,
        price_col: str = "close",
    ):
        self.data = data
        self.factor_values = factor_values
        self.price_col = price_col

    def compute_ic(self, forward_periods: int = 5) -> ICMetrics:
        """计算 IC 指标

        Args:
            forward_periods: 未来收益率计算周期

        Returns:
            IC 指标
        """
        if self.price_col not in self.data.columns:
            msg = f"Column {self.price_col} not found in data"
            raise ValueError(msg)

        # 计算未来收益率
        prices = self.data[self.price_col].to_numpy()
        forward_ret = np.full_like(prices, np.nan)
        for i in range(len(prices) - forward_periods):
            forward_ret[i] = prices[i + forward_periods] / prices[i] - 1

        pl.Series("forward_ret", forward_ret)
        factor_np = self.factor_values.to_numpy()

        # 日度 IC
        valid = ~(np.isnan(factor_np) | np.isnan(forward_ret))
        ic_values = []
        rank_ic_values = []

        # 按位置分组计算 IC（模拟按日期截面）
        n_days = len(valid) // 240 + 1
        group_size = max(len(valid) // max(n_days, 1), 30)

        for i in range(0, len(valid), group_size):
            end = min(i + group_size, len(valid))
            mask = valid[i:end]
            if mask.sum() < 10:
                continue

            f = factor_np[i:end][mask]
            r = forward_ret[i:end][mask]

            corr = np.corrcoef(f, r)[0, 1]
            if not np.isnan(corr):
                ic_values.append(corr)

            rank_corr = stats.spearmanr(f, r)[0]
            if not np.isnan(rank_corr):
                rank_ic_values.append(rank_corr)

        ic_arr = np.array(ic_values)
        rank_ic_arr = np.array(rank_ic_values)

        ic_mean = float(np.mean(ic_arr)) if len(ic_arr) > 0 else 0
        ic_std = float(np.std(ic_arr)) if len(ic_arr) > 0 else 0

        return ICMetrics(
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=ic_mean / (ic_std + 1e-10),
            ic_positive_ratio=float(np.mean(ic_arr > 0)) if len(ic_arr) > 0 else 0,
            rank_ic_mean=float(np.mean(rank_ic_arr)) if len(rank_ic_arr) > 0 else 0,
            rank_icir=float(np.mean(rank_ic_arr) / (np.std(rank_ic_arr) + 1e-10))
            if len(rank_ic_arr) > 0
            else 0,
            ic_series=pl.Series("ic", ic_arr),
        )

    def quantile_returns(
        self,
        forward_periods: int = 5,
        n_quantiles: int = 5,
    ) -> dict[int, float]:
        """计算分层收益"""
        prices = self.data[self.price_col].to_numpy()
        factor_np = self.factor_values.to_numpy()

        forward_ret = np.full_like(prices, np.nan)
        for i in range(len(prices) - forward_periods):
            forward_ret[i] = prices[i + forward_periods] / prices[i] - 1

        valid = ~(np.isnan(factor_np) | np.isnan(forward_ret))
        f_valid = factor_np[valid]
        r_valid = forward_ret[valid]

        quantiles = np.percentile(f_valid, np.linspace(0, 100, n_quantiles + 1))
        labels = np.digitize(f_valid, quantiles[1:-1])

        result = {}
        for q in range(n_quantiles):
            mask = labels == q
            if mask.sum() > 0:
                result[q + 1] = float(np.mean(r_valid[mask]))

        return result

    def long_short_return(self, forward_periods: int = 5) -> float:
        """计算多空收益"""
        prices = self.data[self.price_col].to_numpy()
        factor_np = self.factor_values.to_numpy()

        forward_ret = np.full_like(prices, np.nan)
        for i in range(len(prices) - forward_periods):
            forward_ret[i] = prices[i + forward_periods] / prices[i] - 1

        valid = ~(np.isnan(factor_np) | np.isnan(forward_ret))
        f_valid = factor_np[valid]
        r_valid = forward_ret[valid]

        # 做多 top 20%，做空 bottom 20%
        threshold_high = np.percentile(f_valid, 80)
        threshold_low = np.percentile(f_valid, 20)

        long_mask = f_valid >= threshold_high
        short_mask = f_valid <= threshold_low

        long_ret = float(np.mean(r_valid[long_mask])) if long_mask.sum() > 0 else 0
        short_ret = float(np.mean(r_valid[short_mask])) if short_mask.sum() > 0 else 0

        return long_ret - short_ret

    def full_report(self, forward_periods: int = 5) -> FactorReport:
        """生成完整因子评估报告"""
        ic_metrics = self.compute_ic(forward_periods)
        prices = self.data[self.price_col].to_numpy()

        forward_ret = np.full_like(prices, np.nan)
        for i in range(len(prices) - forward_periods):
            forward_ret[i] = prices[i + forward_periods] / prices[i] - 1

        return FactorReport(
            ic_metrics=ic_metrics,
            factor_name="",
            factor_values=self.factor_values,
            forward_returns=pl.Series("forward_ret", forward_ret),
        )
