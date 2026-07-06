"""横截面因子评估 — date × symbol 面板的日度 IC 与分组多空。

券商金工研报中的选股因子（CPV、聪明钱等）本质是横截面命题：
每天在全体股票间按因子值排序、分组、构建多空组合。
本模块提供与研报同构的评估语义，区别于 analysis.py 的单标的时序 IC。

用法::

    analyzer = CrossSectionAnalyzer(panel)   # 列: date, symbol, factor, close
    report = analyzer.run(forward_periods=1, n_quantiles=5)
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class CrossSectionReport:
    """横截面因子评估报告"""

    ic_mean: float
    ic_std: float
    icir: float
    ic_positive_ratio: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_icir: float
    ic_t_stat: float
    long_short_mean: float
    n_dates: int
    ic_by_date: pl.DataFrame
    quantile_returns: pl.DataFrame

    def summary(self) -> dict[str, str | int]:
        return {
            "IC 均值": f"{self.ic_mean:.4f}",
            "ICIR": f"{self.icir:.2f}",
            "IC 正值比例": f"{self.ic_positive_ratio:.1%}",
            "RankIC 均值": f"{self.rank_ic_mean:.4f}",
            "RankICIR": f"{self.rank_icir:.2f}",
            "IC t 统计量": f"{self.ic_t_stat:.2f}",
            "多空日均收益": f"{self.long_short_mean:.4%}",
            "有效截面数": self.n_dates,
        }


class CrossSectionAnalyzer:
    """横截面因子分析器

    输入为长表面板: 每行一个 (date, symbol) 观测，含因子值与收盘价。
    前瞻收益严格按 symbol 分组计算，避免跨标的泄漏。
    """

    def __init__(
        self,
        data: pl.DataFrame,
        date_col: str = "date",
        symbol_col: str = "symbol",
        factor_col: str = "factor",
        price_col: str = "close",
        min_symbols: int = 5,
    ):
        required = [date_col, symbol_col, factor_col, price_col]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"面板缺少必需列: {missing}")
        self.data = data
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.factor_col = factor_col
        self.price_col = price_col
        self.min_symbols = min_symbols

    def forward_returns(self, forward_periods: int = 1) -> pl.DataFrame:
        """按 symbol 分组计算前瞻收益（收盘到收盘）。

        返回原面板附加 fwd_ret 列; 每个 symbol 的末尾 forward_periods 行为 null。
        """
        return (
            self.data.sort([self.symbol_col, self.date_col])
            .with_columns(
                (
                    pl.col(self.price_col).shift(-forward_periods).over(self.symbol_col)
                    / pl.col(self.price_col)
                    - 1
                ).alias("fwd_ret")
            )
        )

    def _valid_panel(self, forward_periods: int) -> pl.DataFrame:
        """带前瞻收益、剔除空值与样本不足截面的面板。"""
        df = (
            self.forward_returns(forward_periods)
            .drop_nulls(["fwd_ret", self.factor_col])
            .with_columns(pl.len().over(self.date_col).alias("_n_symbols"))
            .filter(pl.col("_n_symbols") >= self.min_symbols)
        )
        return df

    def compute_ic(self, forward_periods: int = 1) -> pl.DataFrame:
        """逐日截面 IC: 每个交易日在股票间算因子值与前瞻收益的相关。"""
        df = self._valid_panel(forward_periods)
        return (
            df.group_by(self.date_col)
            .agg(
                pl.corr(self.factor_col, "fwd_ret").alias("ic"),
                pl.corr(self.factor_col, "fwd_ret", method="spearman").alias("rank_ic"),
                pl.len().alias("n_symbols"),
            )
            .drop_nulls(["ic", "rank_ic"])
            .sort(self.date_col)
        )

    def quantile_returns(
        self, forward_periods: int = 1, n_quantiles: int = 5
    ) -> pl.DataFrame:
        """按日分位分组的前瞻收益均值（quantile 0 = 因子最小组）。"""
        df = self._valid_panel(forward_periods).with_columns(
            (
                (pl.col(self.factor_col).rank("ordinal").over(self.date_col) - 1)
                * n_quantiles
                // pl.col("_n_symbols")
            )
            .cast(pl.Int32)
            .alias("quantile")
        )
        return (
            df.group_by("quantile")
            .agg(pl.col("fwd_ret").mean().alias("mean_fwd_ret"))
            .sort("quantile")
        )

    def long_short_returns(
        self, forward_periods: int = 1, n_quantiles: int = 5
    ) -> pl.DataFrame:
        """逐日多空收益: 顶分位组均值 - 底分位组均值。"""
        df = self._valid_panel(forward_periods).with_columns(
            (
                (pl.col(self.factor_col).rank("ordinal").over(self.date_col) - 1)
                * n_quantiles
                // pl.col("_n_symbols")
            )
            .cast(pl.Int32)
            .alias("quantile")
        )
        return (
            df.group_by(self.date_col)
            .agg(
                pl.col("fwd_ret").filter(pl.col("quantile") == n_quantiles - 1).mean().alias("top"),
                pl.col("fwd_ret").filter(pl.col("quantile") == 0).mean().alias("bottom"),
            )
            .with_columns((pl.col("top") - pl.col("bottom")).alias("long_short"))
            .drop_nulls("long_short")
            .sort(self.date_col)
        )

    def run(self, forward_periods: int = 1, n_quantiles: int = 5) -> CrossSectionReport:
        """完整评估: 日度 IC/RankIC + 分位组合 + 多空。"""
        ic_df = self.compute_ic(forward_periods)
        n_dates = len(ic_df)
        if n_dates == 0:
            raise ValueError(
                f"没有满足 min_symbols={self.min_symbols} 的有效截面 — "
                "请检查面板规模或降低 min_symbols"
            )

        ic = ic_df["ic"].to_numpy()
        rank_ic = ic_df["rank_ic"].to_numpy()

        ic_mean, ic_std = float(np.mean(ic)), float(np.std(ic, ddof=1)) if n_dates > 1 else 0.0
        rank_mean = float(np.mean(rank_ic))
        rank_std = float(np.std(rank_ic, ddof=1)) if n_dates > 1 else 0.0

        ls = self.long_short_returns(forward_periods, n_quantiles)

        return CrossSectionReport(
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=_safe_ratio(ic_mean, ic_std),
            ic_positive_ratio=float(np.mean(ic > 0)),
            rank_ic_mean=rank_mean,
            rank_ic_std=rank_std,
            rank_icir=_safe_ratio(rank_mean, rank_std),
            ic_t_stat=_safe_ratio(ic_mean, ic_std) * float(np.sqrt(n_dates)),
            long_short_mean=float(ls["long_short"].mean()) if len(ls) else 0.0,
            n_dates=n_dates,
            ic_by_date=ic_df,
            quantile_returns=self.quantile_returns(forward_periods, n_quantiles),
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    """均值/波动比; 零波动时返回带符号无穷（完美稳定的 IC 序列）。"""
    if denominator > 0:
        return numerator / denominator
    if numerator == 0:
        return 0.0
    return float(np.inf) if numerator > 0 else float(-np.inf)
