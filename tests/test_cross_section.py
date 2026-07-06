"""横截面因子评估测试 — 券商研报语义的选股因子检验。

背景: 此前框架只有单标的时序 IC（因子值与同一标的自身未来收益的相关），
而 CPV 等研报因子本质是横截面选股因子——每天在全体股票间排序分组。
本模块验证 date × symbol 面板上的日度截面 IC / RankIC / 分组多空。
"""

import numpy as np
import polars as pl
import pytest

from qrp.core.cross_section import CrossSectionAnalyzer


def make_panel(n_symbols: int = 10, n_dates: int = 30, daily_ret_per_rank: float = 0.01):
    """构造完美单调面板: 符号 i 的日收益恒为 i * daily_ret_per_rank。

    因子值 = i（符号序号），因此因子完美预测次日截面收益排序。
    """
    rows = []
    for i in range(n_symbols):
        price = 100.0
        for t in range(n_dates):
            rows.append({"date": t, "symbol": f"S{i:02d}", "factor": float(i), "close": price})
            price *= 1 + i * daily_ret_per_rank
    return pl.DataFrame(rows)


class TestForwardReturns:
    def test_forward_returns_do_not_leak_across_symbols(self):
        """前瞻收益必须按 symbol 分组计算 — 全局 shift 会把 A 的价格泄给 B。"""
        df = pl.DataFrame(
            {
                "date": [0, 0, 1, 1],
                "symbol": ["A", "B", "A", "B"],
                "factor": [1.0, 2.0, 1.0, 2.0],
                "close": [100.0, 200.0, 110.0, 100.0],
            }
        )
        analyzer = CrossSectionAnalyzer(df, min_symbols=2)
        fwd = analyzer.forward_returns(forward_periods=1)
        got = {
            (r["symbol"], r["date"]): r["fwd_ret"]
            for r in fwd.drop_nulls("fwd_ret").to_dicts()
        }
        assert got[("A", 0)] == pytest.approx(0.10)
        assert got[("B", 0)] == pytest.approx(-0.50)


class TestCrossSectionalIC:
    def test_perfect_monotonic_factor_has_rank_ic_one(self):
        report = CrossSectionAnalyzer(make_panel()).run(forward_periods=1)
        assert report.rank_ic_mean == pytest.approx(1.0)
        assert report.rank_icir > 10  # 零波动的完美 IC 序列

    def test_reversed_factor_has_rank_ic_minus_one(self):
        panel = make_panel().with_columns((-pl.col("factor")).alias("factor"))
        report = CrossSectionAnalyzer(panel).run(forward_periods=1)
        assert report.rank_ic_mean == pytest.approx(-1.0)

    def test_random_factor_has_near_zero_ic(self):
        rng = np.random.default_rng(7)
        panel = make_panel(n_symbols=20, n_dates=60).with_columns(
            pl.Series("factor", rng.normal(0, 1, 20 * 60))
        )
        report = CrossSectionAnalyzer(panel).run(forward_periods=1)
        assert abs(report.rank_ic_mean) < 0.15

    def test_dates_below_min_symbols_are_excluded(self):
        panel = make_panel(n_symbols=10, n_dates=10)
        # date 0 只保留 3 只票，低于 min_symbols=5
        panel = panel.filter(
            ~((pl.col("date") == 0) & (pl.col("symbol") > "S02"))
        )
        report = CrossSectionAnalyzer(panel, min_symbols=5).run(forward_periods=1)
        # 最后一天无前瞻收益，date 0 被剔除 → 8 个有效截面
        assert report.n_dates == 8


class TestQuantilePortfolios:
    def test_long_short_captures_monotonic_spread(self):
        """Q5-Q1 多空日均收益应约等于 (顶组均值 - 底组均值) 的收益差。"""
        report = CrossSectionAnalyzer(make_panel(n_symbols=10)).run(
            forward_periods=1, n_quantiles=5
        )
        # 顶组 {8,9} 日收益均值 8.5%，底组 {0,1} 均值 0.5% → 多空 ≈ 8%
        assert report.long_short_mean == pytest.approx(0.08, abs=0.005)

    def test_quantile_returns_are_monotonic_for_perfect_factor(self):
        report = CrossSectionAnalyzer(make_panel(n_symbols=10)).run(
            forward_periods=1, n_quantiles=5
        )
        rets = report.quantile_returns.sort("quantile")["mean_fwd_ret"].to_list()
        assert rets == sorted(rets)
        assert len(rets) == 5
