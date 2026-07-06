"""回测执行延迟回归测试 — 信号不得赚取生成信号那根 K 线的收益。

回归背景: Backtester 曾直接用 position[i] = signal[i] 乘 returns[i]，
而 signal[i] 是基于 bar i 收盘价算出的 —— 用当日收盘才知道的信号
赚当日收益，构成一根 K 线的前视偏差。
"""

import polars as pl
import pytest

from qrp.core.backtest import Backtester


def _run(prices: list[float], signals: list[float]):
    bt = Backtester(transaction_cost=0.0, slippage=0.0)
    return bt.run(
        pl.Series("close", prices),
        pl.Series("signal", signals),
    )


class TestExecutionLag:
    def test_signal_takes_effect_next_bar(self):
        """bar 1 收盘发出的 BUY 信号，应完整吃到 bar 2 的 +10%。"""
        # returns:      [–, 0%, +10%, 0%]
        result = _run(
            prices=[100.0, 100.0, 110.0, 110.0],
            signals=[0.0, 1.0, 0.0, 0.0],
        )
        assert result.total_return == pytest.approx(0.10, rel=1e-9)

    def test_perfect_foresight_signal_earns_nothing(self):
        """完美预知同根 K 线涨跌的信号不应获利 —— 前视偏差探测器。

        收益率交替 +10%/-10%，信号恰好在上涨 K 线为 1、下跌 K 线为 0。
        若存在前视，策略每次持仓都恰好踩中 +10%（总收益 +21%）;
        正确的一根 K 线延迟下，持仓落在其后的 -10% 上（总收益 -19%）。
        """
        result = _run(
            prices=[100.0, 110.0, 99.0, 108.9, 98.01],
            signals=[0.0, 1.0, 0.0, 1.0, 0.0],
        )
        assert result.total_return == pytest.approx(0.9 * 0.9 - 1, rel=1e-9)
        assert result.total_return < 0, "同 K 线信号不应捕获同 K 线收益"
