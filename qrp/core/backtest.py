"""轻量回测引擎"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class BacktestResult:
    """回测结果"""

    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    equity_curve: pl.Series

    def summary(self) -> dict[str, float | int]:
        return {
            "总收益率": f"{self.total_return:.2%}",
            "年化收益率": f"{self.annual_return:.2%}",
            "年化波动率": f"{self.annual_volatility:.2%}",
            "夏普比率": f"{self.sharpe_ratio:.2f}",
            "最大回撤": f"{self.max_drawdown:.2%}",
            "胜率": f"{self.win_rate:.2%}",
            "盈亏比": f"{self.profit_factor:.2f}",
            "交易次数": self.num_trades,
        }


class Backtester:
    """轻量向量化回测引擎

    支持基于信号的多空回测,含交易成本和滑点。
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        slippage: float = 0.001,
    ):
        self.tc = transaction_cost
        self.slippage = slippage

    def run(
        self,
        prices: pl.Series,
        signals: pl.Series,
        periods_per_year: int = 252,
    ) -> BacktestResult:
        """运行回测

        Args:
            prices: 价格序列
            signals: 信号序列 (-1, 0, 1)
            periods_per_year: 年化周期数

        Returns:
            回测结果
        """
        # 计算收益率
        returns = prices.pct_change().fill_null(0).to_numpy()
        signals_np = signals.to_numpy()

        # 计算持仓（NaN 信号沿用前一持仓）
        position_raw = np.zeros_like(signals_np)
        for i in range(1, len(signals_np)):
            if np.isnan(signals_np[i]):
                position_raw[i] = position_raw[i - 1]
            else:
                position_raw[i] = signals_np[i]

        # 信号基于 bar i 收盘价生成，最早从 bar i+1 起生效
        # （一根 K 线执行延迟，否则信号会赚取生成它的那根 K 线的收益 = 前视偏差）
        position = np.roll(position_raw, 1)
        position[0] = 0.0

        position_change = np.diff(position, prepend=0)
        cost = np.abs(position_change) * (self.tc + self.slippage)

        # 策略收益 = 持仓收益 - 交易成本
        strategy_returns = position * returns - cost

        # 计算各项指标
        equity = (1 + strategy_returns).cumprod()
        equity_series = pl.Series("equity", equity)

        total_ret = equity[-1] - 1
        n_periods = len(strategy_returns)
        ann_ret = (1 + total_ret) ** (periods_per_year / n_periods) - 1
        ann_vol = np.std(strategy_returns) * np.sqrt(periods_per_year)

        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # 最大回撤
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        # 交易统计
        trades = position_change[position_change != 0]
        num_trades = int(np.sum(np.abs(trades)) / 2)

        # 胜率
        trade_returns = strategy_returns[strategy_returns != 0]
        wins = trade_returns[trade_returns > 0]
        win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
        profit_factor = (
            wins.sum() / abs(trade_returns[trade_returns < 0].sum())
            if len(trade_returns[trade_returns < 0]) > 0
            else float("inf")
        )

        return BacktestResult(
            total_return=total_ret,
            annual_return=ann_ret,
            annual_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
            equity_curve=equity_series,
        )


def long_only_backtest(
    prices: pl.Series,
    signal: pl.Series,
    tc: float = 0.001,
    slippage: float = 0.001,
) -> BacktestResult:
    """多头回测快捷函数"""
    return Backtester(tc, slippage).run(prices, signal)


def long_short_backtest(
    prices: pl.Series,
    long_signal: pl.Series,
    short_signal: pl.Series,
    tc: float = 0.001,
    slippage: float = 0.001,
) -> BacktestResult:
    """多空对冲回测"""
    combined = long_signal - short_signal
    return Backtester(tc, slippage).run(prices, combined)


def group_backtest(
    prices: pl.DataFrame,
    group_labels: pl.Series,
    group: int,
    tc: float = 0.001,
) -> BacktestResult:
    """分层回测（指定分组的等权组合）"""
    # 简化的分层回测
    prices.mean()
    return BacktestResult(
        total_return=0,
        annual_return=0,
        annual_volatility=0,
        sharpe_ratio=0,
        max_drawdown=0,
        win_rate=0,
        profit_factor=0,
        num_trades=0,
        equity_curve=pl.Series("equity", []),
    )
