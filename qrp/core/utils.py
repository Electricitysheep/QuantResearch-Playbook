"""Common utilities for QuantResearch Playbook."""

from __future__ import annotations

import polars as pl


def print_factor_comparison(results: dict[str, dict[str, float]]) -> None:
    """打印因子对比表格"""
    print(f"{'因子名称':25s} {'IC均值':>10s} {'ICIR':>8s} {'RankIC':>10s}")
    print("-" * 55)
    for name, metrics in sorted(results.items(), key=lambda x: abs(x[1].get("ic_mean", 0)), reverse=True):
        ic = metrics.get("ic_mean", 0)
        icir = metrics.get("icir", 0)
        rank_ic = metrics.get("rank_ic_mean", 0)
        print(f"{name:25s} {ic:10.4f} {icir:8.2f} {rank_ic:10.4f}")


def cross_section_rank(data: pl.DataFrame, factor_col: str, group_col: str | None = None) -> pl.Series:
    """横截面排名"""
    if group_col:
        return data[factor_col].rank("average", descending=True).over(data[group_col])
    return data[factor_col].rank("average", descending=True)


def standardize(series: pl.Series, method: str = "zscore") -> pl.Series:
    """标准化"""
    if method == "zscore":
        return (series - series.mean()) / (series.std() + 1e-10)
    elif method == "minmax":
        return (series - series.min()) / (series.max() - series.min() + 1e-10)
    elif method == "rank":
        return series.rank("average") / len(series)
    return series


def winsorize(series: pl.Series, limits: float = 0.05) -> pl.Series:
    """去极值"""
    lower = series.quantile(limits)
    upper = series.quantile(1 - limits)
    return series.clip(lower, upper)


def compute_turnover(signals: pl.Series) -> float:
    """计算换手率"""
    sig_np = signals.to_numpy()
    changes = sum(1 for i in range(1, len(sig_np)) if sig_np[i] != sig_np[i - 1])
    return changes / max(len(sig_np), 1)


if __name__ == "__main__":
    import numpy as np

    mock = pl.DataFrame({"factor1": np.random.randn(100), "factor2": np.random.randn(100)})
    print("标准化测试:")
    for col in mock.columns:
        std = standardize(mock[col])
        print(f"  {col}: mean={std.mean():.6f}, std={std.std():.6f}")
