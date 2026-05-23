"""分析与回测自测模块"""

import time
from typing import Any

import numpy as np
import polars as pl

from self_test.test_factors import generate_mock_data


def test_backtester() -> dict[str, Any]:
    """测试回测引擎"""
    result = {
        "name": "Backtester",
        "passed": False,
        "details": {},
        "error": None,
    }

    try:
        from qrp.core.backtest import Backtester

        n = 500
        rng = np.random.default_rng(42)
        prices = pl.Series("price", 10.0 + rng.standard_normal(n).cumsum() * 0.1)
        signals = pl.Series("signal", rng.choice([-1, 0, 1], n))

        bt = Backtester(transaction_cost=0.001, slippage=0.001)
        t0 = time.perf_counter()
        res = bt.run(prices, signals)
        elapsed = (time.perf_counter() - t0) * 1000

        result["details"] = {
            "timing_ms": round(elapsed, 2),
            "total_return": round(res.total_return, 4),
            "sharpe_ratio": round(res.sharpe_ratio, 4),
            "max_drawdown": round(res.max_drawdown, 4),
            "win_rate": round(res.win_rate, 4),
            "num_trades": res.num_trades,
        }
        result["passed"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def test_ic_analysis() -> dict[str, Any]:
    """测试 IC 分析"""
    result = {
        "name": "IC Analysis",
        "passed": False,
        "details": {},
        "error": None,
    }

    try:
        from qrp.core.analysis import FactorAnalyzer

        data = generate_mock_data(500)
        rng = np.random.default_rng(42)
        values = pl.Series("factor", rng.standard_normal(500))

        analyzer = FactorAnalyzer(data, values)
        t0 = time.perf_counter()
        ic = analyzer.compute_ic(forward_periods=5)
        elapsed = (time.perf_counter() - t0) * 1000

        quantile_ret = analyzer.quantile_returns(n_quantiles=5)
        ls_ret = analyzer.long_short_return()

        result["details"] = {
            "timing_ms": round(elapsed, 2),
            "ic_mean": round(ic.ic_mean, 4),
            "icir": round(ic.icir, 4),
            "rank_ic_mean": round(ic.rank_ic_mean, 4),
            "rank_icir": round(ic.rank_icir, 4),
            "long_short_return": round(ls_ret, 4),
            "quantile_returns": quantile_ret,
        }
        result["passed"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def test_factor_pipeline() -> dict[str, Any]:
    """测试因子流水线"""
    result = {
        "name": "Factor Pipeline",
        "passed": False,
        "details": {},
        "error": None,
    }

    try:
        from qrp.core.factor import create_base_factors, expand_factors

        data = generate_mock_data(500)
        t0 = time.perf_counter()

        pipe = create_base_factors()
        base_results = pipe.compute(data)
        t1 = time.perf_counter()

        exp_pipe = expand_factors()
        exp_results = exp_pipe.compute(data)
        t2 = time.perf_counter()

        result["details"] = {
            "base_factors": len(base_results),
            "base_timing_ms": round((t1 - t0) * 1000, 2),
            "expanded_factors": len(exp_results),
            "expand_timing_ms": round((t2 - t1) * 1000, 2),
        }
        result["passed"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def test_data_loader() -> dict[str, Any]:
    """测试数据加载器"""
    result = {
        "name": "DataLoader",
        "passed": False,
        "details": {},
        "error": None,
    }

    try:
        from qrp.core.data import DataLoader

        loader = DataLoader(source="mock")
        t0 = time.perf_counter()
        df = loader.load_daily("000001", "20240101", "20240301")
        elapsed = (time.perf_counter() - t0) * 1000

        stock_list = loader.stock_list()

        result["details"] = {
            "timing_ms": round(elapsed, 2),
            "daily_rows": len(df),
            "daily_cols": len(df.columns),
            "source_name": loader.source.get_name(),
            "stock_count": len(stock_list),
        }
        result["passed"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def run_all_analysis_tests() -> list[dict[str, Any]]:
    """运行所有分析与回测测试"""
    tests = [
        test_data_loader,
        test_factor_pipeline,
        test_ic_analysis,
        test_backtester,
    ]

    results = []
    total = len(tests)
    for i, test_fn in enumerate(tests):
        r = test_fn()
        results.append(r)
        status = "✅" if r["passed"] else "❌"
        name = r["name"]
        details = r.get("details", {})
        if "timing_ms" in details:
            print(f"  [{i+1:2d}/{total}] {status} {name:20s} {details.get('timing_ms','?'):>8}ms")
        else:
            print(f"  [{i+1:2d}/{total}] {status} {name}")
        if r.get("error"):
            print(f"        └─ Error: {r['error']}")

    return results
