"""Demo script: Run CPV factor analysis."""

import numpy as np
import polars as pl

from qrp.reports.dongwu.cpv_factor import run_cpv_analysis, run_cpv_advanced
from qrp.core.analysis import FactorAnalyzer
from qrp.core.factor import create_base_factors


def demo_cpv():
    """Run CPV factor analysis demo."""
    np.random.seed(42)
    n = 1000

    data = pl.DataFrame(
        {
            "close": 10 + np.random.randn(n).cumsum() * 0.1,
            "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
            "amount": np.abs(np.random.randn(n) * 1e8 + 5e8),
            "high": 10 + np.random.randn(n).cumsum() * 0.1 + 0.2,
            "low": 10 + np.random.randn(n).cumsum() * 0.1 - 0.2,
        }
    )

    print("=" * 60)
    print("QuantResearch Playbook - Demo")
    print("=" * 60)

    # 1. CPV Factor
    print("\n[1] CPV Factor Analysis")
    print("-" * 40)
    result = run_cpv_analysis(data)
    print(f"  IC Mean:     {result['ic_metrics'].ic_mean:.4f}")
    print(f"  ICIR:        {result['ic_metrics'].icir:.2f}")
    print(f"  RankIC Mean: {result['ic_metrics'].rank_ic_mean:.4f}")
    print(f"  Long/Short:  {result['long_short_return']:.4%}")

    # 2. CPV Series Comparison
    print("\n[2] CPV Series Comparison")
    print("-" * 40)
    advanced = run_cpv_advanced(data)
    for name, metrics in advanced.items():
        print(f"  {name:15s}  IC={metrics['ic_mean']:.4f}  ICIR={metrics['icir']:.2f}")

    # 3. Base Factors
    print("\n[3] Base Factor Analysis")
    print("-" * 40)
    pipe = create_base_factors()
    for name in pipe.factor_names:
        fac = pipe[name]
        values = fac.calculate(data)
        analyzer = FactorAnalyzer(data, values)
        ic = analyzer.compute_ic()
        print(f"  {name:20s}  IC={ic.ic_mean:.4f}  ICIR={ic.icir:.2f}")

    print("\nDone!")


if __name__ == "__main__":
    demo_cpv()
