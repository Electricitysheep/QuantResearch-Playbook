"""性能基准测试 - 对比 Polars 与 Pandas 在量化计算中的性能"""
import time
import polars as pl
import numpy as np

N = 100_000  # 10万行数据

def generate_data(n=N):
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "close": 10.0 + rng.standard_normal(n).cumsum() * 0.1,
        "volume": np.abs(rng.standard_normal(n) * 1e6 + 5e6),
        "high": 10.0 + rng.standard_normal(n).cumsum() * 0.1 + 0.2,
        "low": 10.0 + rng.standard_normal(n).cumsum() * 0.1 - 0.2,
        "amount": np.abs(rng.standard_normal(n) * 1e8 + 5e8),
    })

def benchmark(label, fn, data, n_runs=5):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(data)
        times.append(time.perf_counter() - t0)
    avg = np.mean(times)
    print(f"  {label:30s} {avg*1000:8.2f}ms")

def main():
    print("=" * 50)
    print("QuantResearch-Playbook 性能基准测试")
    print(f"数据量: {N:,} 行, 重复: 5次取平均值")
    print("=" * 50)

    data = generate_data()

    # 测试 Polars 核心操作
    print("\n--- Polars 操作 ---")
    benchmark("读取列", lambda d: d["close"].to_numpy(), data)
    benchmark("diff", lambda d: d["close"].diff(), data)
    benchmark("pct_change", lambda d: d["close"].pct_change(5), data)
    benchmark("rolling_mean(60)", lambda d: d["close"].rolling_mean(60), data)
    benchmark("rolling_std(60)", lambda d: d["close"].rolling_std(60), data)
    benchmark("group_corr(60)", lambda d: d.with_columns(pl.rolling_corr("close", "volume", window_size=60)), data)

    # 测试因子计算
    print("\n--- 因子计算 ---")
    from qrp.core.factor import (FactorMomentum, FactorRSI, FactorVwapDev,
                                  FactorVolumeVol, FactorCorrPriceVolume)
    for name, fac in [("Momentum", FactorMomentum(60)), ("RSI", FactorRSI(14)),
                      ("VWAP Dev", FactorVwapDev(60)), ("Vol Vol", FactorVolumeVol(60)),
                      ("CPV Corr", FactorCorrPriceVolume(60))]:
        benchmark(name, fac.calculate, data)

    # 测试回测
    print("\n--- 回测引擎 ---")
    from qrp.core.backtest import Backtester
    signals = pl.Series("s", np.random.choice([-1, 0, 1], N))
    bt = Backtester()
    benchmark("Backtester.run", lambda d: bt.run(d["close"], signals), data)

    print("\n完成!")

if __name__ == "__main__":
    main()
