"""因子自测模块 - 用模拟数据测试所有因子实现"""

import time
from typing import Any

import numpy as np
import polars as pl


def generate_mock_data(n: int = 500) -> pl.DataFrame:
    """生成标准模拟数据（统一种子保证可复现）"""
    rng = np.random.default_rng(42)
    base = 10.0
    return pl.DataFrame(
        {
            "close": base + rng.standard_normal(n).cumsum() * 0.1,
            "volume": np.abs(rng.standard_normal(n) * 1e6 + 5e6),
            "amount": np.abs(rng.standard_normal(n) * 1e8 + 5e8),
            "high": base + rng.standard_normal(n).cumsum() * 0.1 + 0.2,
            "low": base + rng.standard_normal(n).cumsum() * 0.1 - 0.2,
            "open": base + rng.standard_normal(n).cumsum() * 0.1,
        }
    )


def generate_financial_data(n: int = 200) -> pl.DataFrame:
    """生成含财务数据的模拟数据"""
    base = generate_mock_data(n)
    rng = np.random.default_rng(42)
    return base.with_columns([
        pl.Series("roa", rng.standard_normal(n) * 0.01 + 0.05),
        pl.Series("cfo", rng.standard_normal(n) * 0.01 + 0.03),
        pl.Series("lev", rng.random(n) * 0.5),
        pl.Series("liquid", rng.random(n) * 2 + 1),
        pl.Series("margin", rng.standard_normal(n) * 0.02 + 0.15),
        pl.Series("turnover", rng.standard_normal(n) * 0.1 + 0.8),
    ])


# ── 测试用例注册 ──
TEST_FACTORS: list = []


def register(cls, name: str, data_type: str = "basic", **kwargs):
    TEST_FACTORS.append((cls, name, data_type, kwargs))


def _r(cls, name, dt="basic", kw=None):
    """Shorthand register"""
    TEST_FACTORS.append((cls, name, dt, kw or {}))


def run_factor_test(factor_cls, name: str, data_type: str, kwargs: dict) -> dict[str, Any]:
    """运行单个因子测试"""
    result = {
        "name": name,
        "class": factor_cls.__name__,
        "passed": False,
        "error": None,
        "shape_match": False,
        "has_nan": False,
        "mean": None,
        "std": None,
        "timing_ms": 0,
    }

    data = generate_mock_data() if data_type == "basic" else generate_financial_data()

    try:
        fac = factor_cls(**kwargs) if kwargs else factor_cls()
        t0 = time.perf_counter()
        values = fac.calculate(data)
        elapsed = (time.perf_counter() - t0) * 1000

        result["timing_ms"] = round(elapsed, 2)
        result["shape_match"] = len(values) == len(data)
        result["mean"] = round(float(values.mean()), 6)
        result["std"] = round(float(values.std()), 6)
        result["has_nan"] = int(values.is_null().sum()) > 0

        result["passed"] = result["shape_match"]
    except Exception as e:
        result["error"] = str(e)

    return result


def run_all_factor_tests() -> list[dict[str, Any]]:
    """运行所有注册的因子测试"""

    # ── 注册基础因子 ──
    from qrp.core.factor import (
        FactorCorrPriceVolume,
        FactorDrawdown,
        FactorMomentum,
        FactorRSI,
        FactorVolumeVol,
        FactorVwapDev,
    )

    _r(FactorVwapDev, "vwap_dev_60", "basic", {"window": 60})
    _r(FactorMomentum, "momentum_60", "basic", {"window": 60})
    _r(FactorRSI, "rsi_14", "basic", {"window": 14})
    _r(FactorVolumeVol, "volume_vol_60", "basic", {"window": 60})
    _r(FactorCorrPriceVolume, "corr_pv_60", "basic", {"window": 60})
    _r(FactorDrawdown, "drawdown_60", "basic", {"window": 60})

    # ── 注册东吴证券因子 ──
    from qrp.reports.dongwu.cpv_factor import (
        CPVFactor,
        CPVTimeSegmentFactor,
        RPVFactor,
        SRVFactor,
    )
    _r(CPVFactor, "cpv_240", "basic", {"window": 240})
    _r(CPVFactor, "cpv_60", "basic", {"window": 60})
    _r(CPVTimeSegmentFactor, "cpv_std_8", "basic", {"use_last_only": False})
    _r(CPVTimeSegmentFactor, "cpv_std_1430", "basic", {"use_last_only": True})
    register(RPVFactor, "rpv", "basic")
    register(SRVFactor, "srv", "basic")

    from qrp.reports.dongwu.technical_factors import (
        CandlePowerFactor,
        IdiosyncraticVolatilityFactor,
        SerialCorrelationFactor,
        ShadowLineFactor,
    )
    _r(ShadowLineFactor, "shadow_line_20", "basic", {"window": 20})
    _r(CandlePowerFactor, "candle_power_14", "basic", {"window": 14})
    _r(IdiosyncraticVolatilityFactor, "iv_vol_60", "basic", {"window": 60})
    _r(SerialCorrelationFactor, "serial_corr_5_60", "basic", {"lag": 5, "window": 60})

    # ── 注册华泰证券因子 ──
    from qrp.reports.huatai.ffscore import FFScoreFactor
    register(FFScoreFactor, "ffscore", "financial")

    from qrp.reports.huatai.gpt_factor_factory import (
        BullBearIndex,
    )
    register(BullBearIndex, "bull_bear", "basic")

    # ── 注册光大证券因子 ──
    from qrp.reports.guangda.rsrs_indicator import QRSIndicator, RSRSIndicator
    _r(RSRSIndicator, "rsrs_18", "basic", {"window": 18})
    _r(QRSIndicator, "qrs_18", "basic", {"window": 18})

    # ── 注册开源证券因子 ──
    from qrp.reports.kaiyuan.smart_money import (
        AmplitudeFactor,
        APMFloorFactor,
        MomentumFactor,
        SmartMoneyFactor,
    )
    _r(SmartMoneyFactor, "smart_money_20", "basic", {"window": 20})
    _r(APMFloorFactor, "apm_20", "basic", {"window": 20})
    _r(AmplitudeFactor, "amplitude_20", "basic", {"window": 20})
    _r(MomentumFactor, "momentum_cn_60", "basic", {"window": 60, "skip_days": 21})

    # ── 注册 Alpha101 因子 ──
    from qrp.factors.alpha101 import Alpha101
    _r(Alpha101, "alpha101_full", "basic", {})

    # ── 注册中金高频因子 ──
    from qrp.reports.others.zhongjin_factors import (
        AmihudIlliquidity,
        CorrPriceVolume,
        IntradayMomentum,
        RangeVolatility,
        ShortTermReversal,
    )
    _r(ShortTermReversal, "cicc_sr_5", "basic", {"window": 5})
    _r(IntradayMomentum, "cicc_idm_20", "basic", {"window": 20})
    _r(RangeVolatility, "cicc_range_vol_20", "basic", {"window": 20})
    _r(AmihudIlliquidity, "cicc_amihud_20", "basic", {"window": 20})
    _r(CorrPriceVolume, "cicc_corr_pv_20", "basic", {"window": 20})

    # ── 注册CPV期货版 ──
    from qrp.reports.dongwu.cpv_futures import CPVFuturesV1, CPVFuturesV3, MarketMicrostructureFactor
    _r(CPVFuturesV1, "cpv_futures_v1", "basic", {"window": 30})
    _r(CPVFuturesV3, "cpv_futures_v3", "basic", {"window": 30})
    _r(MarketMicrostructureFactor, "microstructure", "basic")

    # ── 注册东方/广发/国泰新因子 ──
    from qrp.reports.dongfang import BuySellPressureFactor, VolumePriceTrendFactor
    _r(BuySellPressureFactor, "dongfang_bs_pressure", "basic", {"window": 20})
    _r(VolumePriceTrendFactor, "dongfang_vpt", "basic", {"window": 20})

    # ── 执行所有测试 ──
    results = []
    total = len(TEST_FACTORS)
    for i, (cls, name, dt, kw) in enumerate(TEST_FACTORS):
        r = run_factor_test(cls, name, dt, kw)
        results.append(r)
        status = "✅" if r["passed"] else "❌"
        mean_val = r.get('mean', '?')
        mean_str = f"{mean_val:>10.4f}" if isinstance(mean_val, (int, float)) else "?"
        time_str = f"{r.get('timing_ms', 0):>8.2f}" if isinstance(r.get('timing_ms'), (int, float)) else "?"
        print(f"  [{i+1:2d}/{total}] {status} {name:25s} {mean_str}  {time_str}ms")
        if r.get("error"):
            print(f"        └─ Error: {r['error']}")

    return results
