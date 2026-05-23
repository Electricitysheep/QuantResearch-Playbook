"""Tests for QuantResearch Playbook core modules."""

import polars as pl

from qrp.core.factor import (
    FactorVwapDev,
    FactorMomentum,
    FactorRSI,
    FactorCorrPriceVolume,
    FactorPipeline,
    create_base_factors,
)
from qrp.core.backtest import Backtester
from qrp.core.analysis import FactorAnalyzer


def _make_mock_data(n: int = 200) -> pl.DataFrame:
    import numpy as np

    np.random.seed(42)
    base = 10.0
    return pl.DataFrame(
        {
            "close": base + np.random.randn(n).cumsum() * 0.1,
            "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
            "amount": np.abs(np.random.randn(n) * 1e8 + 5e8),
            "high": base + np.random.randn(n).cumsum() * 0.1 + 0.2,
            "low": base + np.random.randn(n).cumsum() * 0.1 - 0.2,
        }
    )


class TestFactors:
    def test_vwap_dev(self):
        data = _make_mock_data()
        fac = FactorVwapDev(window=20)
        values = fac.calculate(data)
        assert len(values) == len(data)
        assert fac.name == "vwap_dev_20"

    def test_momentum(self):
        data = _make_mock_data()
        fac = FactorMomentum(window=10)
        values = fac.calculate(data)
        assert len(values) == len(data)

    def test_rsi(self):
        data = _make_mock_data()
        fac = FactorRSI(window=14)
        values = fac.calculate(data)
        assert len(values) == len(data)
        # RSI should be in [0, 100]
        val_valid = values.drop_nulls()
        if len(val_valid) > 0:
            assert val_valid.min() >= 0
            assert val_valid.max() <= 100

    def test_corr_price_volume(self):
        data = _make_mock_data()
        fac = FactorCorrPriceVolume(window=20)
        values = fac.calculate(data)
        assert len(values) == len(data)

    def test_pipeline(self):
        data = _make_mock_data()
        pipe = create_base_factors()
        assert len(pipe) == 6
        results = pipe.compute(data)
        assert len(results) == 6
        for name, val in results.items():
            assert len(val) == len(data)

    def test_expand_factors(self):
        from qrp.core.factor import expand_factors

        pipe = expand_factors()
        assert len(pipe) > 10


class TestBacktest:
    def test_backtest_basic(self):
        import numpy as np

        n = 200
        prices = pl.Series("price", 10 + np.random.randn(n).cumsum() * 0.1)
        signals = pl.Series("signal", np.random.choice([-1, 0, 1], n))

        bt = Backtester()
        result = bt.run(prices, signals)
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "total_return")
        assert result.num_trades >= 0

    def test_backtest_all_short(self):
        import numpy as np

        n = 100
        prices = pl.Series("price", 10 + np.random.randn(n).cumsum())
        signals = pl.Series("signal", np.full(n, -1))
        bt = Backtester()
        result = bt.run(prices, signals)
        assert result.num_trades == 0


class TestAnalysis:
    def test_ic_computation(self):
        import numpy as np

        data = _make_mock_data(300)
        values = pl.Series("factor", np.random.randn(300))
        analyzer = FactorAnalyzer(data, values)
        ic = analyzer.compute_ic(forward_periods=5)
        assert hasattr(ic, "ic_mean")
        assert hasattr(ic, "icir")

    def test_quantile_returns(self):
        import numpy as np

        data = _make_mock_data(300)
        values = pl.Series("factor", np.random.randn(300))
        analyzer = FactorAnalyzer(data, values)
        q_ret = analyzer.quantile_returns(n_quantiles=5)
        assert len(q_ret) == 5

    def test_long_short(self):
        import numpy as np

        data = _make_mock_data(300)
        values = pl.Series("factor", np.random.randn(300))
        analyzer = FactorAnalyzer(data, values)
        ls = analyzer.long_short_return()
        assert isinstance(ls, float)


class TestReports:
    def test_cpv_factor(self):
        from qrp.reports.dongwu.cpv_factor import CPVFactor

        data = _make_mock_data(500)
        fac = CPVFactor(window=60)
        values = fac.calculate(data)
        assert len(values) == len(data)

    def test_rsrs_indicator(self):
        from qrp.reports.guangda.rsrs_indicator import RSRSIndicator

        data = _make_mock_data(200)
        fac = RSRSIndicator(window=18)
        values = fac.calculate(data)
        assert len(values) == len(data)

    def test_ffscore(self):
        from qrp.reports.huatai.ffscore import FFScoreFactor

        data = _make_mock_data(200)
        fac = FFScoreFactor()
        values = fac.calculate(data)
        assert len(values) == len(data)

    def test_smart_money(self):
        from qrp.reports.kaiyuan.smart_money import SmartMoneyFactor

        data = _make_mock_data(200)
        fac = SmartMoneyFactor()
        values = fac.calculate(data)
        assert len(values) == len(data)
