"""因子基类与常用因子定义"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class Factor(ABC):
    """因子抽象基类

    所有因子必须继承此类并实现 calculate 方法。
    """

    def __init__(self, name: str, description: str = ""):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @abstractmethod
    def calculate(self, data: pl.DataFrame) -> pl.Series:
        """计算因子值

        Args:
            data: 输入数据（日线/分钟线）

        Returns:
            因子值序列
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class FactorPipeline:
    """因子流水线 - 批量计算多个因子"""

    def __init__(self, factors: list[Factor] | None = None):
        self._factors: dict[str, Factor] = {}
        if factors:
            for f in factors:
                self.add(f)

    def add(self, factor: Factor) -> FactorPipeline:
        self._factors[factor.name] = factor
        return self

    def remove(self, name: str) -> FactorPipeline:
        self._factors.pop(name, None)
        return self

    def compute(self, data: pl.DataFrame) -> dict[str, pl.Series]:
        """计算所有因子"""
        return {name: fac.calculate(data) for name, fac in self._factors.items()}

    def compute_to_df(self, data: pl.DataFrame) -> pl.DataFrame:
        """计算所有因子并合并为 DataFrame"""
        results = {}
        for name, fac in self._factors.items():
            results[name] = fac.calculate(data)
        return pl.DataFrame(results)

    @property
    def factor_names(self) -> list[str]:
        return list(self._factors)

    def __len__(self) -> int:
        return len(self._factors)

    def __getitem__(self, name: str) -> Factor:
        return self._factors[name]


# ---------- 基础因子实现 ----------


class FactorVwapDev(Factor):
    """VWAP 偏离度因子

    VWAP 偏离度 = (close - VWAP) / VWAP
    逻辑：大单交易导致价格偏离 VWAP，反映机构行为
    """

    def __init__(self, window: int = 60):
        super().__init__(
            name=f"vwap_dev_{window}",
            description=f"VWAP 偏离度因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        vwap = (data["amount"] / data["volume"]).rolling_mean(
            window_size=self._window
        )
        return (data["close"] - vwap) / vwap


class FactorMomentum(Factor):
    """动量因子"""

    def __init__(self, window: int = 60):
        super().__init__(
            name=f"momentum_{window}",
            description=f"动量因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        return data["close"].pct_change(self._window)


class FactorRSI(Factor):
    """RSI 因子"""

    def __init__(self, window: int = 14):
        super().__init__(
            name=f"rsi_{window}",
            description=f"RSI 因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        delta = data["close"].diff()
        gain = delta.clip(0, None).rolling_mean(window_size=self._window)
        loss = (-delta.clip(None, 0)).rolling_mean(window_size=self._window)
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))


class FactorVolumeVol(Factor):
    """成交量波动率因子"""

    def __init__(self, window: int = 60):
        super().__init__(
            name=f"volume_vol_{window}",
            description=f"成交量波动率因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        vol_std = data["volume"].rolling_std(window_size=self._window)
        vol_mean = data["volume"].rolling_mean(window_size=self._window)
        return vol_std / vol_mean


class FactorCorrPriceVolume(Factor):
    """价量相关性因子（CPV 理论）"""

    def __init__(self, window: int = 60):
        super().__init__(
            name=f"corr_pv_{window}",
            description=f"价量相关性因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        import polars as pl
        return data.with_columns(
            pl.rolling_corr("close", "volume", window_size=self._window).alias("__corr")
        )["__corr"]


class FactorDrawdown(Factor):
    """回撤因子"""

    def __init__(self, window: int = 60):
        super().__init__(
            name=f"drawdown_{window}",
            description=f"回撤因子（窗口={window}）",
        )
        self._window = window

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        rolling_max = data["high"].rolling_max(window_size=self._window)
        return (data["low"] - rolling_max) / rolling_max


# 基础因子工厂：快速创建 6 大基础因子
def create_base_factors() -> FactorPipeline:
    """创建 6 大基础因子"""
    return FactorPipeline(
        [
            FactorVwapDev(),
            FactorMomentum(),
            FactorRSI(),
            FactorDrawdown(),
            FactorVolumeVol(),
            FactorCorrPriceVolume(),
        ]
    )


def expand_factors() -> FactorPipeline:
    """通过参数变化创建扩展因子集"""
    pipe = FactorPipeline()
    for w in [15, 30, 60, 120, 240]:
        pipe.add(FactorMomentum(w))
    for w in [7, 14, 21, 28]:
        pipe.add(FactorRSI(w))
    for w in [15, 30, 60, 120]:
        pipe.add(FactorVolumeVol(w))
        pipe.add(FactorCorrPriceVolume(w))
    pipe.add(FactorVwapDev())
    pipe.add(FactorDrawdown())
    return pipe
