"""数据加载层 - 统一数据接口"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """数据源抽象基类"""

    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def stock_list(self) -> pl.DataFrame:
        """获取全部A股列表"""
        ...

    @abstractmethod
    def daily_bars(self, symbol: str, start: str, end: str) -> pl.DataFrame:
        """获取日线数据"""
        ...

    @abstractmethod
    def minute_bars(self, symbol: str, date: str) -> pl.DataFrame:
        """获取分钟级数据"""
        ...


class AkShareSource(DataSource):
    """AkShare 数据源（免费）"""

    def get_name(self) -> str:
        return "akshare"

    def stock_list(self) -> pl.DataFrame:
        import akshare as ak

        df = ak.stock_info_a_code_name()
        return pl.from_pandas(df)

    def daily_bars(self, symbol: str, start: str, end: str) -> pl.DataFrame:
        import akshare as ak

        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        result = pl.from_pandas(df)
        return self._normalize_daily(result)

    def minute_bars(self, symbol: str, date: str) -> pl.DataFrame:
        import akshare as ak

        df = ak.stock_zh_a_minute(symbol=symbol, period="1", adjust="qfq")
        result = pl.from_pandas(df)
        return self._normalize_minute(result)

    @staticmethod
    def _normalize_daily(df: pl.DataFrame) -> pl.DataFrame:
        rename_map = {}
        for old, new in {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_chg",
            "涨跌额": "price_chg",
            "换手率": "turnover",
        }.items():
            if old in df.columns:
                rename_map[old] = new
        df = df.rename(rename_map)
        if "date" in df.columns:
            df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        return df

    @staticmethod
    def _normalize_minute(df: pl.DataFrame) -> pl.DataFrame:
        rename_map = {}
        for old, new in {
            "时间": "time",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }.items():
            if old in df.columns:
                rename_map[old] = new
        df = df.rename(rename_map)
        return df


class MockSource(DataSource):
    """模拟数据源 - 用于测试"""

    def get_name(self) -> str:
        return "mock"

    def stock_list(self) -> pl.DataFrame:
        return pl.DataFrame({"code": ["000001", "000002"], "name": ["平安银行", "万科A"]})

    def daily_bars(self, symbol: str, start: str, end: str) -> pl.DataFrame:
        import numpy as np
        from datetime import timedelta

        start_dt = datetime.strptime(start[:10], "%Y-%m-%d") if "-" in start else datetime.strptime(start[:8], "%Y%m%d")
        end_dt = datetime.strptime(end[:10], "%Y-%m-%d") if "-" in end else datetime.strptime(end[:8], "%Y%m%d")
        n = min((end_dt - start_dt).days, 60)
        if n < 1:
            n = 60
            start_dt = datetime.now() - timedelta(days=n)

        dates = [start_dt + timedelta(days=i) for i in range(n)]
        base = 10.0
        return pl.DataFrame(
            {
                "date": dates,
                "open": (base + np.random.randn(n).cumsum()).tolist(),
                "high": (base + np.random.randn(n).cumsum() + 0.5).tolist(),
                "low": (base + np.random.randn(n).cumsum() - 0.5).tolist(),
                "close": (base + np.random.randn(n).cumsum()).tolist(),
                "volume": np.random.randint(1_000_000, 100_000_000, n).tolist(),
                "amount": np.random.uniform(100_000_000, 10_000_000_000, n).tolist(),
            }
        )

    def minute_bars(self, symbol: str, date: str) -> pl.DataFrame:
        import numpy as np

        n = 240
        base = 10.0
        return pl.DataFrame(
            {
                "time": list(range(n)),
                "open": base + np.random.randn(n).cumsum() * 0.01,
                "high": base + np.random.randn(n).cumsum() * 0.01 + 0.02,
                "low": base + np.random.randn(n).cumsum() * 0.01 - 0.02,
                "close": base + np.random.randn(n).cumsum() * 0.01,
                "volume": np.random.randint(1e4, 1e6, n),
                "amount": np.random.uniform(1e6, 1e8, n),
            }
        )


class DataLoader:
    """数据加载器 - 统一的数据访问入口"""

    SOURCES: dict[str, type[DataSource]] = {
        "akshare": AkShareSource,
        "mock": MockSource,
    }

    def __init__(self, source: str = "akshare", **kwargs: Any):
        source_cls = self.SOURCES.get(source)
        if source_cls is None:
            msg = f"Unknown data source: {source}. Available: {list(self.SOURCES)}"
            raise ValueError(msg)
        self._source = source_cls(**kwargs)
        logger.info("DataLoader initialized with source: %s", source)

    @property
    def source(self) -> DataSource:
        return self._source

    def load_daily(
        self,
        symbol: str,
        start: str = "20200101",
        end: str | None = None,
    ) -> pl.DataFrame:
        """加载日线数据"""
        end = end or datetime.now().strftime("%Y%m%d")
        return self._source.daily_bars(symbol, start, end)

    def load_minute(self, symbol: str, date: str) -> pl.DataFrame:
        """加载分钟级数据"""
        return self._source.minute_bars(symbol, date)

    def load_multi_daily(
        self,
        symbols: list[str],
        start: str = "20200101",
        end: str | None = None,
    ) -> pl.DataFrame:
        """批量加载多只股票的日线数据"""
        frames = []
        for sym in symbols:
            df = self.load_daily(sym, start, end)
            df = df.with_columns(pl.lit(sym).alias("symbol"))
            frames.append(df)
        return pl.concat(frames) if frames else pl.DataFrame()

    def stock_list(self) -> pl.DataFrame:
        """获取全部A股列表"""
        return self._source.stock_list()

    @classmethod
    def register_source(cls, name: str, src_cls: type[DataSource]) -> None:
        """注册自定义数据源"""
        cls.SOURCES[name] = src_cls
