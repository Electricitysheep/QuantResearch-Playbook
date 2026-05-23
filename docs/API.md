# API 参考文档

## 核心模块 (`qrp.core`)

### `qrp.core.data` - 数据加载层

```python
class DataLoader(source: str = "akshare")
```

统一的数据访问入口。

**参数:**
- `source`: 数据源名称。可选: `"akshare"` (免费A股数据), `"mock"` (模拟数据)

**方法:**

```python
load_daily(symbol: str, start: str = "20200101", end: str | None = None) -> pl.DataFrame
```
加载日线数据。返回包含 `date`, `open`, `close`, `high`, `low`, `volume`, `amount` 等列的 DataFrame。

```python
load_minute(symbol: str, date: str) -> pl.DataFrame
```
加载分钟级数据。

```python
load_multi_daily(symbols: list[str], start, end) -> pl.DataFrame
```
批量加载多只股票数据。

```python
stock_list() -> pl.DataFrame
```
获取全部A股列表。

---

### `qrp.core.factor` - 因子基类

```python
class Factor(name: str, description: str = "")
```

所有因子必须继承的抽象基类。

**方法:**

```python
calculate(data: pl.DataFrame) -> pl.Series
```
计算因子值（必须由子类实现）。

**内置因子:**

| 类 | 说明 |
|---|------|
| `FactorVwapDev(window=60)` | VWAP 偏离度因子 |
| `FactorMomentum(window=60)` | 动量因子 |
| `FactorRSI(window=14)` | RSI 因子 |
| `FactorVolumeVol(window=60)` | 成交量波动率因子 |
| `FactorCorrPriceVolume(window=60)` | 价量相关性因子 |
| `FactorDrawdown(window=60)` | 回撤因子 |

```python
class FactorPipeline(factors: list[Factor] | None = None)
```
因子流水线 - 批量计算多个因子。

**方法:**
- `add(factor)` - 添加因子
- `remove(name)` - 移除因子
- `compute(data)` - 计算所有因子，返回 dict
- `compute_to_df(data)` - 计算并合并为 DataFrame

---

### `qrp.core.analysis` - 因子分析

```python
class FactorAnalyzer(data: pl.DataFrame, factor_values: pl.Series, price_col: str = "close")
```

**方法:**

```python
compute_ic(forward_periods: int = 5) -> ICMetrics
```
计算 IC 指标。

```python
quantile_returns(forward_periods: int = 5, n_quantiles: int = 5) -> dict[int, float]
```
计算分层收益。

```python
long_short_return(forward_periods: int = 5) -> float
```
计算多空收益。

```python
full_report(forward_periods: int = 5) -> FactorReport
```
生成完整评估报告。

**`ICMetrics` 属性:**
- `ic_mean`: IC 均值
- `ic_std`: IC 标准差
- `icir`: ICIR
- `ic_positive_ratio`: IC 正值比例
- `rank_ic_mean`: RankIC 均值
- `rank_icir`: RankICIR

---

### `qrp.core.backtest` - 回测引擎

```python
class Backtester(transaction_cost: float = 0.001, slippage: float = 0.001)
```

**方法:**

```python
run(prices: pl.Series, signals: pl.Series, periods_per_year: int = 252) -> BacktestResult
```

**`BacktestResult` 属性:**
- `total_return`, `annual_return`, `annual_volatility`
- `sharpe_ratio`, `max_drawdown`
- `win_rate`, `profit_factor`, `num_trades`
- `equity_curve`

---

### `qrp.core.utils` - 工具函数

```python
cross_section_rank(data, factor_col, group_col=None) -> pl.Series
standardize(series, method="zscore") -> pl.Series
winsorize(series, limits=0.05) -> pl.Series
compute_turnover(signals) -> float
```

---

## 券商策略模块 (`qrp.reports.*`)

### 东吴证券 (`qrp.reports.dongwu`)

**CPV 因子系列** (`cpv_factor.py`):

| 类 | 说明 |
|---|------|
| `CPVFactor(window)` | 基础 CPV 因子 |
| `CPVTimeSegmentFactor(use_last_only)` | 分时版 CPV |
| `RPVFactor()` | 新价量相关性因子 |
| `SRVFactor()` | 聪明版价量因子 |

函数:
- `run_cpv_analysis(data, window)` - 完整 CPV 分析
- `run_cpv_advanced(data)` - CPV 全系列对比

**技术因子系列** (`technical_factors.py`):

| 类 | 说明 |
|---|------|
| `ShadowLineFactor(window)` | 上下影线因子 |
| `CandlePowerFactor(window)` | 蜡烛力量因子 |
| `IdiosyncraticVolatilityFactor(window)` | 特质波动率 |
| `SerialCorrelationFactor(lag, window)` | 价量自相关性 |

### 华泰证券 (`qrp.reports.huatai`)

**FFScore** (`ffscore.py`):
- `FFScoreFactor()` - FFScore 基本面评分
- `CSCCVFramework(n_splits, n_comparisons)` - CSCV 过拟合概率

**GPT 因子工厂** (`gpt_factor_factory.py`):
- `GPTFactorFactory.generate_all(data)` - 4个 GPT 挖掘因子
- `BullBearIndex(vol_window, turnover_window)` - 牛熊指标

### 光大证券 (`qrp.reports.guangda`)

**RSRS/QRS** (`rsrs_indicator.py`):
- `RSRSIndicator(window, ret_threshold)` - RSRS 择时指标
- `QRSIndicator(window)` - QRS 改进指标

### 开源证券 (`qrp.reports.kaiyuan`)

**聪明钱因子系列** (`smart_money.py`):
- `SmartMoneyFactor(window)` - 聪明钱因子
- `APMFloorFactor(window)` - APM 非流动性因子
- `AmplitudeFactor(window)` - 振幅因子
- `MomentumFactor(window, skip_days)` - A 股动量因子
