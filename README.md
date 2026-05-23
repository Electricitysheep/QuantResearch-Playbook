# QuantResearch Playbook 📈

> **券商金工研报复现框架** — 更全面 · 更先进 · 更轻量

基于 Polars 构建的新一代券商金工研报复现框架。覆盖 **13+ 家券商、50+ 个经典策略**，从数据获取到因子评估到回测验证全流程一体化。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **🚀 高性能** | 基于 Polars 向量化计算，比 Pandas 快 5-10x |
| **📦 轻量级** | 核心依赖仅 Polars + NumPy + SciPy，无沉重 ML 栈 |
| **🔌 可扩展** | 插件式因子/策略架构，新增策略无需修改框架 |
| **📊 全面分析** | 内置 IC/ICIR 分析、分层回测、多空收益、因子归因 |
| **🆓 免费数据** | 首选 AkShare 免费数据源，兼容 Tushare |
| **📝 标准化** | 每个策略统一模板：研报信息 → 因子计算 → 回测 → 报告 |

## 📋 覆盖券商与策略

| 券商 | 策略 | 状态 |
|------|------|------|
| **东吴证券** | CPV 价量相关性因子、上下影线因子、特质波动率因子 | ✅ |
| **华泰证券** | FFScore 模型、GPT 因子工厂、CSCV 框架、牛熊指标 | ✅ |
| **光大证券** | RSRS 阻力支撑相对强度择时、QRS 择时 | ✅ |
| **开源证券** | 聪明钱因子模型、APM 因子、振幅因子、动量因子 | ✅ |
| **广发证券** | 低延迟趋势线、指数高阶矩择时、CGO 行为因子 | ✅ |
| **国泰君安** | CCK 羊群效应模型、分析师预期修正因子 | ✅ |
| **东方证券** | 量价关系买卖压力因子 | ✅ |
| **中金公司** | QRS 择时、高频因子手册 | ✅ |
| **国信证券** | 小波分析择时、波动率研究 | ✅ |
| **申万宏源** | 大师价值投资系列 | ✅ |
| **浙商证券** | 金股组合增强策略 | ✅ |
| **东北证券** | 扩散指标择时 | ✅ |
| **招商证券** | 多因子指数增强 | ✅ |
> 更详细清单见 [docs/REPORTS_CATALOG.md](docs/REPORTS_CATALOG.md)

## 🚀 快速开始

```bash
# 安装
pip install polars numpy scipy pandas matplotlib seaborn

# 可选：安装全部依赖
pip install "qrp[full]"

# 运行一个策略
python -m qrp.reports.dongwu.cpv_factor --help
```

### 示例：计算 CPV 因子

```python
from qrp.data import DataLoader
from qrp.reports.dongwu.cpv_factor import CPVFactor

# 加载数据
loader = DataLoader(source="akshare")
data = loader.load_minute_data("000001", "2024-01-01", "2024-12-31")

# 计算 CPV 因子
cpv = CPVFactor(window=60)
factor_values = cpv.calculate(data)

# 因子评估
from qrp.analysis import FactorAnalyzer
analyzer = FactorAnalyzer(data, factor_values)
report = analyzer.full_report()
print(report.summary())
```

## 🏗️ 项目结构

```
QuantResearch-Playbook/
├── pyproject.toml            # 项目配置
├── README.md                 # 本文件
├── docs/                     # 文档
│   ├── REPORTS_CATALOG.md    # 策略完整目录
│   └── CONTRIBUTING.md       # 贡献指南
├── qrp/                      # 核心包
│   ├── __init__.py
│   ├── core/                 # 基础框架
│   │   ├── data.py           # 数据加载层
│   │   ├── factor.py         # 因子基类
│   │   ├── strategy.py       # 策略基类
│   │   ├── backtest.py       # 轻量回测引擎
│   │   └── analysis.py       # IC 分析与评估
│   ├── factors/              # 通用因子库
│   └── reports/              # 券商策略实现
│       ├── dongwu/           # 东吴证券
│       ├── huatai/           # 华泰证券
│       ├── guangda/          # 光大证券
│       └── ...
├── notebooks/                # 教程 Notebook
├── examples/                 # 使用示例
├── tests/                    # 测试
└── scripts/                  # 工具脚本
```

## 📖 学习路径

1. **入门**: 阅读 [docs/REPORTS_CATALOG.md](docs/REPORTS_CATALOG.md) 了解策略概览
2. **实践**: 运行 `notebooks/01_quickstart.ipynb` 快速上手
3. **深入**: 选择一个感兴趣的研报复现，阅读源码
4. **扩展**: 按照模板贡献你自己的研报复现

## 🤝 贡献

欢迎贡献新的研报复现！请阅读 [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) 了解贡献流程。

## 📄 License

MIT
