# 券商金工研报复现目录 (Reports Catalog)

## 东吴证券
- [x] CPV 价量相关性因子 (2020.02.23) → `qrp/reports/dongwu/cpv_factor.py`
  - CPVFactor: 分钟级价量滚动相关性
  - CPVTimeSegmentFactor: 分时版8时段标准差
  - RPVFactor: 新价量相关性因子 (价量四象限)
  - SRVFactor: 聪明版日频价量相关性

- [x] 上下影线因子 (2020.06.19) → `qrp/reports/dongwu/technical_factors.py`
  - ShadowLineFactor: 上下影线比率
  - CandlePowerFactor: 蜡烛力量因子

- [x] 特质波动率因子 (2020.05.28) → `qrp/reports/dongwu/technical_factors.py`
  - IdiosyncraticVolatilityFactor

- [x] CPV因子移位版 (2021.03.01) → `qrp/reports/dongwu/technical_factors.py`
  - SerialCorrelationFactor

## 华泰证券
- [x] FFScore 模型 (2017.02.09) → `qrp/reports/huatai/ffscore.py`
- [x] CSCV 回测过拟合概率 (2019.06.17) → `qrp/reports/huatai/ffscore.py`
- [x] GPT因子工厂2.0 (2024.09.26) → `qrp/reports/huatai/gpt_factor_factory.py`
  - GPTFactorFactory: 4个GPT挖掘高频因子
- [x] 牛熊指标 (2019.09.27) → `qrp/reports/huatai/gpt_factor_factory.py`

## 光大证券
- [x] RSRS 阻力支撑相对强度 (2017.05.01) → `qrp/reports/guangda/rsrs_indicator.py`
- [x] RSRS改进版 (2019.11.17) → `qrp/reports/guangda/rsrs_indicator.py`

## 中金公司
- [x] QRS 择时指标 (2021.01.21) → `qrp/reports/guangda/rsrs_indicator.py`

## 开源证券
- [x] 聪明钱因子模型 (2020.02.09) → `qrp/reports/kaiyuan/smart_money.py`
- [x] A股反转微观来源 (2019.12.23) → `qrp/reports/kaiyuan/smart_money.py`
- [x] APM因子进阶版 (2020.03.07) → `qrp/reports/kaiyuan/smart_money.py`
- [x] 振幅因子隐藏结构 (2020.05.16) → `qrp/reports/kaiyuan/smart_money.py`
- [x] A股动量因子 (2020.07.21) → `qrp/reports/kaiyuan/smart_money.py`

## 待实现 (Planned)
- [ ] 东方证券 - 量价买卖压力因子 (2019.10.29)
- [ ] 广发证券 - 低延迟趋势线 (2017.03.03)
- [ ] 广发证券 - 指数高阶矩择时 (2015.05.20)
- [ ] 广发证券 - CGO行为因子 (2017.07.07)
- [ ] 国泰君安 - CCK羊群效应 (2018.11.28)
- [ ] 国泰君安 - 分析师预期修正因子
- [ ] 国信证券 - 小波分析择时 (2010.06.21)
- [ ] 国信证券 - 波动率单向差值 (2015.10.22)
- [ ] 申万宏源 - 大师价值投资系列
- [ ] 浙商证券 - 金股组合增强 (2022.08.22)
- [ ] 招商证券 - 多因子指数增强
- [ ] 东北证券 - 扩散指标择时 (2019.09.24)
- [ ] 中金公司 - 高频因子手册 (79个因子)
- [ ] 华泰证券 - 牛熊指标完善版

## 统计
- **已实现**: 15+ 个因子/策略
- **覆盖券商**: 6 家 (东吴/华泰/光大/中金/开源)
- **待实现**: 15+ 个策略 (欢迎PR!)
