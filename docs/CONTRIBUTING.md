# 贡献指南

感谢你考虑为 QuantResearch Playbook 贡献代码！

## 如何添加新的研报复现

1. 确定券商和研报信息
2. 在 `qrp/reports/` 下创建对应的券商目录（如 `dongwu/`）
3. 创建一个 Python 文件，包含因子/策略实现
4. 确保实现继承 `Factor` 或使用框架工具

### 模板

```python
"""券商名称 - 因子名称

参考研报：
  《券商-研报标题》(日期)

核心逻辑：
  因子的经济学逻辑描述
"""
from qrp.core.factor import Factor

class YourFactor(Factor):
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"your_factor_{window}",
            description="描述"
        )
        self._window = window

    def calculate(self, data) -> pl.Series:
        # 实现因子计算逻辑
        return result
```

## 代码规范

- 使用 `ruff` 进行代码检查
- 使用 `mypy` 进行类型检查
- 使用 `pytest` 编写测试
- 确保所有因子可独立运行 (`if __name__ == "__main__"`)
- 添加研报参考文献链接

## 提交 PR

1. Fork 本仓库
2. 创建新分支
3. 提交改动
4. 创建 Pull Request
