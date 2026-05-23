# 使用 Python 3.10+ 运行环境
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/Electricitysheep/QuantResearch-Playbook"
LABEL org.opencontainers.image.description="QuantResearch-Playbook - 券商金工研报复现框架"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY pyproject.toml README.md ./
COPY qrp/ qrp/
COPY tests/ tests/
COPY self_test/ self_test/
COPY examples/ examples/

# 安装 Python 依赖
RUN pip install --no-cache-dir -e ".[full]"

# 运行自测
RUN python -m self_test.run_all --no-report

# 默认命令
CMD ["python", "-m", "self_test.run_all"]
