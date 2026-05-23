"""HTML 测试报告生成器"""

import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_html_report(
    env_results: dict,
    factor_results: list[dict[str, Any]],
    analysis_results: list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    """生成完整的 HTML 自测报告"""

    env = env_results
    deps = env.get("dependencies", {})
    py_ok, py_ver = env.get("python", (False, "?"))
    pkg_ok, pkg_ver = env.get("qrp_package", (False, "?"))

    total_factors = len(factor_results)
    passed_factors = sum(1 for r in factor_results if r["passed"])
    factor_pct = round(passed_factors / max(total_factors, 1) * 100, 1)

    total_analysis = len(analysis_results)
    passed_analysis = sum(1 for r in analysis_results if r["passed"])
    analysis_pct = round(passed_analysis / max(total_analysis, 1) * 100, 1)

    overall_passed = passed_factors + passed_analysis
    overall_total = total_factors + total_analysis
    overall_pct = round(overall_passed / max(overall_total, 1) * 100, 1)

    # 构建 factor 表格行
    factor_rows = ""
    for r in factor_results:
        status = "✅" if r["passed"] else "❌"
        fac_name = r.get("name", "?")
        mean_val = r.get("mean", "N/A")
        std_val = r.get("std", "N/A")
        timing = r.get("timing_ms", "N/A")
        error = r.get("error", "")
        err_display = f"<br><span style='color:red;font-size:12px'>✗ {error}</span>" if error else ""
        cls_name = r.get("class", "")
        factor_rows += f"""
        <tr>
            <td>{status}</td>
            <td><code>{fac_name}</code></td>
            <td style="font-size:12px;color:#666">{cls_name}</td>
            <td style="text-align:right">{mean_val}</td>
            <td style="text-align:right">{std_val}</td>
            <td style="text-align:right">{timing}ms</td>
            <td>{err_display}</td>
        </tr>"""

    # 构建 analysis 表格行
    analysis_rows = ""
    for r in analysis_results:
        status = "✅" if r["passed"] else "❌"
        name = r["name"]
        details = r.get("details", {})
        timing = details.get("timing_ms", "")
        error = r.get("error", "")
        err_display = f"<br><span style='color:red;font-size:12px'>✗ {error}</span>" if error else ""

        # 详细指标
        extra_info = ""
        for k, v in details.items():
            if k != "timing_ms":
                extra_info += f"<span style='margin-left:8px;font-size:11px;color:#666'>{k}={v}</span>"

        analysis_rows += f"""
        <tr>
            <td>{status}</td>
            <td><code>{name}</code></td>
            <td style="text-align:right">{timing}ms</td>
            <td>{extra_info}{err_display}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QuantResearch-Playbook 自测报告</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f7fa; color:#1a1a2e; padding:20px; }}
.container {{ max-width:1200px; margin:0 auto; }}
h1 {{ font-size:24px; margin-bottom:5px; }}
h2 {{ font-size:18px; margin:25px 0 10px; padding-bottom:5px; border-bottom:2px solid #e0e0e0; }}
.subtitle {{ color:#666; font-size:14px; margin-bottom:20px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:20px; }}
.card {{ background:#fff; border-radius:8px; padding:15px; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
.card .num {{ font-size:28px; font-weight:700; }}
.card .label {{ font-size:12px; color:#666; margin-top:4px; }}
.card.green .num {{ color:#22c55e; }}
.card.orange .num {{ color:#f59e0b; }}
.card.blue .num {{ color:#3b82f6; }}
.card.red .num {{ color:#ef4444; }}
table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,.08); margin-bottom:20px; }}
th {{ background:#1a1a2e; color:#fff; padding:10px 12px; font-size:13px; text-align:left; }}
td {{ padding:8px 12px; font-size:13px; border-bottom:1px solid #f0f0f0; }}
tr:hover {{ background:#f8fafc; }}
code {{ background:#f1f5f9; padding:2px 6px; border-radius:3px; font-size:12px; }}
.env-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin-bottom:20px; }}
.env-item {{ background:#fff; border-radius:6px; padding:10px; box-shadow:0 1px 3px rgba(0,0,0,.06); }}
.env-item .key {{ font-size:11px; color:#666; }}
.env-item .val {{ font-size:13px; font-weight:500; }}
.footer {{ text-align:center; color:#999; font-size:12px; margin-top:30px; }}
</style>
</head>
<body>
<div class="container">
<h1>📊 QuantResearch-Playbook 自测报告</h1>
<p class="subtitle">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 一键验证框架完整性</p>

<div class="cards">
    <div class="card green">
        <div class="num">{overall_passed}/{overall_total}</div>
        <div class="label">总测试通过 ({overall_pct}%)</div>
    </div>
    <div class="card blue">
        <div class="num">{passed_factors}/{total_factors}</div>
        <div class="label">因子测试通过 ({factor_pct}%)</div>
    </div>
    <div class="card orange">
        <div class="num">{passed_analysis}/{total_analysis}</div>
        <div class="label">分析模块通过 ({analysis_pct}%)</div>
    </div>
    <div class="card" style="background:#1a1a2e;color:#fff">
        <div class="num">{len(factor_results)}</div>
        <div class="label">已实现因子/策略</div>
    </div>
</div>

<h2>🖥️ 运行环境</h2>
<div class="env-grid">
    <div class="env-item">
        <div class="key">Python</div>
        <div class="val">{'✅' if py_ok else '❌'} {py_ver}</div>
    </div>
    <div class="env-item">
        <div class="key">QRP 包</div>
        <div class="val">{'✅' if pkg_ok else '❌'} {pkg_ver}</div>
    </div>
"""
    for name, status in deps.items():
        mark = "✅" if "(ok)" in status or "(just installed)" in status else "⬜"
        html += f"""
    <div class="env-item">
        <div class="key">{name}</div>
        <div class="val">{mark} {status}</div>
    </div>"""

    html += f"""
</div>

<h2>🔬 因子测试 ({passed_factors}/{total_factors} 通过)</h2>
<table>
<thead><tr><th>状态</th><th>因子名称</th><th>类</th><th style="text-align:right">均值</th><th style="text-align:right">标准差</th><th style="text-align:right">耗时</th><th>备注</th></tr></thead>
<tbody>{factor_rows}</tbody>
</table>

<h2>⚙️ 分析模块测试 ({passed_analysis}/{total_analysis} 通过)</h2>
<table>
<thead><tr><th>状态</th><th>模块</th><th style="text-align:right">耗时</th><th>详细信息</th></tr></thead>
<tbody>{analysis_rows}</tbody>
</table>

<div class="footer">
    QuantResearch-Playbook v0.1.0 | 运行方式: <code>python -m self_test.run_all</code>
</div>
</div>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)
