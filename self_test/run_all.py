#!/usr/bin/env python3
"""QuantResearch-Playbook 一键自测入口

用法:
    python -m self_test.run_all              # 完整自测
    python -m self_test.run_all --quick       # 快速模式（少量数据）
    python -m self_test.run_all --no-report   # 不生成 HTML 报告
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def print_banner():
    print()
    print("=" * 60)
    print("  QuantResearch-Playbook 一键自测")
    print("  券商金工研报复现框架 · 完整验证")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="QuantResearch-Playbook Self-Test")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    parser.add_argument("--no-report", action="store_true", help="不生成 HTML 报告")
    parser.add_argument("--open", action="store_true", help="自动打开报告")
    args = parser.parse_args()

    print_banner()

    # ── 第1步：环境检查 ──
    print("[1/4] 检查运行环境...")
    from self_test.check_env import run_all_checks

    env_results = run_all_checks(quiet=args.quick)

    py_ok, py_ver = env_results["python"]
    pkg_ok, pkg_ver = env_results["qrp_package"]

    if not py_ok:
        print("  ❌ Python 版本过低，需要 3.10+")
        sys.exit(1)
    if not pkg_ok:
        print("  ⚠️  qrp 包未安装，尝试安装...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", str(ROOT), "-q"]
        )
        from self_test.check_env import check_package_installed
        pkg_ok, pkg_ver = check_package_installed()
        if not pkg_ok:
            print("  ❌ qrp 包安装失败")
            sys.exit(1)

    if not env_results.get("required_ok", True):
        print("  ❌ 核心依赖缺失，请检查上方的安装结果")
        sys.exit(1)
    optional_missing = env_results.get("optional_missing", [])
    if optional_missing:
        print(f"  💡 可选依赖未安装: {', '.join(optional_missing)} (不影响核心功能)")
    print()

    # ── 第2步：因子测试 ──
    print("[2/4] 因子实现测试...")
    from self_test.test_factors import run_all_factor_tests

    factor_results = run_all_factor_tests()

    factor_pass = sum(1 for r in factor_results if r["passed"])
    factor_total = len(factor_results)
    print(f"  → 因子测试: {factor_pass}/{factor_total} 通过")
    print()

    # ── 第3步：分析模块测试 ──
    print("[3/4] 分析与回测模块测试...")
    from self_test.test_analysis import run_all_analysis_tests

    analysis_results = run_all_analysis_tests()

    analysis_pass = sum(1 for r in analysis_results if r["passed"])
    analysis_total = len(analysis_results)
    print(f"  → 分析测试: {analysis_pass}/{analysis_total} 通过")
    print()

    # ── 第4步：报告生成 ──
    print("[4/4] 生成 HTML 测试报告...")
    report_path = ROOT / "self_test" / "reports" / "test_report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_report:
        from self_test.report_generator import generate_html_report

        result_path = generate_html_report(
            env_results,
            factor_results,
            analysis_results,
            report_path,
        )
        print(f"  ✅ 报告已保存: {result_path}")
        if args.open:
            import webbrowser
            webbrowser.open(f"file://{result_path}")
    else:
        print("  ⏭️  跳过 HTML 报告生成")

    # ── 最终汇总 ──
    print()
    print("=" * 60)
    total_pass = factor_pass + analysis_pass
    total_all = factor_total + analysis_total
    pct = round(total_pass / max(total_all, 1) * 100, 1)
    status = "✅ 全部通过" if total_pass == total_all else f"⚠️  {total_all - total_pass} 项未通过"
    print(f"  最终结果: {total_pass}/{total_all} ({pct}%) {status}")
    if total_pass == total_all:
        print("  🎉 所有测试通过！框架运行正常。")
    else:
        print("  ❌ 部分测试失败，请查看上方日志排查。")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
