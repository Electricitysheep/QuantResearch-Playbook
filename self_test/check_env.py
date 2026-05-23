"""环境检查 - 自动安装依赖并验证运行环境"""

import importlib
import platform
import subprocess
import sys
from typing import Optional


def check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    return ok, f"{v.major}.{v.minor}.{v.micro}"


def check_platform() -> tuple[bool, str]:
    p = platform.system()
    return True, f"{p} ({platform.release()})"


def check_dependency(name: str, min_version: Optional[str] = None) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        if min_version:
            from packaging.version import Version
            ok = Version(str(ver)) >= Version(min_version)
            return ok, str(ver)
        return True, str(ver)
    except ImportError:
        return False, "not installed"


REQUIRED_DEPS = {
    "polars": "1.0.0",
    "numpy": "1.24.0",
    "scipy": "1.10.0",
    "pandas": "2.0.0",
    "matplotlib": "3.7.0",
}

OPTIONAL_DEPS = {
    "seaborn": None,
    "akshare": None,
}


def auto_install(quiet: bool = False) -> dict[str, str]:
    """自动安装缺失依赖"""
    results = {}
    missing = []

    for name, min_ver in REQUIRED_DEPS.items():
        ok, ver = check_dependency(name, min_ver)
        if not ok:
            missing.append(name)
            results[name] = "missing (will install)"
        else:
            results[name] = f"{ver} (ok)"

    for name, _ in OPTIONAL_DEPS.items():
        ok, ver = check_dependency(name)
        if ok:
            results[name] = f"{ver} (ok)"
        else:
            results[name] = "not installed (optional)"

    if missing:
        if not quiet:
            print(f"Installing missing dependencies: {', '.join(missing)}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing, "-q"],
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.DEVNULL if quiet else None,
        )
        for name in missing:
            ok, ver = check_dependency(name)
            results[name] = f"{ver} (just installed)" if ok else f"{ver} (FAILED)"

    return results


def check_package_installed() -> tuple[bool, str]:
    """检查 qrp 包是否安装"""
    try:
        import qrp
        return True, qrp.__version__
    except ImportError:
        return False, "not installed"


def run_all_checks(quiet: bool = False) -> dict:
    """运行全部环境检查"""
    results = {
        "python": check_python_version(),
        "platform": check_platform(),
        "dependencies": auto_install(quiet=quiet),
        "qrp_package": check_package_installed(),
    }
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)
    results = run_all_checks(quiet=False)
    py_ok, py_ver = results["python"]
    print(f"Python:    {'✅' if py_ok else '❌'} {py_ver}")
    plat_ok, plat_ver = results["platform"]
    print(f"Platform:  {'✅' if plat_ok else '❌'} {plat_ver}")
    for name, status in results["dependencies"].items():
        mark = "✅" if "(ok)" in status or "(just installed)" in status else "⬜"
        print(f"  {name:12s} {mark} {status}")
    pkg_ok, pkg_ver = results["qrp_package"]
    print(f"QRP Pkg:   {'✅' if pkg_ok else '❌'} {pkg_ver}")
