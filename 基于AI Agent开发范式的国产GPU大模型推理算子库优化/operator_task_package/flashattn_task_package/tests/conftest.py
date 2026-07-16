"""
pytest 全局配置。

1. 设置 MACA_PATH 等环境变量（非交互 shell 不加载 ~/.bashrc）
2. 注册自定义 mark
3. 默认跳过 slow 测试（用 --runslow 显式开启）
"""
import os
import sys
from pathlib import Path

# ── 环境变量（非交互 shell 兜底）──
os.environ.setdefault("MACA_PATH", "/opt/maca")
maca = os.environ["MACA_PATH"]
os.environ.setdefault("PATH", f"{maca}/mxgpu_llvm/bin:{os.environ.get('PATH', '')}")
ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("LD_LIBRARY_PATH", f"{maca}/lib:{maca}/mxgpu_llvm/lib:{ld}")

# ── 让 tests/ 能 import agent_system ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 慢测试，默认跳过，用 --runslow 开启")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="运行 slow 标记的测试")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow 测试，加 --runslow 开启")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


import pytest  # noqa: E402（需在 addoption 之后）
