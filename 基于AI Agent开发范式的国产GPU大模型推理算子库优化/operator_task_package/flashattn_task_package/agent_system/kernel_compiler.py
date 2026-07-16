"""
Kernel Compiler —— 用 mxcc 真实编译 CUDA Maca kernel 源码为 .so。

这是 Coder 角色的输出端：把生成的 .cu 源码编译成可加载的共享库。
捕获编译错误，供 Reflector 记录到 failure_cases。

实测确认的编译要点（来自本地 mxcc 调试）：
- MACA 用 __maca_bfloat16（内置不完整类型），不是 __nv_bfloat16
- 用 typedef mctlass::bfloat16_t __nv_bfloat16 兼容 OJ 签名
- mctlass::bfloat16_t 是完整代理类，支持下标和隐式 float 转换
- 编译命令：mxcc -std=c++17 -fPIC -shared -DMACA_ARCH=1000 -I$MACA_PATH/include
"""
from __future__ import annotations

import os
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# MACA 工具链路径（实测）
MACA_PATH = os.environ.get("MACA_PATH", "/opt/maca")
MXCC = os.path.join(MACA_PATH, "mxgpu_llvm", "bin", "mxcc")
MACA_INCLUDE = os.path.join(MACA_PATH, "include")

# C500 架构宏
MACA_ARCH = "1000"


@dataclass
class CompileResult:
    """编译结果。"""
    success: bool
    so_path: Optional[str]    # 成功时的 .so 路径
    error_msg: str = ""       # 失败时的错误信息
    stderr: str = ""          # 原始 stderr
    returncode: int = 0
    compile_time_s: float = 0.0


def _ensure_mxcc() -> bool:
    """检查 mxcc 是否可用。"""
    return os.path.isfile(MXCC) and os.access(MXCC, os.X_OK)


def find_mxcc() -> Optional[str]:
    """查找 mxcc 路径（兼容 PATH 中的 mxcc）。"""
    if _ensure_mxcc():
        return MXCC
    p = shutil.which("mxcc")
    return p


def compile_source(
    source_code: str,
    output_so_path: str,
    extra_flags: Optional[list] = None,
    timeout: int = 120,
) -> CompileResult:
    """
    编译 CUDA Maca 源码为 .so 共享库。

    参数：
      source_code: .cu 源码字符串
      output_so_path: 输出 .so 路径
      extra_flags: 额外编译选项

    返回 CompileResult。
    """
    import time
    mxcc = find_mxcc()
    if not mxcc:
        return CompileResult(
            success=False, so_path=None,
            error_msg=f"mxcc 未找到（MACA_PATH={MACA_PATH}）",
        )

    # 写源码到临时文件
    out_path = Path(output_so_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_path = out_path.with_suffix(".cu")
    src_path.write_text(source_code)

    # 构造编译命令
    cmd = [
        mxcc,
        "-std=c++17", "-fPIC", "-shared",
        f"-DMACA_ARCH={MACA_ARCH}",
        f"-I{MACA_INCLUDE}",
        str(src_path),
        "-o", str(out_path),
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    # 执行编译
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "MACA_PATH": MACA_PATH},
        )
        elapsed = time.time() - t0
        if proc.returncode == 0 and out_path.exists():
            return CompileResult(
                success=True, so_path=str(out_path),
                returncode=0, compile_time_s=elapsed,
            )
        else:
            return CompileResult(
                success=False, so_path=None,
                error_msg=proc.stderr[:2000] if proc.stderr else "未知编译错误",
                stderr=proc.stderr, returncode=proc.returncode,
                compile_time_s=elapsed,
            )
    except subprocess.TimeoutExpired:
        return CompileResult(
            success=False, so_path=None,
            error_msg=f"编译超时（{timeout}s）",
            compile_time_s=timeout,
        )
    except Exception as e:
        return CompileResult(
            success=False, so_path=None,
            error_msg=f"编译异常: {e}",
        )


def compile_file(
    src_path: str,
    output_so_path: str,
    extra_flags: Optional[list] = None,
    timeout: int = 120,
) -> CompileResult:
    """编译已有的 .cu 文件。"""
    src = Path(src_path)
    if not src.exists():
        return CompileResult(success=False, so_path=None,
                             error_msg=f"源文件不存在: {src_path}")
    return compile_source(src.read_text(), output_so_path, extra_flags, timeout)


def classify_error(stderr: str) -> str:
    """
    分类编译错误（对齐 AscendKernelGen 错误分布，供 Reflector 用）。

    返回类别：api_misuse / type_error / syntax_error / link_error / unknown
    """
    s = stderr.lower()
    if "no member named" in s or "no matching function" in s or "undeclared identifier" in s:
        return "api_misuse"
    if "incomplete type" in s or "cannot convert" in s or "no viable conversion" in s:
        return "type_error"
    if "expected" in s and (";" in s or "{" in s or ")" in s):
        return "syntax_error"
    if "undefined reference" in s or "ld:" in s:
        return "link_error"
    return "unknown"
