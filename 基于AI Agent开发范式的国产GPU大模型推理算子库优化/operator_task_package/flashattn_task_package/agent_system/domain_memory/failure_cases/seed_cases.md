# 已知失败案例（错误衍生监督种子数据）

> 这些是从真实 LLM 生成 + mxcc 编译中观察到的失败，供 Coder 检索规避。
> 对齐 AscendKernelGen 的错误衍生监督方法。

## [api_misuse] 编造不存在的头文件
- **症状**: `fatal error: 'mctlass/matrix_utils.h' file not found`
- **根因**: LLM 编造了 CUTLASS 有但 mctlass 没有的头文件
- **修正**: mctlass 只需 `#include "mctlass/bfloat16.h"`，不要用 matrix_utils.h / mctlass/functional.h（除已确认存在的）

## [api_misuse] dynamic shared memory 用法
- **症状**: `warning: dynamic ...` 或 shared memory 相关错误
- **根因**: MACA 的 shared memory 语法与 CUDA 略有差异
- **修正**: 用静态 shared memory 或确认 MACA 的 extern __shared__ 语法

## [type_error] __nv_bfloat16 是不完整类型
- **症状**: `subscript of pointer to incomplete type '__maca_bfloat16'`
- **根因**: MACA 的 __maca_bfloat16 是内置不完整类型，不能直接下标
- **修正**: `typedef mctlass::bfloat16_t __nv_bfloat16`，用 mctlass::bfloat16_t（完整代理类）

## [api_misuse] cuda_runtime 头文件不存在
- **症状**: `fatal error: 'cuda_bf16.h' file not found`
- **根因**: MACA 不用 cuda_* 头文件
- **修正**: 用 `#include "mctlass/bfloat16.h"`，bf16 类型由它提供

## 确认可用的 include（白名单）
```
#include <cstdint>
#include <cmath>
#include "mctlass/bfloat16.h"    // bf16 类型 + 转换
#include "mctlass/mctlass.h"     // 完整 mctlass（含 mc_runtime）
typedef mctlass::bfloat16_t __nv_bfloat16;  // 兼容 OJ 签名
```
