---
name: diffusers-env-setup
description: HuggingFace Diffusers 环境配置指南，用于华为昇腾 NPU。覆盖 CANN 版本检测（8.5 vs 旧版）、PyTorch + torch_npu 安装、Diffusers 库安装及环境验证。当用户需要在昇腾 NPU 上配置 Diffusers 环境、安装 torch_npu、或验证开发环境时使用。
keywords:
    - diffusers
    - pytorch
    - torch_npu
    - cann
    - npu
    - environment
---

# Diffusers 昇腾 NPU 环境配置

本 Skill 指导用户在华为昇腾 NPU 上配置 HuggingFace Diffusers 开发环境，涵盖 CANN 环境验证、PyTorch + torch_npu 安装、Diffusers 库安装及完整的环境验证流程。

## 快速开始

在昇腾 NPU 上快速搭建 Diffusers 环境：

```bash
# 1. 激活 CANN 环境（自动检测版本）
if [ -d "/usr/local/Ascend/cann" ]; then
    source /usr/local/Ascend/cann/set_env.sh
else
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# 2. 安装 PyTorch + torch_npu
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-npu==2.7.1

# 3. 安装 Diffusers
pip install diffusers["torch"] transformers

# 4. 验证环境
python scripts/validate_environment.py
```

## CANN 验证

### 检测 CANN 版本

CANN 8.5 及之后版本采用新的目录结构。使用以下逻辑自动检测：

```bash
# CANN 8.5+
if [ -d "/usr/local/Ascend/cann" ]; then
    source /usr/local/Ascend/cann/set_env.sh
    echo "CANN 8.5+ detected"
# CANN 8.5 之前
elif [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "CANN (before 8.5) detected"
else
    echo "Error: CANN not found"
fi
```

### 验证 CANN 环境变量

激活环境后，检查以下变量是否设置：

```bash
echo $ASCEND_HOME_PATH
echo $ASCEND_OPP_PATH
echo $ASCEND_AICPU_PATH
```

正常输出应显示具体路径，而非空值。

### Python 验证

```python
import os

# 检查关键环境变量
required_vars = ["ASCEND_HOME_PATH", "ASCEND_OPP_PATH"]
for var in required_vars:
    value = os.environ.get(var)
    if value:
        print(f"✓ {var} = {value}")
    else:
        print(f"✗ {var} not set")
```

**注意**：本 Skill 假设 CANN 已预先安装。如需安装 CANN，请参考[官方文档](https://www.hiascend.com/document)。

## PyTorch + torch_npu 安装

### 版本配套要求

| PyTorch 版本 | torch_npu 版本 | Python 版本 | 推荐 CANN |
|--------------|----------------|-------------|-----------|
| 2.7.1        | 2.7.1          | 3.9 - 3.11  | 8.0.RC3+  |
| 2.5.1        | 2.5.1          | 3.9 - 3.11  | 8.0.RC1+  |
| 2.1.0        | 2.1.0          | 3.8 - 3.11  | 7.0+      |

### 安装步骤

**1. 安装前置依赖**

```bash
# numpy 必须 < 2.0（关键要求）
pip install "numpy<2.0"
pip install pyyaml setuptools
```

**2. 安装 PyTorch**

```bash
# x86 架构
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu

# aarch64 架构
pip install torch==2.7.1
```

**3. 安装 torch_npu**

```bash
# 方式一：从 PyPI 安装（推荐）
pip install torch-npu==2.7.1

# 方式二：从 GitCode Release 下载
# 访问 https://gitcode.com/Ascend/pytorch/releases
# 下载对应版本的 whl 文件后安装
pip install torch_npu-2.7.1-cp310-cp310-linux_x86_64.whl
```

### 快速验证

```python
import torch
import torch_npu

# 检查 NPU 可用性
print(f"PyTorch version: {torch.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU count: {torch.npu.device_count()}")

# 简单张量测试
if torch.npu.is_available():
    x = torch.tensor([1.0, 2.0, 3.0]).npu()
    print(f"Tensor on NPU: {x}")
```

## Diffusers 安装

### 标准安装

根据[官方安装指南](https://huggingface.co/docs/diffusers/en/installation)：

```bash
# 基础安装（仅 PyTorch 后端）
pip install diffusers["torch"]

# 完整安装（推荐用于开发）
pip install diffusers["torch"] transformers accelerate

# 带可选依赖的完整安装
pip install diffusers["torch"] transformers accelerate safetensors
```

### 开发版安装

如需使用最新特性或修复：

```bash
# 从源码安装
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e ".[torch]"
```

### 版本兼容性说明

| Diffusers 版本 | 推荐 PyTorch | 推荐 transformers | 说明 |
|----------------|--------------|-------------------|------|
| 0.30.0+        | 2.0+         | 4.40.0+           | 支持 SD3、Flux |
| 0.28.0+        | 2.0+         | 4.30.0+           | 支持 SDXL |
| 0.21.0+        | 1.13+        | 4.25.0+           | 支持 ControlNet |

### 模型缓存配置

Diffusers 默认将模型下载到用户目录的缓存中。可配置缓存位置：

```bash
# 设置环境变量
export HF_HOME="/path/to/your/cache"
export HF_HUB_CACHE="/path/to/your/hub/cache"
```

或在代码中指定：

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir="/path/to/cache"
)
```

### 禁用遥测

Diffusers 会收集使用统计信息。如需禁用：

```bash
export HF_HUB_DISABLE_TELEMETRY=1
```

## 环境验证

### 运行验证脚本

```bash
python scripts/validate_environment.py
```

该脚本执行以下检查：

| 检查项 | 通过标准 | 失败处理 |
|--------|----------|----------|
| CANN 安装 | 目录存在 | 检查安装路径 |
| CANN 环境变量 | ASCEND_HOME_PATH 已设置 | 执行 source set_env.sh |
| PyTorch 安装 | import 成功 | 重新安装 PyTorch |
| torch_npu 安装 | import 成功 | 重新安装 torch_npu |
| NPU 可见性 | device_count() > 0 | 检查硬件和驱动 |
| Diffusers 安装 | import 成功 | 重新安装 diffusers |
| numpy 版本 | < 2.0 | 降级 numpy |

### 手动验证流程

```python
# 完整环境验证
import sys

def check_environment():
    results = []
    
    # 1. 检查 CANN 环境
    import os
    cann_path = os.environ.get("ASCEND_HOME_PATH")
    results.append(("CANN Path", cann_path or "Not set"))
    
    # 2. 检查 PyTorch
    try:
        import torch
        results.append(("PyTorch", torch.__version__))
    except ImportError:
        results.append(("PyTorch", "Not installed"))
        return results
    
    # 3. 检查 torch_npu
    try:
        import torch_npu
        results.append(("torch_npu", torch_npu.__version__))
    except ImportError:
        results.append(("torch_npu", "Not installed"))
    
    # 4. 检查 NPU
    if torch.npu.is_available():
        results.append(("NPU Available", f"Yes ({torch.npu.device_count()} devices)"))
    else:
        results.append(("NPU Available", "No"))
    
    # 5. 检查 Diffusers
    try:
        import diffusers
        results.append(("Diffusers", diffusers.__version__))
    except ImportError:
        results.append(("Diffusers", "Not installed"))
    
    # 6. 检查 numpy
    import numpy as np
    np_version = np.__version__
    results.append(("NumPy", f"{np_version} {'✓' if int(np_version.split('.')[0]) < 2 else '✗ (need < 2.0)'}"))
    
    return results

# 输出结果
for name, value in check_environment():
    print(f"{name:20s}: {value}")
```

## 参考资源

### 本 Skill 内部参考

- **CANN 版本说明**：[references/cann-versions.md](references/cann-versions.md) - 详细的版本检测和差异说明
- **故障排查**：[references/troubleshooting.md](references/troubleshooting.md) - 常见问题和解决方案

### 外部文档

- **Diffusers 官方文档**：[huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)
- **torch_npu 文档**：[昇腾 PyTorch 扩展文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/torchnpuCustomsapi/context/概述.md)
- **CANN 文档**：[昇腾社区文档中心](https://www.hiascend.com/document)
- **torch_npu Release**：[GitCode Ascend/pytorch](https://gitcode.com/Ascend/pytorch/releases)

### 相关 Skills

- **torch_npu** - PyTorch 昇腾扩展的完整使用指南
- **msmodelslim** - 模型量化压缩（可用于 Diffusers 模型）
- **vllm-ascend** - 大模型推理服务（部分 Diffusers 模型支持）
