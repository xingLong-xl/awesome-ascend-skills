---
name: diffusers-ascend-pipeline
description: Diffusers Pipeline 推理指南，用于华为昇腾 NPU。覆盖环境预检、通用 Pipeline 推理（图像/视频模型）、内存优化（CPU offload、attention slicing、VAE slicing）和 LoRA 加载与融合。当用户需要在昇腾 NPU 上运行 Diffusers 推理时使用。
keywords:
    - diffusers
    - pipeline
    - inference
    - npu
    - image-generation
    - video-generation
    - lora
    - memory-optimization
    - flux
    - sdxl
    - wan
    - cogvideox
---

# Diffusers 昇腾 NPU Pipeline 推理

本 Skill 指导用户在华为昇腾 NPU 上运行 HuggingFace Diffusers Pipeline 推理，适用于任意 Diffusers 模型。

## 前置要求

| 依赖 | 说明 | 参考 |
|------|------|------|
| CANN + torch_npu | NPU 运行环境 | [diffusers-ascend-env-setup](../diffusers-ascend-env-setup/SKILL.md) |
| 模型权重 | 真实权重或假权重 | [diffusers-ascend-weight-prep](../diffusers-ascend-weight-prep/SKILL.md) |
| diffusers | `pip install diffusers["torch"] transformers accelerate` | |

## 快速开始

### 1. 预检

运行预检脚本，确认环境、NPU 内存和模型权重就绪：

```bash
python scripts/validate_pipeline.py --model ./my_model --device npu:0 --min-memory 16
```

检查项：

| 检查项 | 说明 |
|--------|------|
| Python 包 | torch, torch_npu, diffusers, transformers |
| CANN 环境 | ASCEND_HOME_PATH 等环境变量 |
| NPU 可用性 | 设备数量和名称 |
| NPU 内存 | 空闲内存是否满足最低要求 |
| 模型权重 | model_index.json 存在，组件完整 |

### 2. 图像模型推理

**通用模式**（适用于 FLUX、SDXL、SD3 等任何图像 Pipeline）：

```python
import torch
import torch_npu

from diffusers import DiffusionPipeline

# 加载 Pipeline（自动识别 Pipeline 类型）
pipe = DiffusionPipeline.from_pretrained(
    "./my_model",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("npu:0")

# 生成图像
generator = torch.Generator("npu").manual_seed(42)
image = pipe(
    prompt="a cat sitting on a windowsill, watercolor style",
    num_inference_steps=20,
    guidance_scale=3.5,
    generator=generator,
).images[0]

image.save("output.png")
```

**使用脚本：**

```bash
# FLUX.1-dev 推理
python scripts/run_pipeline.py \
    --model ./fake_flux_dev \
    --prompt "a cat sitting on a windowsill" \
    --device npu:0 --dtype bfloat16 \
    --steps 20 --seed 42 \
    --output flux_output.png --benchmark

# SDXL 推理
python scripts/run_pipeline.py \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --prompt "a beautiful landscape" \
    --device npu:0 --dtype float16 \
    --steps 30 --guidance-scale 7.5 \
    --output sdxl_output.png --benchmark
```

### 3. 视频模型推理

视频 Pipeline（Wan、CogVideoX 等）输出帧序列，脚本自动检测并导出为 MP4：

```python
import torch
import torch_npu

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained(
    "./wan_model",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("npu:0")

generator = torch.Generator("npu").manual_seed(42)
output = pipe(
    prompt="a dog running on the beach",
    num_inference_steps=30,
    generator=generator,
)

# 导出视频
export_to_video(output.frames[0], "output.mp4", fps=16)
```

**使用脚本：**

```bash
python scripts/run_pipeline.py \
    --model ./wan_model \
    --prompt "a dog running on the beach" \
    --device npu:0 --dtype bfloat16 \
    --steps 30 --seed 42 \
    --output wan_output.mp4 --benchmark
```

### 4. 内存优化

大模型推理时可能遇到 NPU 内存不足，可组合使用以下优化技术：

| 技术 | 方法 | NPU 兼容 | 内存节省 |
|------|------|---------|---------|
| Attention Slicing | `pipe.enable_attention_slicing()` | ✅ 验证通过 | 中等 |
| VAE Slicing | `pipe.enable_vae_slicing()` | ✅ 验证通过 | 低 |
| VAE Tiling | `pipe.enable_vae_tiling()` | ✅ 验证通过 | 高（高分辨率） |
| BF16 推理 | `torch_dtype=torch.bfloat16` | ✅ 推荐 | 约 50% |
| Sequential CPU Offload | `pipe.enable_sequential_model_cpu_offload()` | ⚠️ 实验性 | 很高 |
| Model CPU Offload | `pipe.enable_model_cpu_offload()` | ⚠️ 实验性 | 高 |

```bash
# 使用内存优化运行
python scripts/run_pipeline.py \
    --model ./fake_flux_dev \
    --prompt "a landscape" \
    --device npu:0 --dtype bfloat16 \
    --steps 20 \
    --attention-slicing --vae-tiling \
    --output optimized.png --benchmark
```

详细指南：[references/memory-optimization.md](references/memory-optimization.md)

### 5. LoRA 集成

加载 LoRA 适配器增强生成效果：

```python
import torch
import torch_npu

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("./my_model", torch_dtype=torch.bfloat16)
pipe = pipe.to("npu:0")

# 加载 LoRA
pipe.load_lora_weights("./my_lora", adapter_name="style")

# 生成（控制 LoRA 强度）
image = pipe(
    prompt="a portrait in oil painting style",
    num_inference_steps=20,
    cross_attention_kwargs={"scale": 0.8},
).images[0]
```

**多 LoRA 叠加：**

```python
pipe.load_lora_weights("./lora_style", adapter_name="style")
pipe.load_lora_weights("./lora_detail", adapter_name="detail")
pipe.set_adapters(["style", "detail"], adapter_weights=[0.7, 0.3])
```

**使用脚本：**

```bash
python scripts/run_pipeline.py \
    --model ./my_model \
    --prompt "a portrait" \
    --device npu:0 --dtype bfloat16 \
    --lora ./my_lora --lora-scale 0.8 \
    --output lora_output.png
```

详细指南：[references/lora-guide.md](references/lora-guide.md)

## 性能基准测试

使用 benchmark 脚本测量推理性能：

```bash
python scripts/benchmark_pipeline.py \
    --model ./model_weights \
    --prompt "a photo of a cat" \
    --num-runs 5 \
    --warmup-runs 1 \
    --output-json benchmark_results.json
```

关键指标：

| 指标 | 说明 |
|------|------|
| 首次推理延迟 | 含图编译/缓存构建，通常较慢 |
| 平均推理延迟 | 稳态性能（排除预热） |
| P50 / P95 延迟 | 延迟分布 |
| NPU 内存峰值 | `torch.npu.max_memory_allocated()` |
| 吞吐量 | images/sec 或 frames/sec |

## 脚本参考

### run_pipeline.py

通用 Pipeline 推理脚本，支持任意 Diffusers 模型：

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `--model` | 是 | - | 模型路径（本地目录或 HF 模型 ID） |
| `--prompt` | 是 | - | 生成提示词 |
| `--device` | 否 | `npu:0` | 目标设备 |
| `--dtype` | 否 | `bfloat16` | 数据类型：float32, float16, bfloat16 |
| `--steps` | 否 | `20` | 推理步数 |
| `--seed` | 否 | `42` | 随机种子 |
| `--output` | 否 | `output.png` | 输出文件（.png 图像 / .mp4 视频） |
| `--width` | 否 | 模型默认 | 输出宽度 |
| `--height` | 否 | 模型默认 | 输出高度 |
| `--guidance-scale` | 否 | `3.5` | Classifier-free guidance |
| `--lora` | 否 | - | LoRA 权重路径 |
| `--lora-scale` | 否 | `1.0` | LoRA 强度 |
| `--attention-slicing` | 否 | 关闭 | 启用 attention slicing |
| `--vae-slicing` | 否 | 关闭 | 启用 VAE slicing |
| `--vae-tiling` | 否 | 关闭 | 启用 VAE tiling |
| `--cpu-offload` | 否 | 关闭 | 启用 sequential CPU offload |
| `--benchmark` | 否 | 关闭 | 打印详细计时指标 |

### validate_pipeline.py

推理前预检脚本：

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `--model` | 否 | - | 模型路径（提供时检查权重结构） |
| `--device` | 否 | `npu:0` | 目标设备 |
| `--min-memory` | 否 | `16` | 最低空闲 NPU 内存（GB） |

### benchmark_pipeline.py

详细性能基准测试脚本：

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `--model` | 是 | - | 模型权重路径 |
| `--prompt` | 是 | - | 生成提示词 |
| `--device` | 否 | `npu:0` | 设备 |
| `--dtype` | 否 | `bfloat16` | 数据类型 |
| `--steps` | 否 | `20` | 推理步数 |
| `--warmup-runs` | 否 | `1` | 预热次数 |
| `--num-runs` | 否 | `5` | 测试次数 |
| `--output-json` | 否 | - | 结果 JSON 输出路径 |

## 常见模型参考

| 模型 | Pipeline 类型 | 推荐 dtype | 推荐 NPU 内存 |
|------|-------------|-----------|-------------|
| FLUX.1-dev | FluxPipeline | bfloat16 | ≥24 GB |
| SDXL | StableDiffusionXLPipeline | float16 | ≥8 GB |
| SD 3.5 | StableDiffusion3Pipeline | bfloat16 | ≥16 GB |
| Wan 2.1 | WanPipeline | bfloat16 | ≥24 GB |
| CogVideoX | CogVideoXPipeline | bfloat16 | ≥24 GB |

## 常见问题

### NPU 内存不足（OOM）

```
RuntimeError: NPU out of memory
```

解决：启用内存优化（`--attention-slicing --vae-tiling`），使用 `bfloat16`，减少分辨率或步数。

### 首次推理很慢

NPU 首次推理需要编译计算图，后续推理速度正常。脚本的 `--benchmark` 模式会自动进行预热。

### Generator 设备错误

```
RuntimeError: Expected a 'npu' device type for generator
```

解决：使用 `torch.Generator("npu")` 而非 `torch.Generator("cpu")`。

### CPU Offload 不生效

`accelerate` 的设备钩子可能不完全支持 `npu`。建议优先使用 attention slicing 和 VAE tiling。

更多问题：[references/troubleshooting.md](references/troubleshooting.md)

## 参考资源

- [环境配置](../diffusers-ascend-env-setup/SKILL.md) - Phase #1: CANN + torch_npu 环境搭建
- [权重准备](../diffusers-ascend-weight-prep/SKILL.md) - Phase #2: 模型下载与假权重生成
- [内存优化详解](references/memory-optimization.md) - NPU 内存优化完整指南
- [LoRA 详解](references/lora-guide.md) - LoRA 加载、多 LoRA、权重融合
- [故障排查](references/troubleshooting.md) - NPU 推理常见问题与修复
- [Diffusers 官方文档](https://huggingface.co/docs/diffusers)
- [昇腾 PyTorch 扩展](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/torchnpuCustomsapi/context/概述.md)
