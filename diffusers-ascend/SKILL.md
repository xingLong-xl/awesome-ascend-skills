---
name: diffusers-ascend
description: HuggingFace Diffusers on Huawei Ascend NPU. Parent skill providing overview of Diffusers deployment on Ascend, including environment setup, inference optimization, and model compatibility. Use when users need comprehensive Diffusers + Ascend guidance or want to understand available sub-skills.
keywords:
    - diffusers
    - stable-diffusion
    - image-generation
    - ascend
    - npu
---

# Diffusers on Ascend NPU

华为昇腾 NPU 上的 HuggingFace Diffusers 部署指南。

## 可用技能

| 技能 | 用途 |
|------|------|
| [diffusers-ascend-env-setup](skills/diffusers-ascend-env-setup/SKILL.md) | 环境配置（CANN、PyTorch、torch_npu、Diffusers） |

*即将推出：推理优化、模型量化、分布式推理*

## 快速开始

```bash
# 1. 环境验证
cd skills/diffusers-ascend-env-setup
python scripts/validate_environment.py

# 2. 安装依赖
pip install diffusers["torch"] transformers

# 3. 运行推理
python -c "from diffusers import StableDiffusionPipeline; print('Ready!')"
```

## 版本兼容性

| 组件 | 版本 | 说明 |
|------|------|------|
| CANN | 8.0.RC1+ | NPU 支持必需 |
| PyTorch | 2.5.1+ | 含 torch_npu 扩展 |
| Diffusers | 0.28.0+ | Stable Diffusion、SDXL、SD3 |

## 参考资源

- [Diffusers 官方文档](https://huggingface.co/docs/diffusers)
- [昇腾社区文档](https://www.hiascend.com/document)
- [torch_npu 参考](../torch_npu/SKILL.md)
