# CANN 版本检测与验证指南

本指南介绍如何检测 CANN (Compute Architecture for Neural Networks) 安装状态、区分 8.5+ 与 8.5 之前版本的差异，并正确配置环境变量。

> **注意**：本文档仅涵盖 CANN **检测与验证**，不包含 CANN 安装教程。

---

## 1. CANN 安装验证

### 1.1 检查 CANN 是否已安装

```bash
# 检查 CANN 安装目录
ls -la /usr/local/Ascend/
```

输出示例：

| CANN 版本 | 目录结构 |
|-----------|----------|
| CANN 8.5+ | `/usr/local/Ascend/cann/` |
| CANN 8.5 之前 | `/usr/local/Ascend/ascend-toolkit/` |

### 1.2 验证 CANN 版本

```bash
# 方法一：查看版本文件
if [ -f /usr/local/Ascend/cann/version.cfg ]; then
    cat /usr/local/Ascend/cann/version.cfg
elif [ -f /usr/local/Ascend/ascend-toolkit/version.cfg ]; then
    cat /usr/local/Ascend/ascend-toolkit/version.cfg
else
    echo "CANN version.cfg not found"
fi

# 方法二：通过 npu-smi 查询
npu-smi info -t version
```

---

## 2. 版本检测逻辑

### 2.1 自动检测脚本

```bash
#!/bin/bash
# detect_cann_version.sh - 自动检测 CANN 版本并加载环境

if [ -d "/usr/local/Ascend/cann" ]; then
    # CANN 8.5+ 新路径结构
    echo "Detected CANN 8.5+"
    CANN_HOME="/usr/local/Ascend/cann"
    ENV_SCRIPT="${CANN_HOME}/set_env.sh"
else
    # CANN 8.5 之前旧路径结构
    echo "Detected CANN before 8.5"
    CANN_HOME="/usr/local/Ascend/ascend-toolkit"
    ENV_SCRIPT="${CANN_HOME}/set_env.sh"
fi

# 加载环境
if [ -f "$ENV_SCRIPT" ]; then
    source "$ENV_SCRIPT"
    echo "CANN environment loaded from: $ENV_SCRIPT"
else
    echo "Error: CANN environment script not found at $ENV_SCRIPT"
    exit 1
fi
```

### 2.2 Python 检测脚本

```python
#!/usr/bin/env python3
# detect_cann_version.py

import os

def get_cann_home():
    """自动检测 CANN 安装路径"""
    if os.path.isdir("/usr/local/Ascend/cann"):
        return "/usr/local/Ascend/cann", "8.5+"
    elif os.path.isdir("/usr/local/Ascend/ascend-toolkit"):
        return "/usr/local/Ascend/ascend-toolkit", "<8.5"
    else:
        return None, None

def main():
    cann_home, version = get_cann_home()
    
    if cann_home is None:
        print("Error: CANN not found")
        return 1
    
    print(f"CANN Version: {version}")
    print(f"CANN Home: {cann_home}")
    
    # 检查环境变量
    env_vars = ["ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "LD_LIBRARY_PATH"]
    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        status = "OK" if value != "NOT SET" else "MISSING"
        print(f"{var}: {status}")
    
    return 0

if __name__ == "__main__":
    exit(main())
```

---

## 3. 环境变量说明

### 3.1 CANN 8.5+ 环境变量

| 环境变量 | 说明 | 示例值 |
|----------|------|--------|
| `ASCEND_HOME_PATH` | CANN 安装根目录 | `/usr/local/Ascend/cann` |
| `ASCEND_OPP_PATH` | 算子库路径 | `${ASCEND_HOME_PATH}/opp` |
| `ASCEND_AICPU_PATH` | AI CPU 路径 | `${ASCEND_HOME_PATH}/runtime` |
| `LD_LIBRARY_PATH` | 库搜索路径 | `${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/runtime/lib64` |
| `PYTHONPATH` | Python 模块路径 | `${ASCEND_HOME_PATH}/python/site-packages` |

### 3.2 CANN 8.5 之前环境变量

| 环境变量 | 说明 | 示例值 |
|----------|------|--------|
| `ASCEND_HOME_PATH` | CANN 安装根目录 | `/usr/local/Ascend/ascend-toolkit/latest` |
| `ASCEND_OPP_PATH` | 算子库路径 | `${ASCEND_HOME_PATH}/opp` |
| `ASCEND_AICPU_PATH` | AI CPU 路径 | `${ASCEND_HOME_PATH}/runtime` |
| `LD_LIBRARY_PATH` | 库搜索路径 | `${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/runtime/lib64` |
| `PYTHONPATH` | Python 模块路径 | `${ASCEND_HOME_PATH}/python/site-packages` |

### 3.3 验证环境变量

```bash
# 检查关键环境变量
echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
echo "ASCEND_OPP_PATH: $ASCEND_OPP_PATH"
echo "ASCEND_AICPU_PATH: $ASCEND_AICPU_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"

# 验证库文件是否存在
ls -la ${ASCEND_HOME_PATH}/lib64/libascendcl.so 2>/dev/null && echo "OK: ascendcl library found" || echo "Error: ascendcl library missing"
```

---

## 4. 常见错误及解决方案

### 4.1 CANN 未找到

**错误信息**：
```
Error: CANN not found
Error: CANN environment script not found
```

**解决方案**：
1. 确认 CANN 已安装：
   ```bash
   ls -la /usr/local/Ascend/
   ```
2. 如未安装，请参考华为官方文档安装 CANN
3. 检查安装路径是否正确

### 4.2 环境变量未设置

**错误信息**：
```
ASCEND_HOME_PATH: NOT SET
Error: cannot find CANN libraries
```

**解决方案**：
```bash
# 手动设置环境变量
export ASCEND_HOME_PATH=/usr/local/Ascend/cann  # 或 ascend-toolkit
source ${ASCEND_HOME_PATH}/set_env.sh

# 添加到 ~/.bashrc
if ! grep -q "ASCEND_HOME_PATH" ~/.bashrc; then
    echo "export ASCEND_HOME_PATH=/usr/local/Ascend/cann" >> ~/.bashrc
    echo "source \${ASCEND_HOME_PATH}/set_env.sh" >> ~/.bashrc
fi
```

### 4.3 版本不匹配

**错误信息**：
```
ImportError: libascendcl.so: cannot open shared object file
RuntimeError: CANN version mismatch
```

**解决方案**：
1. 检查 torch_npu 与 CANN 版本兼容性
2. 确认加载了正确版本的 CANN 环境
3. 重新加载环境：
   ```bash
   unset ASCEND_HOME_PATH
   source /usr/local/Ascend/cann/set_env.sh  # 或 ascend-toolkit
   ```

### 4.4 权限错误

**错误信息**：
```
Permission denied: /usr/local/Ascend/
```

**解决方案**：
```bash
# 检查目录权限
ls -ld /usr/local/Ascend/

# 确保当前用户有读取权限
# 如需修改权限（需 root）：
sudo chmod -R 755 /usr/local/Ascend/
```

### 4.5 Python 无法导入 CANN 模块

**错误信息**：
```
ModuleNotFoundError: No module named 'acl'
ImportError: cannot import name 'aclruntime'
```

**解决方案**：
```bash
# 检查 PYTHONPATH
python3 -c "import sys; print('\n'.join(sys.path))"

# 确保包含 CANN Python 路径
export PYTHONPATH=${ASCEND_HOME_PATH}/python/site-packages:${PYTHONPATH}
```

---

## 5. 快速检查清单

部署前请确认：

- [ ] CANN 已安装（`ls /usr/local/Ascend/`）
- [ ] 版本正确识别（8.5+ 或 <8.5）
- [ ] 环境变量已设置（`echo $ASCEND_HOME_PATH`）
- [ ] 库文件可访问（`ls $ASCEND_HOME_PATH/lib64/libascendcl.so`）
- [ ] Python 模块可导入（`python3 -c "import acl"`）

---

## 6. 参考文档

- [华为昇腾官方文档](https://www.hiascend.com/document)
- [CANN 环境变量说明](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/envdeployment/instg/instg_0045.html)
