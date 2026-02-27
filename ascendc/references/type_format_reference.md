## CANN DataType / Format 参考

- **DataType 枚举（`ge::DataType`）**
  - 定义位置：`graph/types.h` 中的 `enum DataType`，典型路径：
    - `/usr/local/Ascend/cann-8.5.0-beta.1/aarch64-linux/include/graph/types.h`（约 80–123 行）
  - 常用取值示例：
    - `ge::DT_FLOAT`
    - `ge::DT_FLOAT16`
    - `ge::DT_BF16`
    - `ge::DT_INT8`
    - `ge::DT_INT32`
    - `ge::DT_BOOL`

- **Format 枚举（`ge::Format`）**
  - 定义位置：同一文件 `graph/types.h` 中的 `enum Format`，典型路径：
    - `/usr/local/Ascend/cann-8.5.0-beta.1/aarch64-linux/include/graph/types.h`（约 189–247 行）
  - 常用取值示例：
    - `ge::FORMAT_ND`
    - `ge::FORMAT_NCHW`
    - `ge::FORMAT_NHWC`
    - `ge::FORMAT_FRACTAL_NZ`

- **JSON → C++ 映射约定（示例）**
  - `type: "fp16"` → `ge::DT_FLOAT16`
  - `type: "bf16"` → `ge::DT_BF16`
  - `type: "float"` → `ge::DT_FLOAT`
  - `type: "int32"` → `ge::DT_INT32`
  - `format: "ND"` → `ge::FORMAT_ND`

> 约定：所有新算子的 `op_host` `.DataType()` / `.Format()`，以及 tiling 中的 dtype 校验，都应只使用 `graph/types.h` 中已经定义的枚举值，并与算子 JSON 中的 `type` / `format` 保持一致。

