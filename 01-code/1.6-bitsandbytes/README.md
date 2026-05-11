# 1.6 - bitsandbytes

bitsandbytes 提供 GPU 上的 8-bit 和 4-bit 量化，是 QLoRA 方案的核心依赖。

## 量化方案

- **LLM.int8()**：8-bit 矩阵乘法，基本无精度损失
- **NF4**：4-bit 正态浮点，QLoRA 默认格式，信息密度最优
- **FP4**：4-bit 浮点，精度略低于 NF4

## 使用示例

```python
from transformers import BitsAndBytesConfig
import torch

# 8-bit 量化
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
)

# 4-bit 量化（NF4 + double quantization）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # 对量化常数再次量化，额外节省 0.4 bit/param
)
```

## 显存对比（7B 模型）

| 精度 | 显存占用 |
|------|---------|
| FP32 | ~28 GB |
| BF16 | ~14 GB |
| INT8 | ~7 GB |
| NF4  | ~4 GB |

## 注意事项

- 量化模型不能直接训练，需配合 PEFT/LoRA 使用
- 推理时量化会轻微降低精度，生产环境需评估
- 需要 CUDA 环境，CPU-only 不支持
