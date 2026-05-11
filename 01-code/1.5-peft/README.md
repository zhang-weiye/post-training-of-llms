# 1.5 - PEFT（参数高效微调）

PEFT 让我们只训练模型的极小一部分参数，大幅降低显存需求和训练成本。

## 主要方法

| 方法 | 可训练参数比例 | 原理 |
|------|--------------|------|
| LoRA | 0.1% ~ 1% | 低秩矩阵分解适配器 |
| QLoRA | 0.1% ~ 1% | 4-bit 量化基座 + LoRA |
| Prefix Tuning | < 0.1% | 可训练前缀 token |
| IA³ | < 0.1% | 学习激活缩放向量 |

## LoRA 快速上手

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # 低秩维度
    lora_alpha=32,           # 缩放系数
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()  # 查看可训练参数比例
```

## QLoRA 配置

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
```

## 学习重点

- [ ] 理解 LoRA 的数学原理：`W = W₀ + BA`
- [ ] 掌握 `r` 和 `lora_alpha` 的调参经验
- [ ] 学会合并 LoRA 权重：`model.merge_and_unload()`
