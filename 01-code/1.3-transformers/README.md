# 1.3 - HuggingFace Transformers

Transformers 是加载、使用和微调预训练模型的核心框架。

## 核心类

- **AutoModelForCausalLM**：加载因果语言模型（GPT 系列、LLaMA、Qwen 等）
- **AutoModelForSequenceClassification**：加载分类模型（用于奖励建模）
- **Trainer / TrainingArguments**：封装完整训练循环
- **GenerationConfig**：控制解码策略（greedy / beam / sampling）

## 模型加载

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
```

## Trainer 最简示例

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
```

## 学习重点

- [ ] 理解 `from_pretrained` 的权重加载机制
- [ ] 掌握 `TrainingArguments` 中与显存相关的参数
- [ ] 学会自定义 `compute_metrics` 和 `data_collator`
