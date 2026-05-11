# 1.2 - HuggingFace Datasets

`datasets` 库提供统一的数据加载接口，支持本地文件、Hub 在线数据集和流式处理。

## 核心功能

- **加载数据**：`load_dataset("name")` 支持 csv / json / parquet / arrow
- **流式处理**：`streaming=True` 处理超大数据集，无需全量下载
- **map 变换**：批量预处理，支持多进程加速
- **过滤与采样**：`filter`、`select`、`shuffle`
- **格式转换**：`set_format("torch")` 直接输出 Tensor

## 常用模式

```python
from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca", split="train")
ds = ds.map(lambda x: {"text": x["instruction"] + x["output"]},
            batched=True, num_proc=4)
ds = ds.filter(lambda x: len(x["text"]) < 2048)
```

## 后训练常用数据集

| 数据集 | 用途 | 规模 |
|--------|------|------|
| tatsu-lab/alpaca | SFT | 52K |
| HuggingFaceH4/ultrachat_200k | SFT | 200K |
| Anthropic/hh-rlhf | RLHF / DPO | 160K |
| openai/summarize_from_feedback | RM 训练 | 93K |
