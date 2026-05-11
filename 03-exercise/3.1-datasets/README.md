# 3.1 - 数据集练习

掌握常见后训练数据集的加载、检查、清洗和格式转换。

## 练习内容

### 练习 1：加载并探索 Alpaca

```python
from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca", split="train")
print(ds[0])
print(ds.features)
print(f"总样本数: {len(ds)}")
```

**任务**：统计指令长度分布，找出 > 512 token 的样本比例。

### 练习 2：转换为 Chat 格式

将 Alpaca 格式（instruction / input / output）转换为 messages 列表格式：

```python
def alpaca_to_chat(example):
    user_content = example["instruction"]
    if example["input"]:
        user_content += "\n\n" + example["input"]
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }
```

### 练习 3：构造偏好对数据

从 `Anthropic/hh-rlhf` 加载 chosen / rejected 对，检查 rejected 的质量并过滤掉 margin 过小的样本。

## 数据质量检查清单

- [ ] 去重（基于文本相似度）
- [ ] 长度过滤（去除过短/过长样本）
- [ ] 语言检测（确保语言一致性）
- [ ] 格式验证（字段完整，无 None 值）
- [ ] 内容过滤（去除明显有害样本）
