# 1.4 - HuggingFace Tokenizers

Tokenizer 将原始文本转换为模型可消费的 token id，正确使用分词器对后训练效果至关重要。

## 核心概念

- **词表（Vocabulary）**：BPE / WordPiece / Unigram 算法
- **特殊 token**：`<|im_start|>`、`<|endoftext|>`、`[INST]` 等对话格式标记
- **Chat Template**：将多轮对话转换为模型输入格式
- **Padding & Truncation**：批处理时的对齐策略

## Chat Template 使用

```python
messages = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的？"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

## 后训练中的注意事项

- **Labels masking**：SFT 时只对 assistant 回复部分计算 loss，instruction 部分设为 -100
- **EOS token**：确保每条样本末尾有正确的结束符
- **padding_side**：训练时用 right padding，生成时用 left padding

## 学习重点

- [ ] 理解不同模型的 chat template 格式差异
- [ ] 掌握如何构造带 label mask 的训练样本
