# 1.8 - Accelerate

HuggingFace Accelerate 提供设备无关的分布式训练抽象，只需少量代码改动即可从单 GPU 扩展到多 GPU / TPU。

## 核心功能

- **设备无关**：同一份代码在单卡、多卡、CPU 上均可运行
- **混合精度**：内置 FP16 / BF16 支持
- **梯度累积**：正确处理分布式场景下的梯度同步
- **DeepSpeed 集成**：通过配置文件接入 ZeRO 优化

## 基本用法

```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with accelerator.accumulate(model):
        loss = model(**batch).loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

## 配置启动

```bash
# 交互式配置
accelerate config

# 多卡启动
accelerate launch --num_processes 4 train.py

# 使用配置文件
accelerate launch --config_file config.yaml train.py
```

## 与 Trainer 的关系

`transformers.Trainer` 和 `trl.SFTTrainer` 内部都基于 Accelerate，直接使用 Trainer 时无需手动操作 Accelerator。手写训练循环时才需要显式使用 Accelerate API。
