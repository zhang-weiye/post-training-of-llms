# 1.1 - PyTorch

PyTorch 是 LLM 训练的底层引擎，理解其核心机制对调优训练过程至关重要。

## 核心概念

- **Tensor & Autograd**：计算图构建与梯度反传
- **nn.Module**：模型定义与参数管理
- **DataLoader**：批量数据加载与采样策略
- **Optimizer & Scheduler**：AdamW、余弦退火、Warmup
- **Mixed Precision**：`torch.autocast` + GradScaler
- **Gradient Checkpointing**：以计算换显存

## 关键 API

```python
# 混合精度训练模板
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(inputs)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 学习重点

- [ ] 理解反向传播中梯度的流动
- [ ] 掌握显存管理：`del`、`torch.no_grad()`、gradient checkpointing
- [ ] 熟悉 `torch.distributed` 基本原语（为多 GPU 训练打基础）
