# Jetson Orin Nano 8G 上训练 ResNet18 的完整测试方案

## 1. 目标与范围

### 1.1 目标
在 Jetson Orin Nano 8G 开发板上完成 ResNet18 训练实验，评估以下内容：
- 模型训练可行性（是否稳定训练、不报错）
- 训练效果（收敛速度、准确率）
- 性能表现（吞吐、单 epoch 耗时、GPU/内存占用）
- 训练上限（最大稳定 batch size、最长稳定训练时长、热限制/功耗限制下的极限）

### 1.2 上限定义（建议）
将“可用上限”定义为同时满足以下条件的最高配置：
- 连续训练 >= 30 分钟稳定，无 OOM、无进程崩溃
- GPU 温度不过热降频（或降频占比很低）
- 平均吞吐在该配置下达到峰值附近（与更激进配置相比无明显收益）
- 准确率曲线正常（无数值爆炸、loss 为 NaN）

---

## 2. 推荐实验策略（分三阶段）

### 阶段 A：快速功能验证（30~90 分钟）
- 数据集：CIFAR-10
- 目标：验证环境、训练脚本、日志链路是否正确
- 结果：得到第一条可收敛曲线和基础吞吐数据

### 阶段 B：性能与上限摸底（半天）
- 数据集：ImageNet-100（ImageNet 子集，100 类）或 Tiny-ImageNet
- 目标：扫描 batch size / 精度模式 / 输入分辨率，找到性能拐点
- 结果：得到最大稳定 batch size、热瓶颈位置、吞吐峰值

### 阶段 C：稳定性与效果评估（1~3 天）
- 数据集：阶段 B 同一数据集（必要时上完整 ImageNet）
- 目标：长时训练稳定性 + 最终准确率评估
- 结果：输出“推荐配置（默认）”和“极限配置（压力测试）”

---

## 3. 环境基线与前置准备

## 3.1 硬件建议
- Jetson Orin Nano 8G
- 散热：必须加风扇/主动散热（强烈建议）
- 存储：至少 64GB 可用空间（建议 NVMe SSD）
- 电源：官方推荐稳定供电（建议高质量适配器）

### 3.2 软件建议
- JetPack 6.x（优先）或 JetPack 5.1.2+
- Python 3.10+（随系统版本）
- PyTorch for Jetson（使用 NVIDIA 提供的 Jetson 兼容 wheel/容器）
- torchvision（与 torch 版本严格匹配）

### 3.3 性能模式设置（训练前每次执行）
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```
说明：
- `nvpmodel -m 0` 通常是最大性能模式（不同系统版本可能模式编号不同，以 `nvpmodel -q --verbose` 为准）。
- `jetson_clocks` 用于锁高频，避免 DVFS 波动干扰测试。

### 3.4 监控命令（单独终端）
```bash
tegrastats --interval 1000
```
记录项重点关注：
- GPU 利用率、频率
- CPU 利用率、频率
- RAM 占用
- 温度（是否触发热限制）

---

## 4. 项目结构建议

```text
resnet/
  ├─ data/
  │   ├─ cifar10/
  │   └─ imagenet100/
  ├─ logs/
  │   ├─ runs/
  │   └─ csv/
  ├─ checkpoints/
  ├─ scripts/
  │   ├─ train_resnet18.py
  │   └─ benchmark_matrix.sh
  └─ jetson_orin_nano_resnet18_training_plan.md
```

---

## 5. 训练脚本关键参数（必须支持）

建议训练脚本至少支持以下参数：
- `--dataset`：`cifar10 | imagenet100 | tiny-imagenet`
- `--data-dir`
- `--epochs`
- `--batch-size`
- `--img-size`（如 224）
- `--workers`（Jetson 建议从 2/4 起测）
- `--amp`（混合精度）
- `--grad-accum-steps`
- `--lr`, `--weight-decay`, `--momentum`
- `--label-smoothing`（可选）
- `--save-every`
- `--seed`
- 输出：每个 epoch 的 `train_loss`, `val_top1`, `val_top5`, `epoch_time`, `images_per_sec`, `max_memory`

---

## 6. 实验矩阵（核心）

### 6.1 第一轮：batch size 扫描（固定输入 224）
- 模式 1：FP32
- 模式 2：AMP（FP16 混合精度）

建议 batch 值：
- `8, 16, 24, 32, 48, 64`（遇 OOM 停止继续增大）

每个配置执行：
- 先 warmup 100 step
- 再统计 300 step 平均吞吐
- 记录是否 OOM、是否降频、是否出现 NaN

### 6.2 第二轮：输入分辨率扫描
在第一轮最佳配置附近，测试：
- `img_size = 160, 192, 224, 256`

目的：找出“精度收益 vs 训练速度”最佳平衡点。

### 6.3 第三轮：DataLoader 与稳定性
测试：
- `workers = 2, 4, 6`
- `pin_memory = True/False`
- `prefetch_factor = 2/4`（若脚本支持）

目的：避免输入管道成为瓶颈。

---

## 7. 上限测试方法（建议使用二分策略）

1. 固定：模型 ResNet18、数据集、img_size、AMP 开启。
2. 对 batch size 做二分搜索：
- 下界：可稳定训练的 batch
- 上界：必然 OOM 或明显热降频的 batch
3. 每个候选 batch 运行 30 分钟：
- 若稳定且吞吐提升明显，继续增大
- 若 OOM/频繁降频/吞吐收益接近 0，回退
4. 得到：
- 最大稳定 batch size
- 峰值吞吐（images/s）
- 推荐长期训练 batch（通常为峰值附近稍保守值）

---

## 8. 建议训练超参数（起点）

### 8.1 CIFAR-10（快速验证）
- `epochs=50`
- `optimizer=SGD(momentum=0.9)`
- `lr=0.1 * (global_batch/256)`
- `weight_decay=5e-4`
- `scheduler=cosine`
- `amp=True`

### 8.2 ImageNet-100 / Tiny-ImageNet（性能与效果）
- `epochs=90`
- `optimizer=SGD(momentum=0.9)`
- `lr=0.1 * (global_batch/256)`
- `weight_decay=1e-4`
- `scheduler=cosine + 5 epoch warmup`
- `label_smoothing=0.1`
- `amp=True`

---

## 9. 命令模板（可直接改参数执行）

### 9.1 快速验证
```bash
python scripts/train_resnet18.py \
  --dataset cifar10 \
  --data-dir ./data/cifar10 \
  --epochs 10 \
  --img-size 224 \
  --batch-size 32 \
  --workers 4 \
  --amp \
  --seed 42
```

### 9.2 上限测试单次运行
```bash
python scripts/train_resnet18.py \
  --dataset imagenet100 \
  --data-dir ./data/imagenet100 \
  --epochs 5 \
  --img-size 224 \
  --batch-size 48 \
  --workers 4 \
  --amp \
  --seed 42
```

### 9.3 稳定性长跑
```bash
python scripts/train_resnet18.py \
  --dataset imagenet100 \
  --data-dir ./data/imagenet100 \
  --epochs 90 \
  --img-size 224 \
  --batch-size <best_batch> \
  --workers 4 \
  --amp \
  --save-every 5 \
  --seed 42
```

---

## 10. 结果记录模板（建议直接填表）

| 时间 | 数据集 | img_size | batch | AMP | workers | 吞吐(img/s) | epoch时长(s) | Val Top1 | GPU温度峰值 | RAM峰值 | 是否降频 | 备注 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-xx-xx | CIFAR10 | 224 | 32 | 是 | 4 |  |  |  |  |  | 否 |  |

建议额外输出：
- 最佳性价比配置（速度/精度平衡）
- 峰值配置（压力测试）
- 长时稳定配置（生产推荐）

---

## 11. 验收标准（可量化）

最少满足：
- 成功完成 1 次 CIFAR-10 收敛训练（Top1 明显上升）
- 完成至少 8 组配置对比（batch/AMP/workers）
- 找到最大稳定 batch size
- 形成最终结论：
  - 推荐默认训练配置
  - 可达到的吞吐上限
  - 主要瓶颈（内存/散热/IO）

---

## 12. 常见问题与处理

- OOM：
  - 降低 `batch_size`
  - 开启 AMP
  - 降低 `img_size`
  - 增大 `grad_accum_steps` 以保持等效 batch

- 吞吐低但 GPU 利用率不高：
  - 提升 `workers`
  - 数据放到 NVMe
  - 检查数据增强是否过重

- 训练中后期变慢：
  - 检查温度是否持续升高导致降频
  - 改善散热（风扇曲线、环境温度）

- 结果不稳定：
  - 固定随机种子
  - 固定电源和性能模式
  - 每组配置重复 2~3 次取均值

---

## 13. 交付产物清单

实验完成后建议输出：
- 训练日志（TensorBoard 或 CSV）
- 最优模型权重（`checkpoints/best.pth`）
- 配置汇总表（CSV/Markdown）
- 一页结论报告：
  - 最佳配置
  - 上限配置
  - 稳定配置
  - 风险与下一步优化方向（量化、蒸馏、剪枝、TensorRT 推理）

---

## 14. 你可以直接照这个顺序执行

1. 设定性能模式（`nvpmodel` + `jetson_clocks`）
2. 跑 CIFAR-10 10 epoch 确认链路
3. 跑 batch 扫描矩阵（FP32 vs AMP）
4. 做分辨率扫描（160/192/224/256）
5. 做 workers 扫描（2/4/6）
6. 30 分钟稳定性测试确认上限
7. 用推荐配置跑完整训练并导出报告

如果你愿意，我下一步可以继续给你生成：
- `scripts/train_resnet18.py` 的可运行版本（Jetson 友好）
- `scripts/benchmark_matrix.sh` 自动化批量实验脚本
- 结果汇总脚本（自动生成 CSV + 排行结论）
