# TRAC-Net 累计实验结果

## 实验设置

- 数据：竞赛提供的 CMU-MOSEI `aligned_50.pkl`。
- 划分：训练 3395 条，验证 728 条，测试 727 条。
- 文本：本地 `bert-base-uncased`，参数完全冻结。
- 模态维度：文本 768、语音 74、视觉 35。
- 公共隐空间：128 维。
- 时序编码：每个模态 2 层 Transformer，4 个注意力头。
- 局部补偿窗口：左右各 2 个位置。
- 优化：AdamW，初始学习率 3e-4，batch size 24，AMP。
- 任务：三分类情感极性和连续情感强度回归。
- 官方指标：Accuracy、Macro-F1、MAE、Pearson。

模型选择综合分数为：

```text
缺失验证 Macro-F1 + 缺失验证 Pearson - 0.25 × 缺失验证 MAE
```

## 结果汇总

| 实验 | 最佳轮次 | 停止轮次 | 测试 Accuracy | 测试 Macro-F1 | 测试 MAE | 测试 Pearson |
|---|---:|---:|---:|---:|---:|---:|
| bert_frozen_full | 5 | 13 | 0.6754 | 0.6078 | 0.6207 | 0.6792 |
| bert_frozen_stable | 4 | 16 | 0.6630 | 0.6335 | 0.6224 | 0.6980 |
| text_only | 7 | 19 | **0.6933** | 0.6420 | 0.6410 | 0.6763 |
| simple_multimodal | 5 | 17 | 0.6795 | **0.6437** | **0.6176** | 0.6895 |
| simple_clean | 4 | 14 | 0.6836 | 0.6367 | 0.6342 | 0.6715 |
| hierarchical_clean | 3 | 13 | 0.6713 | 0.5911 | 0.6609 | 0.6551 |
| anchor_hierarchical_clean | 4 | 14 | 0.6726 | 0.6030 | 0.6610 | 0.6686 |

第二次实验相对第一次实验：

- Macro-F1 提升 0.0257；
- Pearson 提升 0.0188；
- Accuracy 下降 0.0124；
- MAE 增加 0.0017。

第二次实验在类别均衡性和连续情感相关性上更好，因此建议将 `bert_frozen_stable` 作为后续消融与附件推理的主模型。

新增诊断实验表明，文本单模态取得最高三分类 Accuracy；简单多模态在 Macro-F1 和 MAE 上更均衡。分类优先训练仅将简单多模态 Accuracy 从 0.6795 提升到 0.6836。当前层次化分类使中性类召回率下降，未能改善总体 Accuracy。完整协议、混淆矩阵和结论见 [ACCURACY_EXPERIMENTS.md](ACCURACY_EXPERIMENTS.md)。

## bert_frozen_full

### 协议

- 每轮随机生成一套连续局部缺失验证样本；
- 早停耐心为 8 轮；
- 不使用学习率衰减；
- 第 13 轮触发早停。

### 最佳验证指标 第 5 轮

| 条件 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| 完整输入 | 0.6346 | 0.5911 | 0.5790 | 0.6697 |
| 随机缺失输入 | 0.6236 | 0.5760 | 0.6032 | 0.6100 |

原始结果：

- `results/bert_frozen_full/history.json`
- `results/bert_frozen_full/test_metrics.json`
- `results/bert_frozen_full/train.log`
- `results/bert_frozen_full/train.err.log`

## bert_frozen_stable

### 协议

- 固定 3 套连续局部缺失模式，每轮取三次验证的平均值；
- 固定种子为 3026、3027、3028；
- 早停耐心为 12 轮；
- 验证综合分数停滞 3 轮后将学习率乘以 0.5；
- 学习率从 3e-4 逐步下降到 3.75e-5；
- 第 16 轮触发早停。

### 最佳验证指标 第 4 轮

| 条件 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| 完整输入 | 0.6085 | 0.5963 | 0.6098 | 0.6415 |
| 固定缺失输入 | 0.6053 | 0.5831 | 0.6322 | 0.6020 |

原始结果：

- `results/bert_frozen_stable/history.json`
- `results/bert_frozen_stable/test_metrics.json`
- `results/bert_frozen_stable/train.log`
- `results/bert_frozen_stable/train.err.log`

## 限制与后续工作

- 两次实验均冻结 BERT，因此结果主要反映 TRAC-Net 上层结构的学习能力。
- 第一次实验的缺失验证模式每轮随机变化，其验证分数不能与第二次实验的固定三模式平均值直接横向比较。
- 当前结果尚未包含附件 3 无标签专项预测、附件 4 解释 CSV 或 Evidence Map 忠实度结果。
- 尚未进行多随机种子重复实验，因此没有报告均值、标准差或统计显著性。
- 检查点被有意排除；复现实验需自行准备竞赛数据和本地 BERT 权重。
- 新增实验仍只运行了随机种子 2026，模型间约 1 个百分点的差异需要多种子复核。
