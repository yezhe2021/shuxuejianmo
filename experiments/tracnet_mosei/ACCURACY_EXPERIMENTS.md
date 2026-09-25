# 三分类准确率诊断与优化实验

## 实验目的

本组实验用于判断冻结 BERT 条件下，文本、简单多模态融合、分类优先损失和层次化分类结构对三分类情感识别的影响。所有实验使用同一份 `aligned_50.pkl`，划分为训练 3395 条、验证 728 条、测试 727 条，随机种子为 2026。

BERT 在所有实验中完全冻结，不包含 BERT 微调。仓库仅保存代码、配置、日志、逐轮历史和最终指标，不保存模型检查点。

## 模型与协议

### 诊断基线

- `text_only`：冻结 BERT，768 到 128 维投影，注意力时间池化，三分类与强度回归双头。
- `simple_multimodal`：文本、音频、视觉分别投影到 128 维；逐时间位置拼接三个模态和观测标志；一层 Transformer Encoder；注意力池化；三分类与强度回归双头。
- 两个模型均使用完整输入和人工连续局部缺失输入的双视图训练，并按固定缺失验证集综合分数选择模型。

### 准确率优先实验

- `simple_clean`：保持简单多模态结构，只使用完整输入训练；降低回归损失权重；按完整验证集 Accuracy 选择模型。
- `hierarchical_clean`：在简单多模态结构上，将三分类分解为中性检测和非中性极性判断。
- `anchor_hierarchical_clean`：以文本表示为锚点，音频和视觉通过门控残差修正文本表示，并使用层次化分类头。

准确率优先损失为：

```text
flat:         CE_3class + 0.2 × SmoothL1 + 0.05 × PearsonLoss
hierarchical: BCE_neutral + BCE_polarity + 0.3 × NLL_3class
              + 0.2 × SmoothL1 + 0.05 × PearsonLoss
```

## 测试结果

| 模型 | 最佳轮次 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|---:|
| text_only | 7 | **0.6933** | 0.6420 | 0.6410 | 0.6763 |
| simple_multimodal | 5 | 0.6795 | **0.6437** | **0.6176** | **0.6895** |
| simple_clean | 4 | 0.6836 | 0.6367 | 0.6342 | 0.6715 |
| hierarchical_clean | 3 | 0.6713 | 0.5911 | 0.6609 | 0.6551 |
| anchor_hierarchical_clean | 4 | 0.6726 | 0.6030 | 0.6610 | 0.6686 |

## 混淆矩阵

类别顺序为负面、中性、正面。

```text
simple_clean
[[134, 28, 45],
 [ 22, 66, 70],
 [ 25, 40,297]]

hierarchical_clean
[[152, 14, 41],
 [ 43, 37, 78],
 [ 41, 22,299]]

anchor_hierarchical_clean
[[143, 25, 39],
 [ 33, 45, 80],
 [ 33, 28,301]]
```

三个准确率优先模型的负面、中性、正面召回率分别为：

| 模型 | 负面召回率 | 中性召回率 | 正面召回率 |
|---|---:|---:|---:|
| simple_clean | 0.6473 | 0.4177 | 0.8204 |
| hierarchical_clean | 0.7343 | 0.2342 | 0.8260 |
| anchor_hierarchical_clean | 0.6908 | 0.2848 | 0.8315 |

## 结论

1. 文本单模态取得最高测试 Accuracy，说明当前融合结构尚未将音频和视觉转化为有效的分类增量。
2. 从双视图训练改为完整输入分类优先训练，仅将简单多模态 Accuracy 提高 0.0041，同时 Macro-F1 下降 0.0070。
3. 当前层次化分类结构明显漏判中性类，导致 Accuracy 和 Macro-F1 同时下降，不应作为下一阶段主结构。
4. 所有新增模型均在第 3 到 7 轮达到最佳验证结果，随后训练损失继续下降而验证指标下降，存在明显过拟合。
5. 结果来自单一随机种子，尚未报告均值、标准差或显著性检验。

## 原始结果位置

- `results/baseline_diagnostic_seed2026/`：诊断基线完整日志、配置、历史和测试指标。
- `results/accuracy_first_seed2026/`：准确率优先实验完整日志、配置、历史和测试指标。
- `tracnet/baselines.py`：诊断基线、层次化头和文本锚定模型。
- `tracnet/train_baseline.py`、`tracnet/run_baseline_suite.py`：诊断基线训练入口。
- `tracnet/train_accuracy.py`、`tracnet/run_accuracy_suite.py`：准确率优先实验训练入口。
