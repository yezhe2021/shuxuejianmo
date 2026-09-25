# TRAC-Net 实现

该目录实现时序可靠性感知补偿网络，用于题目中的局部模态缺失鲁棒预测与可解释性输出。

## 已实现的关键设计

- 分离 Padding Mask 与三模态 Observation Mask。
- 使用本地 `bert-base-uncased` 处理 `text_bert (3×50)`，全程可离线运行。
- 三模态独立浅层时序编码器。
- 局部窗口跨模态注意力补偿。
- 分离观测可靠度 `q_obs` 与补偿置信度 `q_comp`。
- 修正后的有效可靠度不会把缺失位置的补偿表示屏蔽掉。
- 完整视图到缺失视图的单向蒸馏、一致性约束和辅助潜表示重建。
- 单模态辅助预测头，减轻文本模态垄断。
- 输出 `模态×时间` Evidence Map、主要参考模态和关键位置。
- 使用固定的 3 套验证缺失模式取平均，消除每轮随机缺失造成的选模噪声。
- 验证分数停滞时自动降低学习率，默认早停耐心为 12 轮。

## 数据接口

默认使用对齐数据：

```text
E题数据/data/附件2-数据集特征文件/aligned_50.pkl
```

本地 BERT：

```text
E题数据/model
```

已核验该目录为 `bert-base-uncased`，隐藏维度 768、12 层、词表大小 30522，与数据中的 token ID 范围兼容。

## 快速自检

在项目根目录执行：

```powershell
D:\mysoft\python\python.exe -m tracnet.smoke_test
```

该测试不加载 BERT，主要验证补偿、融合、Evidence Map 和全部损失可以完成前向与反向传播。

## 训练

```powershell
D:\mysoft\python\python.exe -m tracnet.train `
  --data ".\E题数据\data\附件2-数据集特征文件\aligned_50.pkl" `
  --bert ".\E题数据\model" `
  --text-mode bert `
  --output-dir ".\tracnet\runs\bert_aligned" `
  --epochs 30 `
  --batch-size 24 `
  --amp
```

默认冻结 BERT，适合先稳定验证新结构。确认模型有效后，可使用：

```powershell
--unfreeze-bert-layers 2 --bert-learning-rate 2e-5
```

若要快速完成消融实验，可改用题目提供的连续文本特征：

```powershell
--text-mode features
```

## 附件 3 推理

```powershell
D:\mysoft\python\python.exe -m tracnet.infer `
  --checkpoint ".\tracnet\runs\bert_aligned\best.pt" `
  --input ".\E题数据\data\附件3-模态缺失特征样本\对齐版本" `
  --output ".\tracnet\runs\bert_aligned\attachment3_predictions.csv"
```

## 附件 4 预测与解释

```powershell
D:\mysoft\python\python.exe -m tracnet.infer `
  --checkpoint ".\tracnet\runs\bert_aligned\best.pt" `
  --input ".\E题数据\data\附件4-可解释专项视频样本与特征文件\附件4-可解释专项视频样本与特征文件\对齐版本" `
  --output ".\tracnet\runs\bert_aligned\attachment4_explanations.csv"
```

输出包括三分类概率、连续情感强度、主要模态、三模态重要度及 Top-k 模态—时间位置证据。

## Evidence Map 忠实度验证

对验证集执行 Top-k 删除、Top-k 保留和随机删除对照：

```powershell
D:\mysoft\python\python.exe -m tracnet.faithfulness `
  --checkpoint ".\tracnet\runs\bert_aligned\best.pt" `
  --data ".\E题数据\data\附件2-数据集特征文件\aligned_50.pkl" `
  --split valid `
  --ratio 0.1 `
  --output ".\tracnet\runs\bert_aligned\faithfulness.json"
```

若证据图有效，Top-k 删除导致的分类置信度下降和回归变化应显著大于同规模随机删除，同时只保留 Top-k 后应维持较高置信度。

## 建议实验顺序

1. `text-mode=features` 跑通完整训练并建立基线。
2. 切换本地 BERT，保持冻结，比较文本表示差异。
3. 分别移除补偿、可靠度、一致性蒸馏和单模态辅助头进行消融。
4. 在验证集构造不同模态、位置、长度和比例的连续缺失。
5. 对 Evidence Map 做 Top-k 删除、Top-k 保留和随机遮挡对照。
