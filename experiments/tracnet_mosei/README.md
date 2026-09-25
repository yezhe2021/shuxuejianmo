# TRAC-Net 多模态情感预测实验

TRAC-Net（Temporal Reliability-Aware Compensation Network）面向 CMU-MOSEI 对齐特征上的局部模态缺失场景，实现时序可靠度估计、局部跨模态补偿、完整—缺失双视图一致性训练，以及模态×时间 Evidence Map。

## 目录

- `tracnet/`：模型、数据接口、训练、推理、忠实度验证和自检脚本。
- `results/bert_frozen_full/`：首次冻结 BERT 训练的原始配置、日志和指标。
- `results/bert_frozen_stable/`：固定三套验证缺失模式并加入学习率衰减后的原始配置、日志和指标。
- `RESULTS.md`：累计实验协议、主要结果、比较和限制。

## 数据与模型

本实验使用竞赛提供的 `aligned_50.pkl`：训练、验证、测试样本数分别为 3395、728、727。文本输入使用本地 `bert-base-uncased`，BERT 参数在两次实验中均完全冻结。

仓库不包含：

- 原始或处理后的竞赛数据；
- BERT 权重；
- `best.pt`、`last.pt` 等训练检查点；
- PID、缓存和临时文件。

本地准备好竞赛数据与 BERT 后，具体训练、推理和解释性验证命令见 [tracnet/README.md](tracnet/README.md)。上传版配置仅将本机绝对 BERT 路径规范化为相对路径，其他超参数和原始结果保持不变。
