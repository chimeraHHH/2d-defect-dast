# 研究进度记录

> 项目目标: 基于 IMP2D 数据库构建二维材料缺陷形成能高通量预测平台
> 时间预算: 24 小时
> 当前状态: 实验运行中 + 论文撰写

## 已完成

- [x] **基线发现**: 克隆 GitHub 参考仓库 `wuleyan2004/defect_formation_energy_prediction`，
  解读其 CrystalTransformer 架构（局部 + 全连接全局）。
- [x] **数据获取**: 从 [DTU CMR](https://cmr.fysik.dtu.dk/_downloads/imp2d.db)
  下载 IMP2D (68 MB)；过滤后保留 10641 个收敛 + |Eform| < 20 eV 样本，
  与团队中期报告记录完全一致。
- [x] **管道实现**: 数据预处理 (PBC 邻居 + 三体)、9 维元素描述符（采用参考仓库的
  ref 表）、Dataset/collate、Normalizer、训练循环、AdamW + ReduceLROnPlateau。
- [x] **模型实现**:
  - `CrystalTransformer` (基线)：3 层 SchNet 风格局部 + 2 层带距离偏置的 dense
    Transformer。0.20 M 参数。
  - `DefectAwareTransformer` (DAST, 本文方案)：在基线之上加入虚拟缺陷锚点、
    星型稀疏注意力 mask、晶格自连接编码、缺陷边偏置。
- [x] **关键调试**:
  - 修复 MPS softmax+masked_fill(-inf) 导致的 NaN —— 改用 -1e9 + 每行至少自循环。
  - 修复 LocalInteractionLayer 收敛不动问题 —— 改用 SchNet 风格连续滤波卷积
    （`m_ij = phi(d_ij) ⊙ W h_j`）替代原来的 (h_i, h_j, RBF) 拼接 MLP；
    此修复使 n_global=2 与 DAST 都能稳定下降。
- [x] **基线训练动力学**:
  - Epoch 1: val MAE 2.44 (≈ constant predictor)
  - Epoch 5: val MAE 1.74 (开始描刻结构)
  - Epoch 9: val MAE 1.37 (持续下降中)

## 进行中

- [ ] **基线主实验** (epoch 9/30, val MAE 1.37 eV)
- [ ] **DAST + 三组消融**: 由 `scripts/run_queue.sh` 自动串联

## 待办

- [ ] 生成 parity / curves / error-dist 图表 (`scripts/make_figures.py`)
- [ ] 输出统一指标表 (`scripts/analyze_results.py`)
- [ ] 论文最终化（数字填入、作图整理、参考文献最终核对）
- [ ] 视实验结果决定是否启动数据增强对比 / 模型加宽 / 跨主体外推等额外
      子实验
- [ ] 打包交付（git tag / paper PDF）

## 决策记录

- 2026-05-02 14:25 — `LocalInteractionLayer` 的拼接式消息传递在完整数据集上
  长期停留在常量预测；切换到 SchNet 风格连续滤波卷积后训练曲线在 2 个 epoch
  内突破常量基线，验证此修复必要。
- 2026-05-02 14:30 — 切换到参考仓库 `atom_features.pth` (min-max 归一化)
  以消除我自建 z-score 表带来的训练发散；此为复现性提升而非创新点。
- 2026-05-02 14:35 — MPS 后端在 `softmax+masked_fill(-inf)` 与
  `scatter_reduce(amax)` 上有 NaN bug，固定为 CPU 训练；CPU 50-70 s/epoch
  在 8 核 M3 上对 ~0.2 M 参数模型完全可接受。
- 2026-05-02 14:48 — 启动主实验队列：基线 → 改进 → 三组消融，预计完成时间
  ~17:00 之前；同时 `scripts/run_queue.sh` 在 baseline metrics.json 出现后
  自动串联后续实验，避免人工编排开销。
