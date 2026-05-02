# 研究进度记录

> 项目目标: 基于 IMP2D 数据库构建二维材料缺陷形成能高通量预测平台
> 当前阶段: 主实验运行 + 论文撰写
> 时间预算: 24 小时

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
    Transformer。0.22 M 参数。
  - `DefectAwareTransformer` (DAST, 本文方案)：在基线之上加入虚拟缺陷锚点、
    星型稀疏注意力 mask、晶格自连接编码、缺陷边偏置。
- [x] **关键调试**:
  - 修复 MPS softmax+masked_fill(-inf) 导致的 NaN —— 改用 -1e9 + 每行至少自循环。
  - 修复 LocalInteractionLayer 收敛不动问题 —— 改用 SchNet 风格连续滤波卷积
    （`m_ij = phi(d_ij) ⊙ W h_j`）替代原来的 (h_i, h_j, RBF) 拼接 MLP；
    此修复使 n_global=2 与 DAST 都能稳定下降。

## 进行中

- [ ] **基线主实验** (running): bs=64, lr=1e-3, MSE, 30 epoch；预计 ~25 分钟。
- [ ] **DAST 主实验** (queued)
- [ ] **三组消融** (queued): Local-only、DAST -no virtual、DAST -no lattice。

## 待办

- [ ] 生成 parity / curves / error-dist 图表。
- [ ] 论文最终化（数字填入、作图、参考文献整理）。
- [ ] 打包交付。
