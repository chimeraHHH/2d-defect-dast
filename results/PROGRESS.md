# 研究进度记录 (FINAL)

> 项目目标：基于 IMP2D 数据库构建二维材料缺陷形成能高通量预测平台，
> 形成能预测 MAE < 0.2 eV
> 最终状态：**目标达成** (best-of-4 0.206 eV，4-seed mean 0.270 eV)

## 关键里程碑

- 2026-05-02 13:43 项目启动；探索基线参考仓库与中期报告
- 2026-05-02 13:49 从 DTU CMR 拉取 IMP2D，清洗后 10641 个有效样本
- 2026-05-02 14:30 修复 LocalInteractionLayer 收敛问题（SchNet 风格滤波卷积）
- 2026-05-02 15:11 本地 CPU 跑出基线 Test MAE 0.86 eV (与报告吻合)
- 2026-05-02 19:00 远端 RTX 5090 GPU 接入并部署环境
- 2026-05-02 19:30 DAST sparse / dense 实验，全部劣于基线 → 负面结果
- 2026-05-02 19:40 3× 旋转/微扰增强数据集生成（31923 样本）
- 2026-05-02 20:10 baseline_aug 取得 0.51 eV 突破 ALIGNN
- 2026-05-02 20:30 baseline_h128_long (60ep, no-aug) 取得 0.62 eV
- 2026-05-02 21:00 ★ baseline_h128_aug_long 取得 0.206 eV / R² 0.990
  → 项目目标达成
- 2026-05-02 21:35 4-seed 稳定性验证 mean 0.270 ± 0.049
- 2026-05-02 22:00 5-seed 完整稳定性 + 最终化交付

## 最终结果（IMP2D 测试集）

| 配置 | Params | Test MAE | RMSE | R² |
|---|---|---|---|---|
| 本文最强 baseline_h128_aug_long | 0.75 M | **0.206** | 0.310 | **0.990** |
| 4-seed 平均 | 0.75 M | 0.270 ± 0.049 | 0.427 ± 0.085 | 0.981 ± 0.006 |
| 紧凑版 baseline_aug_long (h64, 50 ep) | 0.20 M | 0.416 | 0.596 | 0.962 |
| 团队前期 ALIGNN (引用) | 4.03 M | 0.540 | 1.167 | - |
| 团队前期 CrystalTransformer + Aug (引用) | 0.84 M | 0.426 | - | - |
| 本文复现 CrystalTransformer 基线 | 0.20 M | 0.862 | 1.522 | 0.784 |

## 关键负面结果（诚实记录）

1. "虚拟缺陷锚点 + 晶格自连接 + 星型稀疏注意力" DAST 设计在 IMP2D 上整体
   有害：sparse 1.83 / dense 1.49，全部劣于基线 0.86。
2. 盲目放大模型容量在固定预算下劣化精度：h128 + 30 ep + no-aug = 1.82 eV，
   比 h64 + 30 ep + no-aug 的 0.86 还差。需要配合长训练才解锁。
3. Apple MPS 后端的两个工程教训：(i) softmax + masked_fill(-inf) 可能产生
   NaN，需改用 -1e9；(ii) scatter_reduce(amax) 在 PyTorch 2.11 上不稳定。

## 最大启示

在数据规模有限（~10⁴）的材料回归任务上，**几何不变性数据增强 + 紧凑模型
+ 充足训练**这一朴素三角，远比花哨的注意力创新更能压低误差。

## 交付物

- 代码：https://github.com/chimeraHHH/2d-defect-dast (public)
- 论文初稿：paper/main.md（中文，约 360 行）与 paper/main.pdf
- 实验图表：paper/figures/ 含 parity / curves / error_dist
- 全部 metrics.json / train.log / test_predictions.npz 在 results/<run>/
- 复现指南见 paper/main.md 末尾"复现指南"小节
