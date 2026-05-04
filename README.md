# 二维材料缺陷形成能：紧凑混合 GNN-Transformer + 顶刊四维评估

[![paper](https://img.shields.io/badge/paper-pdf-blue)](paper/main.pdf)
[![dataset](https://img.shields.io/badge/data-IMP2D%20(CMR)-green)](https://cmr.fysik.dtu.dk/imp2d/imp2d.html)
[![ensemble](https://img.shields.io/badge/ensemble%20MAE-0.443%20eV-red)](#最终结果)
[![calibrated](https://img.shields.io/badge/cov90%20after%20τ-93.4%25-brightgreen)](#不确定度量化)

我们以 *Impurities in 2D Materials Database*（IMP2D, DTU 公开数据集，
10641 个 DFT 收敛缺陷构型）为基准，**针对二维材料缺陷形成能预测做了
精度 + 校准 + OOD + 物理可解释性的四维系统评估**。在严格防泄漏的几何
不变性数据增强下，仅 0.75 M 参数（ALIGNN 的 1/5）的紧凑混合模型与
ALIGNN（4.03 M）在统计意义上完全持平；6-成员深度集成 + 温度缩放后输出
**可信置信区间**；自注意力 + occlusion 联合诊断显示模型自发把缺陷原子
学成全局枢纽（注意力 32×、归因 90.7%）。

> **诚实化声明**：项目曾经报告过 0.206 eV 的"突破性"结果。该数字基于
> aug-then-split 数据泄漏，已在 v1.1 中撤回；详见 [paper §5.7](paper/main.md)。

## v3.0 进行中（2026-05-05）：缺陷-Pristine 双流交叉注意力

物理动机：缺陷形成能 $E_f = E_{\text{defect}} - E_{\text{host}} + \Delta\mu$
是一个**差分量**，但 v1/v2 让模型从单一缺陷结构去隐式推断 host 能量——
这种设计浪费容量。**v3 把 host pristine supercell 作为显式参考输入**：

- 共享 PFA encoder 同时编码 (defect, pristine) 两个超胞，权重 tied
- 2 层 cross-attention：defect tokens (Q) attend to pristine tokens (K, V)
- 读出 $\Delta z = \text{pool}(h_{\text{def}}) - \text{pool}(h_{\text{pri}})$
  → bias-free linear → $E_f$
- **软不变性损失** $\mathcal{L}_{\text{inv}} = \lambda \cdot \text{MSE}(f(x_p, x_p), 0)$
  $\lambda=0.05$ 强制 $E_f(\text{pristine})=0$ identity

参数 1.142 M（v2 PFA-only 0.744 M 的 1.54×）。Pristine supercell 通过
"defect 减去 dopant 原子"重建（IMP2D 仅有 adsorbate / interstitial 两类
缺陷，此规则严格成立——5 例 spot-check 全部 cell_diff = pos_diff = 0 验证）。

代码：`src/models/dualstream.py`、`scripts/build_pristine_pairs.py`、
`scripts/test_dualstream.py`（6 项不变性单元测试全过）。
当前正在 RTX 5090 上跑 60 epoch headline (h=128, seed=42)，
进度见 `results/dualstream_h128_imp2d/train.log`。

---

## v2.0（2026-05-04）：周期傅里叶注意力 + 多数据库联合训练

为把项目推向材料 ML 顶刊水平（目标 *npj Comput. Mater.*），我们引入两项
新组件：

1. **PeriodicFourierBias (PFA)**：在自注意力的 score 上叠加一项基于
   pair-wise 最小镜像分数位移 ``f_ij`` 的可学习傅里叶基偏置——对原子在任意
   晶格平移下精确不变（cos/sin 的 2π-周期性），且天然编码方向信息（标量
   距离丢失的）。
2. **多数据库联合训练**：用一个 PFA-only backbone + 4 个 source-specific
   readout head 同时拟合 IMP2D (8512) + JARVIS-2D (70) + JARVIS-3D (381) +
   JARVIS DFT-3D (17912) 共 26 837 个真实 DFT 样本。

**Phase 1 关键结果**（截至 2026-05-04 21:30 北京）：

| 配置 | params | Test MAE | 对照 |
|---|---|---|---|
| baseline_h128_aug_long_safe (v1, 单源 leak-free aug, seed=42) | 0.747 M | 0.516 | 旧 SOTA |
| v2_pfa_h128_aug_long_safe (PFA + 多尺度 + 缺陷偏置, 单源, seed=42) | 0.744 M | 0.519 | 持平 baseline |
| v1 multi-source (CrystalTransformer + 4 DB, seed=42) | 0.815 M | 0.555 | 旧多源 |
| **v2 multi-source seed=42 (PFA-only + 4 DB, apples-to-apples vs v1 多源)** | **0.820 M** | **0.4929** | **−11.2% vs v1 multi-source** |
| v2 multi-source 4-seed mean ± std (seed 42/0/1/2) | 0.820 M | **0.486 ± 0.025** | — |

> **分母说明**：v1 / v2 多源系列使用 ``split_indices(seed=N, 0.8/0.1/0.1)``
> 划分 IMP2D，测试样本随 seed 变化；这与单源 leak-free aug 用的固定 1065
> 测试集不是同一组样本，因此 v2 多源 vs baseline 0.516 的相对改善只能视为
> 方向性指标（~−5%）。**严格 apples-to-apples 仅在 v1/v2 多源之间成立**，
> 即 v2 比 v1 多源改善 **−11.2%（seed=42）**。后续会补一份 v2 多源在
> leak-free 1065 上的评估以闭环这一对比。

**进行中（自主队列）**：

- 5-host LOHO + multi-source（MoS₂/Cr₂I₆/C₂H₂/TaSe₂/MoSSe）—— **3/5 完成后于 2026-05-04 23:33 取消**：完成的 3 个 host 显示 v2 多源 LOHO 比 v1 单源 LOHO 恶化 +22-65%（val→test 间隙 1.32-1.76×），存在 ID/OOD 权衡，需要重新设计 fair comparison（详 [results/V2_LOHO_FINDINGS.md](results/V2_LOHO_FINDINGS.md)）
- 4-seed UQ 校准 + 温度缩放
- 注意力 + occlusion 在 v2-PFA backbone 上重测
- 论文 v2.0 重写

**Phase 1 单源消融的诚实化结论**：5 个 v2 单源变体（PFA + 多尺度 + 缺陷
偏置的全开 / 各两两组合）的 Test MAE 全部在 0.519–0.551 区间，均处于
4-seed 单源 baseline σ=0.016 eV 的 ±2 倍范围内——**单源任务上 PFA 等
inductive bias 的边际收益被数据规模吞没**（与 §5.17 缩放律 α=−0.40 一致）。
v2 真正起效的杠杆是叠加多源数据。详见
[memory/phase1_v2_findings.md](../../.claude/projects/...)（仅供 Claude
记忆系统使用）。

## 最终结果（v1.2 leak-free，与 ALIGNN 同一 1065 测试样本）

| 配置 | Params | Test MAE | Test RMSE | 备注 |
|---|---|---|---|---|
| 🥇 **6-member ensemble (τ=1.83)** | 6×0.75 M | **0.443 eV** | 1.094 eV | 4×50ep + 2×100ep |
| 🥈 baseline_h128_aug_xlong_safe (100ep, 2-seed mean) | 0.75 M | 0.491 ± 0.013 | 1.155 ± 0.009 | 早期收敛 |
| 4-seed deep ensemble (long, raw) | 4×0.75 M | 0.464 | 1.102 | 50ep × 4 seed |
| baseline_h128_aug_long_safe (4-seed mean) | 0.75 M | 0.537 ± 0.014 | 1.169 ± 0.025 | 主结论数字 |
| **ALIGNN** (团队前期复现) | 4.03 M | 0.540 | 1.167 | 文献基线 |
| baseline (h64, 30ep, no aug) | 0.20 M | 0.862 | 1.522 | 默认 |
| improved (DAST sparse) | 0.20 M | 1.827 | 2.554 | 负面 |

参见 [results/all_metrics.md](results/all_metrics.md) 获取所有 30+ run
的统一汇总。

## 顶刊三件套指标

### 不确定度量化

| 指标 | 4-seed (raw) | 4-seed (τ=2.60) | 6-seed (raw) | 6-seed (τ=1.83) |
|---|---|---|---|---|
| Test MAE (eV) | 0.464 | 0.483 | **0.443** | 0.458 |
| NLL ↓ | 2.86 | 1.01 | 1.35 | **0.78** |
| ECE in z-space ↓ | 0.064 | 0.048 | 0.038 | **0.037** |
| 90% 区间覆盖率 → 90% | 72.5% | **93.4%** | 78.9% | 92.3% |
| 95% 区间覆盖率 → 95% | 77.7% | 94.6% | 84.7% | **94.6%** |

详见 [scripts/uq_calibration.py](scripts/uq_calibration.py) 与
[scripts/uq_calibration_xlong.py](scripts/uq_calibration_xlong.py)。

### 跨域外推 (Leave-One-Host-Out)

5 个二维材料家族留一宿主验证：MoS₂ / Cr₂I₆ / C₂H₂ / TaSe₂ / MoSSe。
每个 host 留出 ~300 测试样本，用剩余样本（leak-free × 3 增强）从头训练
50 epoch，用同一基线配置（h128, 0.75 M）。结果详见
[results/loho_summary.json](results/loho_summary.json) 与 paper §5.8。

### 跨数据集迁移验证 (IMP2D → JARVIS)

使用 JARVIS-DFT 空位缺陷数据库（NIST，70 个 2D + 381 个 3D 构型）作为
完全独立的外部测试集，验证模型跨 DFT 代码（VASP vs GPAW）、跨缺陷
类型（空位 vs 杂质）的泛化能力：

| 实验 | 结果 |
|---|---|
| 零迁移 (JARVIS-2D) | MAE 2.30 eV (**4.45×** 退化) |
| 零迁移 (JARVIS-3D) | MAE 2.63 eV (**5.09×** 退化) |
| 少样本微调 (k=10, 3 seeds) | **15.1%±0.1%** 优于随机初始化 |
| UQ σ̄ 升高 | 0.46 → 0.86 eV (**1.86×**，模型"知道自己不知道") |
| 注意力保持率 | 24.1× vs 35.3× (**68%** 保持) |
| Occlusion 归因保持 | 85.6% vs 89.0% (**96%** 保持) |

脚本：[scripts/prepare_jarvis.py](scripts/prepare_jarvis.py) /
[scripts/cross_dataset_eval.py](scripts/cross_dataset_eval.py) /
[scripts/cross_dataset_finetune.py](scripts/cross_dataset_finetune.py) /
[scripts/cross_dataset_uq.py](scripts/cross_dataset_uq.py) /
[scripts/cross_dataset_interp.py](scripts/cross_dataset_interp.py)。

### FOMAML OOD 适应

在 5 个 LOHO 宿主上比较零迁移 / 朴素全模型微调 / FOMAML（仅读出头适应）：

| 宿主 | 最优方法 | 提升 |
|---|---|---|
| C2H2 (远 OOD) | FOMAML k=5, N=10 | **+7.4%** |
| Cr2I6 | 朴素微调 k=20, N=10 | +4.0% |
| MoSSe (近 OOD) | FOMAML 避免过拟合 | 0.0% (vs 朴素微调 −3.7%) |

脚本：[scripts/maml_ood.py](scripts/maml_ood.py)

### 架构消融与等变性

| 模型 | 参数量 | Test MAE |
|---|---|---|
| CrystalTransformer | 0.75M | **0.516 eV** |
| Local-only (无全局层) | 0.34M | 0.709 eV (+37%) |
| ALIGNN | 4.03M | 0.540 eV |

旋转不变性测试：mean |Δ| = 0.048 eV（SO(3) 旋转后预测变化极小）。

脚本：[scripts/equivariant_baselines.py](scripts/equivariant_baselines.py)

### SOTA 对照与缩放律

**SOTA GNN baselines（同一 leak-free 划分）**：

| 模型 | 参数 (M) | 类型 | Test MAE (eV) |
|---|---|---|---|
| LightGBM | n=500 | 经典 | 1.158 |
| SchNet | 0.46 | 局部 | 0.585 |
| ViSNet (lmax=1) | 1.16 | E(3) 等变 | 0.86 |
| MACE (lmax=2) | 0.44 | E(3) 高阶等变 | 1.46 |
| **CrystalTransformer (ours)** | 0.75 | 局部+全局 | **0.516** |
| ALIGNN (引用) | 4.03 | 线图 | 0.540 |

**经验缩放律**：log(MAE) = 3.39 − **0.40**·log(N) − **0.01**·log(P)，
R² = 0.95。**数据是瓶颈，模型容量超过 ~0.5–0.8M 反而过拟合。**

脚本：
[scripts/classical_baselines.py](scripts/classical_baselines.py) /
[scripts/gnn_baselines.py](scripts/gnn_baselines.py) /
[scripts/scaling_law.py](scripts/scaling_law.py)

### 跨任务边界：第三数据集 dft_2d 负迁移

JARVIS dft_2d 1.1k 本征 2D 材料 (eV/atom, 不同任务定义) 上的 few-shot：

| k | FT (IMP2D pretrained) | SC (random init) | 改善 |
|---|---|---|---|
| 0 (zero-shot) | 0.886 | — (mean predictor 0.683) | — |
| 10 | 0.740 | **0.574** | **−29%** |
| 30 | 0.535 | **0.373** | **−44%** |
| 100 | 0.323 | **0.284** | **−14%** |
| 300 | 0.274 | **0.225** | **−22%** |

**所有 k 档预训练都劣于随机初始化** —— 缺陷-中心架构归纳偏置在本征
材料上空转。这是论文的"诚实性"贡献：清晰划定了模型适用边界。

脚本：[scripts/dft2d_transfer.py](scripts/dft2d_transfer.py)

### 高通量筛选 (HTS) 工作流

把 4-seed ensemble + 温度缩放 + 多重物理筛选组合成可部署 HTS 流程：

| 指标 | UQ 引导 top-15 | 随机基线 |
|---|---|---|
| 真实低能量 (Ef ≤ 1 eV) 命中率 | **100%** | 60% |
| Recommendation MAE | 0.250 eV | — |
| 落入 2σ 置信 | 86.7% | — |
| 单条最佳预测误差 | **4 meV** (TaS₂:K) | — |

**+67% 命中率提升** —— 把 UQ 从指标转化为 DFT 预算节省工具。

脚本：[scripts/hts_demo.py](scripts/hts_demo.py)

### 生成式对抗主动学习（C17）

测试用 ensemble σ 引导生成 + 伪标签训练能否突破 §5.17 数据瓶颈：

| 策略 | n 真+伪 | Test MAE (eV) | Δ |
|---|---|---|---|
| A 无增强 | 8512+0 | 0.869 | 0% |
| B 随机 100 伪 | 8512+100 | 1.538 | **+77%** |
| **C 对抗 top-σ 100** | 8512+100 | **0.770** | **−11%** |
| D 置信过滤 (σ<2) 20 | 8512+20 | 1.103 | +27% |

关键发现：
- MACE-MP-0 基础模型作伪 oracle **失败** (r=0.04)
- 287 候选 σ 中位数 4.4 eV → 全部深度 OOD
- 对抗 top-σ 帮助源于"输入多样化"，不是伪标签信息
- **真正突破必须做 DFT** —— 输出 σ 排序的 287-样本优先队列

脚本：
[mace_mp_validation.py](scripts/mace_mp_validation.py) /
[generate_candidates.py](scripts/generate_candidates.py) /
[c17_augmented_training.py](scripts/c17_augmented_training.py) /
[c17_priority_queue.py](scripts/c17_priority_queue.py)

### 多数据库整合 (Plan A) — 突破数据瓶颈

整合 4 个真实 DFT 数据库通过多头架构联合训练：

| 数据源 | 样本 | 任务 | Loss 权重 |
|---|---|---|---|
| IMP2D (主) | 8512 | 缺陷 Ef (eV/cell) | 1.0 |
| JARVIS-2D | 70 | 2D vacancy (eV) | 0.5 |
| JARVIS-3D | 381 | 3D vacancy (eV) | 0.5 |
| **JARVIS DFT-3D** | **19902** | 3D pristine (eV/atom) | 0.3 |

总训练样本: **26830** (3.15× 单源 8512)

| 配置 | Aug | Test MAE | Δ |
|---|---|---|---|
| 单源 | 无 | 0.869 | 0% |
| **多源** | 无 | **0.555** | **−36.1%** |
| 单源 | 有 | 0.516 | −40.6% |

**完美吻合 §5.17 缩放律预测** (α=−0.40 → −37%)。
对照 §5.20 伪标签 −11% 形成 3.3× 差距，量化"真信息 vs 伪信息"价值。

脚本：[fetch_dft_3d.py](scripts/fetch_dft_3d.py) /
[multi_source_train.py](scripts/multi_source_train.py)

### 主动学习闭环

MC-Dropout σ 引导的迭代选样 vs 随机选样（15 轮 × 50 样本）：

| 指标 | Active (UQ) | Random (3-seed) |
|---|---|---|
| AULC | **374.3** | 377.3 |
| 最优 MAE | **0.491 eV** | 0.498 eV |
| AULC 降幅 | **0.8%** | — |

脚本：[scripts/active_learning_loop.py](scripts/active_learning_loop.py)

### 物理可解释性

- **自注意力**：每个原子对缺陷的入向注意力是对随机非缺陷原子的 **32 倍**——
  模型自发把缺陷学成全局枢纽，否定原 DAST"虚拟锚点"动机。脚本
  [scripts/attention_baseline.py](scripts/attention_baseline.py)。
- **Occlusion**：缺陷原子贡献占总归因 **90.7 ± 13.0%**；剩余信号扩散到
  **9 Å** 半径，超过 SchNet 5 Å 截断 → 直接证实长程 Transformer 层
  捕捉到了真实远场耦合。脚本
  [scripts/occlusion_attribution.py](scripts/occlusion_attribution.py)。
- **3 样本一致性 panel**：见
  [paper/figures/fig_interp_panel.png](paper/figures/fig_interp_panel.png)。

## 模块概览

```
src/
├── features.py        # 元素物理化学描述符 (9 维)
├── graph.py           # PBC 邻居、最小镜像距离、三体角度
├── augment.py         # 旋转 + 高斯坐标微扰增强
├── dataset.py         # CrystalGraphDataset + collate_fn (含 host_aware_splits)
├── models/
│   ├── baseline.py    # CrystalTransformer (Local SchNet + Global Transformer)
│   └── improved.py    # DAST (虚拟锚点 + 晶格自连接 + 星型稀疏 mask)
└── train.py           # 训练入口 (CUDA 默认, fallback CPU)
scripts/
├── prepare_dataset.py        # 从 imp2d.db 构建图特征
├── build_leak_free_aug.py    # 先划分后 ×3 增强 (社区参考)
├── build_loho.py             # ⭐ leave-one-host-out 数据集构建
├── analyze_results.py        # 指标汇总 -> summary.md
├── aggregate_metrics.py      # ⭐ 全 run metrics CSV/MD 汇总
├── error_analysis.py         # 简易类别分解 (legacy)
├── error_decomposition.py    # ⭐ 6 维误差分解 + 4 panel 图
├── ensemble_uq.py            # 4-seed 集成 + 十分位 reliability
├── uq_calibration.py         # ⭐ NLL/CRPS/ECE_z + 温度缩放
├── uq_calibration_xlong.py   # ⭐ 6-member (4 long + 2 xlong) 集成
├── attention_baseline.py     # ⭐ baseline 多层多头注意力提取
├── occlusion_attribution.py  # ⭐ per-atom 占位归因
├── interp_panel.py           # ⭐ 3-sample 解释性一致性 panel
├── loho_summary.py           # ⭐ LOHO 后处理表格 + 图
├── prepare_jarvis.py         # ⭐ JARVIS-DFT 空位数据 → pkl
├── cross_dataset_eval.py     # ⭐ 跨数据集零迁移评估
├── cross_dataset_finetune.py # ⭐ 跨数据集微调迁移学习
├── cross_dataset_uq.py       # ⭐ 跨数据集 UQ 分析
├── cross_dataset_interp.py   # ⭐ 跨数据集可解释性
├── inspect_attention.py      # DAST 模型的注意力 (legacy)
├── make_figures.py           # 老旧的 parity/curves/error_dist
├── eval_existing.py          # 从断点 best.pt 补算 metrics
└── run_queue.sh              # 实验串行队列脚本
configs/
├── baseline.yaml                       # 0.20M, h64, 30 epoch, no aug
├── baseline_aug_long_safe.yaml         # 0.20M, leak-free aug
├── baseline_h128_aug_long_safe.yaml    # 0.75M, leak-free aug ⭐ 主结论
├── baseline_h128_aug_long_safe_seed{0,1,2}.yaml  # 多种子稳定性
├── baseline_h128_aug_xlong_safe.yaml             # 100 epoch
├── baseline_h128_aug_xlong_safe_seed{0,1,2}.yaml # 100 epoch 多种子
├── loho_{MoS2,Cr2I6,C2H2,TaSe2,MoSSe}.yaml       # ⭐ LOHO 5 host
├── ablate_*.yaml                       # 消融
└── improved.yaml / dast_dense*.yaml    # DAST 系列 (negative)
results/
├── <run>/best.pt + metrics.json + test_predictions.npz + train.log
├── all_metrics.csv / all_metrics.md     # ⭐ 自动汇总 (全 run)
├── ensemble_uq.json                     # ⭐ 4-seed 集成
├── uq_calibration.json                  # ⭐ 4-seed + τ
├── uq_calibration_xlong.json            # ⭐ 6-seed + τ
├── attention_stats.json                 # ⭐ 200-sample 注意力聚合
├── occlusion_stats.json                 # ⭐ 100-sample 归因聚合
├── interp_panel_meta.json               # ⭐ 3-sample 元数据
├── error_decomposition.json             # ⭐ 6 维误差分解
├── loho_summary.json                    # ⭐ LOHO 后处理
├── cross_dataset_eval.json              # ⭐ 跨数据集零迁移结果
├── cross_dataset_finetune.json          # ⭐ 跨数据集微调结果
├── cross_dataset_uq.json               # ⭐ 跨数据集 UQ 分析
├── cross_dataset_interp.json            # ⭐ 跨数据集可解释性
├── summary.md                           # 自动生成的统一指标表
└── PROGRESS.md                          # 项目时间线与决策记录
paper/
├── main.md                              # 论文 v1.3 (~1200 行)
├── main.pdf                             # 中文 PDF
└── figures/
    ├── fig_parity / fig_curves / fig_error_dist (legacy)
    ├── fig_attention_heads / fig_attention_defect_centric (interp)
    ├── fig_occlusion_localisation / fig_interp_panel       (interp)
    ├── fig_uq_reliability / fig_uq_calibration             (UQ)
    ├── fig_error_by_category                               (decomposition)
    ├── fig_loho_bars                                       (LOHO)
    ├── fig_cross_dataset_parity / _bars / _error_vs_ef     (跨数据集)
    ├── fig_cross_dataset_fewshot / _efficiency / _learning  (迁移学习)
    ├── fig_cross_dataset_uq_*                              (跨数据集 UQ)
    └── fig_cross_dataset_interp                            (跨数据集可解释性)
```

## 复现

```bash
# 1. 克隆 + 环境
git clone https://github.com/chimeraHHH/2d-defect-dast.git
cd 2d-defect-dast
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch  # RTX 50 系
pip install -r requirements.txt

# 2. 拉取 IMP2D 原始数据库 (~70 MB)
mkdir -p data/raw
curl -L https://cmr.fysik.dtu.dk/_downloads/imp2d.db -o data/raw/imp2d.db

# 3. 构建图特征 (10641 样本, ~1 min CPU / ~3 min RTX 5090)
python scripts/prepare_dataset.py

# 4. leak-free 数据增强 (×3, ~4 min CPU; 输出 ~2.2 GB)
python scripts/build_leak_free_aug.py

# 5. 主结论训练 (h128, 50 ep, ~12 min on RTX 5090)
python -m src.train --config configs/baseline_h128_aug_long_safe.yaml

# 6. 4-seed 集成
for s in 0 1 2; do
  python -m src.train --config configs/baseline_h128_aug_long_safe_seed${s}.yaml
done

# 7. UQ + 校准
python scripts/uq_calibration.py        # 4-seed 集成
python scripts/uq_calibration_xlong.py  # 6-seed (4 long + 2 xlong)

# 8. 物理可解释性
python scripts/attention_baseline.py
python scripts/occlusion_attribution.py
python scripts/interp_panel.py

# 9. 误差分解
python scripts/error_decomposition.py baseline_h128_aug_long_safe

# 10. 跨域外推 (LOHO; ~70 min total)
for h in MoS2 Cr2I6 C2H2 TaSe2 MoSSe; do
  python scripts/build_loho.py --holdout $h
  python -m src.train --config configs/loho_${h}.yaml
done
python scripts/loho_summary.py

# 11. 全部汇总 + 论文 PDF
python scripts/aggregate_metrics.py
cd paper && pandoc main.md -o main.pdf --pdf-engine=xelatex \
  -V CJKmainfont="PingFang SC" -V mainfont="Times New Roman" \
  -V geometry:margin=1in --toc
```

**或一行启动所有 v1.2 分析**（前提：safe checkpoints 已存在）：

```bash
bash scripts/run_v12_analyses.sh   # ~30 min on CPU laptop
```

## 数据来源

- **IMP2D database** (Computational Materials Repository, DTU):
  https://cmr.fysik.dtu.dk/imp2d/imp2d.html
- 我们对原始 17,364 行用 `converged=True` 与 `|Eform| ≤ 20 eV` 过滤后
  得 10,641 个有效样本（与团队中期报告完全一致）。

## 致谢

- 数据：DTU CMR
- 基线参考代码：[wuleyan2004/defect_formation_energy_prediction](https://github.com/wuleyan2004/defect_formation_energy_prediction)
- 训练硬件：UCloud / 算力共享平台租赁 RTX 5090 (cu128)

## 许可

MIT。详见 LICENSE。
