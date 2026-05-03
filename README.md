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

## 最终结果（leak-free，与 ALIGNN 同一 1065 测试样本）

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
