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
├── loho_summary.json                    # ⭐ LOHO 后处理 (待填)
├── summary.md                           # 自动生成的统一指标表
└── PROGRESS.md                          # 项目时间线与决策记录
paper/
├── main.md                              # 论文初稿 v1.2 (~610 行)
├── main.pdf                             # 中文 PDF
└── figures/
    ├── fig_parity / fig_curves / fig_error_dist (legacy)
    ├── fig_attention_heads / fig_attention_defect_centric (interp)
    ├── fig_occlusion_localisation / fig_interp_panel       (interp)
    ├── fig_uq_reliability / fig_uq_calibration             (UQ)
    ├── fig_error_by_category                               (decomposition)
    └── fig_loho_bars                                       (待填)
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
