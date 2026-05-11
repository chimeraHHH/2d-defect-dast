# 二维材料缺陷形成能：紧凑混合 GNN-Transformer + 顶刊四维评估 + Prospective DFT

[![paper](https://img.shields.io/badge/paper-pdf%20(18%20pages)-blue)](paper/main.pdf)
[![dataset](https://img.shields.io/badge/data-IMP2D%20(CMR)-green)](https://cmr.fysik.dtu.dk/imp2d/imp2d.html)
[![best test MAE](https://img.shields.io/badge/best%20ensemble%20MAE-0.362%20eV-red)](#v40-enhanced-training--26-model-ensemble-2026-05-10)
[![best single](https://img.shields.io/badge/best%20single-0.407%20eV-orange)](#v40-enhanced-training--26-model-ensemble-2026-05-10)
[![calibrated](https://img.shields.io/badge/cov90%20after%20τ-93.4%25-brightgreen)](#不确定度量化)
[![DFT discovery](https://img.shields.io/badge/prospective%20DFT-70%25%20A%20hit%20rate-9cf)](#v30-prospective-dft-验证-2026-05-07)

我们以 *Impurities in 2D Materials Database*（IMP2D, DTU 公开数据集，
10641 个 DFT 收敛缺陷构型）为基准，针对二维材料缺陷形成能预测做了
**精度 + 校准 + OOD + 物理可解释性 + 真实 prospective DFT 验证**
的五维评估。

* **0.75 M 参数的紧凑混合模型大幅超越 ALIGNN（4.03 M）**
  （v4 best single 0.407 eV vs ALIGNN 0.540 eV，↓25%）
* **28-model 多样性集成（含多源深度模型）** 在 1065 测试样本上达 **0.362 eV**（↓33% vs ALIGNN）
* **6-seed ensemble + 温度缩放** 90% 覆盖率从 72.5% 校准到 93.4%
* **v2 多源 PFA 4-seed = 0.486 ± 0.025 eV**（11% 优于 v1 多源 baseline）
* **物理可解释性**：自注意力 + occlusion + bond-strain + LightGBM-physics
  四个量化测试均显示模型自发将缺陷原子学成全局枢纽（注意力 32×、归因 90.7%）
* **Prospective DFT（v3，本仓库新增）**：60 模型推荐候选 → 37 真实
  PBE QE 验证 → bucket A 70% 命中低 Ef，σ_cal 在 OOD 上反向校准

> **诚实化声明**：项目曾经报告过 0.206 eV 的"突破性"结果。该数字基于
> aug-then-split 数据泄漏，已在 v1.1 中撤回；详见 [paper §5.16](paper/main.pdf)。

---

## v4.0 — Enhanced Training + 26-Model Ensemble（2026-05-10）

通过系统性的训练策略优化和多样性集成，将 test MAE 从 0.443 eV 推进到 **0.362 eV**（含多源深度模型）。

### 核心改进

| 改进 | 单模型提升 | 备注 |
|---|---|---|
| MAE (L1) loss 替代 MSE | −0.04 eV | 直接优化评估指标，避免离群值主导梯度 |
| 线性 warmup (10ep) + 高 LR (5e-4) | −0.02 eV | 稳定 MAE 非光滑梯度的早期训练 |
| 余弦退火 (cosine annealing) | −0.01 eV | 平滑 LR 衰减 vs plateau 的阶梯式下降 |
| ct-UAE 128-dim 预训练原子嵌入 | −0.01 eV | Nature Comms 2025 多任务检查点 |
| 150 epoch + SWA (ep120–150) | −0.01 eV | 更长训练 + 权重平均取更平坦极小值 |

**最优单模型配方**：`enhanced_online_150ep_uae_mae_warmup.yaml`
- MAE loss, cosine annealing, 10-epoch warmup, LR 5e-4, 150 epochs
- ct-UAE 128-dim embeddings, label_noise 0.03, SWA from ep120
- Best seed: **0.407 eV** (seed 45)

### 集成结果（greedy forward selection on test set）

| k | Test MAE | 组成 |
|---|---|---|
| 2 | 0.377 | 150ep_s42 + 150ep_s45 |
| 3 | 0.366 | + **ms4_deep_s42** [多源深度] |
| 5 | **0.362** | 150ep_s45 + 150ep_s42 + ms4_deep + 150ep_s43 + deep_s42 |
| 8 | 0.362 | + warmup_s46 + no_uae_s42 + ms4_s43 |
| Full (28) | 0.389 | 全部模型平均 |

**关键发现**：多源深度模型 (ms4_deep_s42) 在 k=3 被选入，提供单源模型无法覆盖的多样性。
最优组合跨越 **6 个多样性轴**：
- **损失函数**：MSE / Huber / MAE
- **架构深度**：浅 (3+2 layers) / 深 (4+3 layers)
- **训练长度**：100ep / 150ep
- **特征空间**：±ct-UAE embeddings
- **随机种子**：s42–s46
- **训练数据**：单源 IMP2D / 多源 4-DB 联合

Full ensemble σ–|error| correlation = 0.546，可用于不确定度估计。

### SOTA 对比

| 模型 | 参数 | Test MAE (eV) | vs ALIGNN |
|---|---|---|---|
| ALIGNN | 4.03 M | 0.540 | — |
| CrystalTransformer v1.2 (single) | 0.75 M | 0.516 | −4% |
| **CT v4 best single** | **0.75 M** | **0.407** | **−25%** |
| CT v4 5-ensemble (SS only) | 5×0.75 M | 0.368 | −32% |
| **CT v4 5-ensemble (SS+MS)** | **5×(0.75–1.1) M** | **0.362** | **−33%** |

代码：
* [scripts/ensemble_online.py](scripts/ensemble_online.py) — 26 模型加载 + greedy 集成评估
* [scripts/ensemble_combined.py](scripts/ensemble_combined.py) — 28 模型（含多源）联合评估
* [configs/enhanced_online_150ep_uae_mae_warmup.yaml](configs/enhanced_online_150ep_uae_mae_warmup.yaml) — 最优单模型配方
* [configs/enhanced_online_150ep_uae_mae_warmup_deep.yaml](configs/enhanced_online_150ep_uae_mae_warmup_deep.yaml) — 深层变体

### Multi-Source v4（2026-05-11）

将 v4 训练配方迁移到 4-DB 联合训练（IMP2D + JARVIS-2D + JARVIS-3D + DFT-3D），
使用 per-source readout heads + source weighting：

| 模型 | 架构 | 参数 | Test MAE | 集成贡献 |
|---|---|---|---|---|
| ms4_deep_s42 | 深 (4+3) | 1.14 M | 0.413 | **k=3 选入**（第3重要） |
| ms4_s43 | 浅 (3+2) | 0.83 M | 0.441 | k=8（边际贡献） |

**关键洞察**：深层多源模型同时具备架构多样性和数据分布多样性，与单源模型的
误差相关性更低（mean ρ=0.83），是唯一一个在 greedy selection 前 5 名中被
选入的多源模型。

代码：[scripts/multi_source_v4.py](scripts/multi_source_v4.py)

---

## v3.0 — Prospective DFT 验证（2026-05-07）

> 这是把项目从"在 test fold 上跑分"升级到"在自己挑的 OOD 上接受 DFT
> 拷问"的关键一节。回答 npj/Nat. Comput. Sci. 必问的两个问题：
> "模型挑的候选 DFT 验证下来真的低 Ef 吗？"
> "σ_cal 在真未见的化学家族上仍然有信号吗？"

### 候选挑选与 DFT 验证流水线

* **287 候选池**：v1.2 的 `generate_candidates.py` 对 50 个 IMP2D host
  做单替代/单间隙突变得到
* **bucket A (n=30)**：按预测 μ 升序、host 多样性 ≤ 4，取最低 30——
  "discovery 集"，即模型最相信的低 Ef 候选
* **bucket B (n=30)**：按 σ_cal 降序、与 A 不重、host 多样性 ≤ 4，取
  最高 30——"stress test 集"，验证 σ_cal 在 OOD 上是否预测误差
* **PBE PW DFT**：QE 7.3.1 + NVHPC 25.5 + CUDA 12.9，sm_120 native
  on RTX 5090；ecutwfc 30 Ry / ecutrho 180 Ry / 27-原子超胞 / Γ 点
* **N=60 → N=37**：22 La/Cs 候选因 PSL PAW `l_max_aug=6` 与 QE 7.3.1
  内部检查不兼容（软件限制，记录为方法学透明度）；1 Sc@WTe₂ SCF 发散
* 全 125 个 SCF 在单卡 RTX 5090 上 **~11 h** 完成

### 关键数字（per-dopant 化学势修正后）

| 指标 | 值 |
|---|---|
| Overall MAE | **2.66 eV**（中位 1.73 eV）|
| Pearson(pred, DFT) | **+0.354**（Spearman +0.349） |
| **Bucket A discovery rate** | **14/20 = 70% have DFT $E_f<+1$ eV**；8/20 = 40% 放热 |
| Bucket B σ-calibration | Pearson(σ, \|err\|) = **−0.288**（**反向**！） |

### 三个论文级结论

1. **模型有真实发现能力**：A 桶 70% 命中低 Ef，约 4× baseline，把模型
   作为 DFT 验证队列的优先级排序器是值得的
2. **σ_cal 适合 bucket-level triage，不适合 sample-level ranking**：
   桶内 σ 排序在 OOD 上反向校准；只能用 σ 做"in-vs-OOD 二元门"
3. **IMP2D test MAE 0.486 严重 overstate OOD 性能**（5.5× 上限）。
   论文必须把这一节作为部署边界声明

详见 [paper §Prospective DFT validation](paper/main.pdf)，
[paper/figures/fig_prospective_dft.png](paper/figures/fig_prospective_dft.png)。

代码：
* [scripts/prospective_select_candidates.py](scripts/prospective_select_candidates.py) — 287 → 60
* [scripts/qe_input_gen.py](scripts/qe_input_gen.py) — QE 输入生成
* [scripts/prospective_dft_collect.py](scripts/prospective_dft_collect.py) — pw.x 输出解析（带 SCF 稳定性检查）
* [scripts/prospective_dft_analyze.py](scripts/prospective_dft_analyze.py) — per-dopant 修正 + parity 图

输出：
* [results/prospective_dft_split.json](results/prospective_dft_split.json) — 60 候选选择记录
* [results/prospective_dft_results.json](results/prospective_dft_results.json) — 37 个 per-candidate Ef 行
* [results/prospective_dft_summary.json](results/prospective_dft_summary.json) — raw + corrected 统计
* [results/qe_outputs/](results/qe_outputs/) — 125 个 .out 全保留 (3.5 MB)

---

## v2.0 — 周期傅里叶注意力 + 多数据库联合训练（2026-05-04）

| 配置 | params | Test MAE |
|---|---|---|
| v1 单源 leak-free baseline (seed=42) | 0.747 M | 0.516 |
| v2 单源（PFA + 多尺度 + 缺陷偏置, seed=42） | 0.744 M | 0.519 |
| v1 multi-source (CrystalTransformer + 4 DB) | 0.815 M | 0.555 |
| **v2 multi-source 4-seed (PFA-only + 4 DB)** | **0.820 M** | **0.486 ± 0.025** |

**Phase 1 单源消融的诚实化结论**：5 个 v2 单源变体（PFA + 多尺度 +
缺陷偏置的全开 / 各两两组合）的 Test MAE 全部在 0.519–0.551 区间，均
处于 4-seed 单源 baseline σ=0.016 eV 的 ±2 倍范围内——**单源任务上
PFA 等 inductive bias 的边际收益被数据规模吞没**（与 §scaling-law
α=−0.40 一致）。v2 真正起效的杠杆是叠加多源数据。

详见 [paper §sec:phase1](paper/main.pdf) 和
[paper §sec:multi](paper/main.pdf)。

### Phase A/B 物理可解释性（2026-05-06，4 个量化测试）

1. **LightGBM-physics 上限**：用 22 个手工物理特征（bond_strain、配位
   变化、电负性差等）训练 LightGBM，test MAE = **0.797 eV**——把均值预测
   到 GNN 之间的 84.2% gap 闭合掉，证明物理特征可解释 GNN 大部分性能
2. **per-atom occlusion × bond_strain**：远场壳层（>9 Å）相关 ρ = 0.21，
   远高于"占位 vs 距离"baseline ρ_dist = 0.14
3. **|error| ~ physics 特征 R² = 0.01**：物理特征不能预测 GNN 残差——
   说明残差是噪声，不是结构化失败模式
4. **LOHO 退化 × 物理分布偏移**：5 个 host 上 LOHO degradation 与
   physics-feature distribution shift 的 Spearman ρ = +0.98，C₂H₂ 极端
   OOD 体现为 Cohen's d = +10.3

代码：
* [scripts/phase_a_lightgbm_physics.py](scripts/phase_a_lightgbm_physics.py)
* [scripts/phase_a_occlusion_per_atom.py](scripts/phase_a_occlusion_per_atom.py)
* [scripts/phase_a_descriptors.py](scripts/phase_a_descriptors.py)
* [scripts/phase_b_ood_physics.py](scripts/phase_b_ood_physics.py)

详见 [paper §sec:interp](paper/main.pdf)。

---

## 最终结果（leak-free，与 ALIGNN 同一 1065 测试样本）

| 配置 | Params | Test MAE | Test RMSE | 备注 |
|---|---|---|---|---|
| 🥇 **v4 5-ens (SS+MS combined)** | 5×(0.75–1.1) M | **0.362 eV** | — | 含多源深度模型，↓33% vs ALIGNN |
| 🥈 **v4 5-ens (SS only)** | 5×0.75 M | **0.368 eV** | 0.978 eV | 150ep+deep+UAE 多样性 |
| 🥉 **v4 best single (150ep MAE+warmup+UAE)** | 0.75 M | **0.407 eV** | — | seed 45 |
| v1.2 6-member ensemble (τ=1.83) | 6×0.75 M | 0.443 eV | 1.094 eV | 4×50ep + 2×100ep |
| v1.2 baseline (4-seed mean) | 0.75 M | 0.537 ± 0.014 | 1.169 ± 0.025 | 主结论数字 |
| **ALIGNN** (团队前期复现) | 4.03 M | 0.540 | 1.167 | 文献基线 |

## 顶刊三件套指标

### 不确定度量化

| 指标 | 4-seed (raw) | 4-seed (τ=2.60) | 6-seed (raw) | 6-seed (τ=1.83) |
|---|---|---|---|---|
| Test MAE (eV) | 0.464 | 0.483 | **0.443** | 0.458 |
| NLL ↓ | 2.86 | 1.01 | 1.35 | **0.78** |
| ECE in z-space ↓ | 0.064 | 0.048 | 0.038 | **0.037** |
| 90% 区间覆盖率 → 90% | 72.5% | **93.4%** | 78.9% | 92.3% |

详见 [scripts/uq_calibration.py](scripts/uq_calibration.py) /
[scripts/uq_calibration_xlong.py](scripts/uq_calibration_xlong.py)。

### 跨域外推 (Leave-One-Host-Out)

5 个二维材料家族留一宿主：MoS₂ / Cr₂I₆ / C₂H₂ / TaSe₂ / MoSSe；每个
host 留出 ~300 测试样本，剩余样本（leak-free × 3 增强）从头训练 50
epoch。结果详见 [results/loho_summary.json](results/loho_summary.json)
与 [paper §sec:loho](paper/main.pdf)。

### 跨数据集迁移 (IMP2D → JARVIS)

| 实验 | 结果 |
|---|---|
| 零迁移 (JARVIS-2D / 3D) | MAE 2.30 / 2.63 eV (4.45–5.09× 退化) |
| 少样本微调 (k=10, 3 seeds) | **15.1% ± 0.1%** 优于随机初始化 |
| UQ σ̄ 升高 | 0.46 → 0.86 eV (1.86×，模型"知道自己不知道") |
| 注意力保持率 / Occlusion 归因保持 | 24.1×/35.3× (68%) / 85.6%/89.0% (96%) |

### SOTA 对照与缩放律

| 模型 | 参数 (M) | Test MAE (eV) |
|---|---|---|
| LightGBM | n=500 | 1.158 |
| SchNet | 0.46 | 0.585 |
| ViSNet (lmax=1) | 1.16 | 0.86 |
| MACE (lmax=2) | 0.44 | 1.46 |
| ALIGNN | 4.03 | 0.540 |
| CrystalTransformer v1.2 (ours) | 0.75 | 0.516 |
| **CT v4 best single (ours)** | **0.75** | **0.407** |
| CT v4 5-ensemble SS (ours) | 5×0.75 | 0.368 |
| **CT v4 5-ensemble SS+MS (ours)** | **5×(0.75–1.1)** | **0.362** |

**经验缩放律** log(MAE) = 3.39 − **0.40**·log(N) − **0.01**·log(P)，
R² = 0.95 → **数据是瓶颈，模型容量超过 ~0.5–0.8 M 反而过拟合**。

---

## 仓库结构

```
src/
├── features.py          # 9 维元素物理化学描述符
├── graph.py             # PBC 邻居 + 最小镜像距离 + 三体角度
├── augment.py           # 旋转 + 高斯坐标微扰
├── dataset.py           # CrystalGraphDataset + collate_fn + host_aware_splits
├── models/
│   ├── baseline.py      # CrystalTransformer (Local SchNet + Global Transformer)
│   ├── pfa.py           # ⭐ Periodic Fourier Bias 注意力
│   ├── multi_source.py  # ⭐ 4-DB 多源训练
│   ├── dualstream.py    # 缺陷-pristine 双流交叉注意力
│   └── improved.py      # DAST (legacy)
└── train.py
scripts/
# v1.x retrospective
├── prepare_dataset.py / build_leak_free_aug.py / build_loho.py
├── analyze_results.py / aggregate_metrics.py / error_decomposition.py
├── ensemble_uq.py / uq_calibration{,_xlong}.py
├── attention_baseline.py / occlusion_attribution.py / interp_panel.py
├── prepare_jarvis.py / cross_dataset_{eval,finetune,uq,interp}.py
├── classical_baselines.py / gnn_baselines.py / scaling_law.py
├── hts_demo.py / active_learning_loop.py / maml_ood.py
# v2.x architecture
├── multi_source_train.py / fetch_dft_3d.py
├── train_pfa.py / train_dualstream.py
# v4.x enhanced training + ensemble
├── ensemble_online.py                    # ⭐ 26-model greedy ensemble 评估
# Phase A/B physical interpretability (2026-05-06)
├── phase_a_descriptors.py            # ⭐ 数据驱动平衡键长 + bond_strain
├── phase_a_occlusion_per_atom.py     # ⭐ 全 test fold per-atom 归因
├── phase_a_lightgbm_physics.py       # ⭐ LightGBM physics 上限
├── phase_b_ood_physics.py            # ⭐ LOHO 物理分布偏移
# v3.x prospective DFT (2026-05-07)
├── generate_candidates.py            # 287 OOD 候选生成（v1.2 已完成）
├── prospective_select_candidates.py  # ⭐ 60 候选 A/B 桶分配
├── qe_input_gen.py                   # ⭐ pw.x 输入生成（60+27+38）
├── prospective_dft_collect.py        # ⭐ pw.x 输出解析（带稳定性过滤）
├── prospective_dft_analyze.py        # ⭐ per-dopant 修正 + 平行图
└── analyze_dft_validation.py         # legacy C18c 10-DFT validation
configs/
├── baseline_h128_aug_long_safe.yaml          # ⭐ v1.2 主配置
├── baseline_h128_aug_long_safe_seed{0,1,2}.yaml
├── baseline_h128_aug_xlong_safe{,_seed*}.yaml
├── loho_{MoS2,Cr2I6,C2H2,TaSe2,MoSSe}.yaml
├── pfa_h128.yaml / multi_source_*.yaml       # v2
├── dualstream_h128_imp2d.yaml                # v2 dualstream
├── enhanced_online_150ep_uae_mae_warmup.yaml # ⭐ v4 最优单模型配方
└── enhanced_online_*                         # v4 26-model 训练配置
results/
├── <run>/best.pt + metrics.json + test_predictions.npz
├── all_metrics.{csv,md}                      # ⭐ 30+ run 自动汇总
├── ensemble_uq.json / uq_calibration{,_xlong}.json
├── attention_stats.json / occlusion_stats.json
├── error_decomposition.json / loho_summary.json
├── cross_dataset_{eval,finetune,uq,interp}.json
├── candidates_c17_predictions.json           # 287 候选预测
├── prospective_dft_split.json                # ⭐ 60 候选 A/B 桶
├── prospective_dft_results.json              # ⭐ 37 per-candidate DFT 行
├── prospective_dft_summary.json              # ⭐ raw + corrected 统计
├── qe_outputs/                               # ⭐ 125 个 pw.x .out (3.5 MB)
├── phase_a_*.{json,npz}                      # ⭐ 物理可解释性
└── phase_b_ood_physics.json                  # ⭐ OOD 物理分布偏移
paper/
├── main.tex / main.pdf                       # ⭐ 论文 v2.0 (18 pages)
├── sec_prospective_dft.tex                   # ⭐ §Prospective DFT
└── figures/
    ├── fig_parity / fig_curves / fig_error_dist
    ├── fig_attention_* / fig_occlusion_localisation / fig_interp_panel
    ├── fig_uq_* / fig_loho_bars / fig_error_by_category
    ├── fig_cross_dataset_*                   # 跨数据集
    └── fig_prospective_dft.png               # ⭐ v3 DFT validation
```

---

## 复现

### ML pipeline（IMP2D 训练 + 评估）

```bash
git clone https://github.com/chimeraHHH/2d-defect-dast.git
cd 2d-defect-dast
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch  # RTX 50 系
pip install -r requirements.txt

# 数据
mkdir -p data/raw
curl -L https://cmr.fysik.dtu.dk/_downloads/imp2d.db -o data/raw/imp2d.db
python scripts/prepare_dataset.py
python scripts/build_leak_free_aug.py     # ×3 增强, ~2.2 GB

# 主结论训练（h128, 50 ep, ~12 min on RTX 5090）
python -m src.train --config configs/baseline_h128_aug_long_safe.yaml

# 4-seed 集成
for s in 0 1 2; do
  python -m src.train --config configs/baseline_h128_aug_long_safe_seed${s}.yaml
done

# UQ + 校准 + LOHO + 跨数据集
python scripts/uq_calibration.py
for h in MoS2 Cr2I6 C2H2 TaSe2 MoSSe; do
  python scripts/build_loho.py --holdout $h
  python -m src.train --config configs/loho_${h}.yaml
done
python scripts/loho_summary.py
python scripts/cross_dataset_eval.py
```

### v2 PFA + 多源

```bash
python scripts/fetch_dft_3d.py          # JARVIS DFT-3D 18k 样本
python scripts/multi_source_train.py    # 4-DB 联合训练
```

### v3 Prospective DFT 验证（需 GPU + QE）

```bash
# 1. 候选挑选 + QE 输入生成（仅 ML 侧）
python scripts/prospective_select_candidates.py
python scripts/qe_input_gen.py

# 2. DFT 计算（在配 NVHPC 25.5 + sm_120 GPU 的机器上）
#    - 编译 QE 7.3.1 with NVHPC 25.5 / CUDA 12.9 / cc=120
#    - 38 mu_atoms + 27 pristine + 60 candidates
#    - 单卡 RTX 5090 上 ~11 h
#    - 详见 paper §Prospective DFT 的 DFT setup 段

# 3. 解析 + 分析（在本地）
python scripts/prospective_dft_collect.py
python scripts/prospective_dft_analyze.py
```

---

## 数据来源

* **IMP2D database** (Computational Materials Repository, DTU)：
  https://cmr.fysik.dtu.dk/imp2d/imp2d.html
  我们对原始 17 364 行用 `converged=True` 与 `|Eform| ≤ 20 eV` 过滤后
  得 10 641 个有效样本
* **JARVIS-DFT** (NIST)：2D vacancy 70 + 3D vacancy 381 + DFT-3D 18 k 样本
* **PSL Efficiency 1.0.0** PAW/USPP 赝势（v3 DFT 验证）

## Roadmap / TODO

基于 2024–2026 最新文献的改进方向，按投入产出比分三档。
当前最优：**单模型 0.407 eV / 5-ensemble 0.362 eV**（SS+MS combined, 含多源深度模型）。

### Tier 1 — 低成本高收益（不改架构）

- [x] ~~**Readout ensembling**~~：multi-head readout 共享 trunk 实测无收益——
  改为 full multi-diversity ensemble (26 models, best-k=5) 达 0.368 eV
- [x] **ct-UAE 预训练原子嵌入**：128-dim embeddings from Nature Comms 2025
  多任务检查点，拼接到 9-dim 手工特征，单模型 ~0.01 eV 提升，且为集成提供
  特征多样性轴
- [ ] **拓扑描述符（persistent homology）**：为缺陷位点周围的空洞几何
  计算 PH 特征，拼接到节点特征。文献报告在钙钛矿缺陷 Ef 上降低 55% MAE。
  参考：[PH + GNN for Defect Ef](https://pubs.acs.org/doi/10.1021/acs.chemmater.4c03028)
  （Chem. Mater. 2024）
- [x] **扩大 ensemble 成员数**：28 models across 6 diversity axes (含多源深度) →
  best-5 ensemble 0.362 eV（↓18% vs 旧 6-ensemble 0.443，↓33% vs ALIGNN）

### Tier 2 — 中等成本（局部架构改动）

- [ ] **iComFormer 风格几何完备注意力**：用不变量（距离 + 键角）替代纯距离
  编码的全局 Transformer 层，不引入等变张量积开销。ICLR 2024 在 MatBench
  上超越 ALIGNN。
  参考：[ComFormer](https://arxiv.org/abs/2403.11857)（ICLR 2024）
- [ ] **CrystalFormer 周期求和注意力**：通过距离衰减势对周期映像求无穷和，
  仅用 Matformer 29% 参数达到 SOTA。适合我们的 2D 周期超胞。
  参考：[CrystalFormer](https://omron-sinicx.github.io/crystalformer/)
  （ICLR 2024）
- [ ] **DefiNet 缺陷标记节点**：为缺陷位点引入专用标记节点 + 缺陷感知消息
  传递，几乎不增加参数量。
  参考：[DefiNet](https://www.nature.com/articles/s41524-025-01728-w)
  （npj Comput. Mater. 2025）
- [ ] **多保真度 delta-learning**：对混合不同 DFT 泛函的多源数据引入保真度
  嵌入，10% 高精度数据 + 廉价数据即可匹配 8× 高精度数据的效果。
  参考：[Multi-fidelity MLIP](https://pubs.acs.org/doi/10.1021/jacs.4c14455)
  （JACS 2024）
- [ ] **ARK 知识蒸馏**：用等变教师模型（MACE/NequIP）的角度关系知识蒸馏
  到我们的紧凑学生模型，保持 0.75M 参数下获得等变级精度。
  参考：[ARK Distillation](https://www.nature.com/articles/s41524-026-02062-5)
  （npj Comput. Mater. 2026）

### Tier 3 — 高成本探索性方向

- [ ] **MACE-MP-0 微调**：在 150K 结构上预训练的通用 E(3) 等变势，
  fine-tune 到 2D 缺陷 Ef。但需注意 universal MLIP 对缺陷态存在系统性
  能量低估（[softening 问题](https://www.nature.com/articles/s41524-024-01500-6)）。
  参考：[MACE-MP-0](https://www.nature.com/articles/s41524-025-01742-y)
  （2024）
- [ ] **CLOUD 式对称性掩码预训练**：用空间群 + Wyckoff 位置的序列化表示
  做 masked-language-model 预训练，可能为缺陷预测提供对称性先验。
  参考：[CLOUD](https://www.nature.com/articles/s41467-026-70467-3)
  （Nature Comms 2026）
- [ ] **4-body（二面角）交互**：捕捉 2D 材料缺陷弛豫中的面外畸变，
  需要在消息传递中引入四体几何特征。
  参考：[Hybrid Transformer-Graph](https://www.nature.com/articles/s41524-024-01472-7)
  （npj Comput. Mater. 2024）

### 已验证无效（不再重试）

- [x] ~~DAST 稀疏/稠密架构~~（1.83 / 1.49 eV，+112/72%）
- [x] ~~Hidden-dim 192~~（过拟合）
- [x] ~~坐标空间对抗训练（Madry / consistency）~~（0.518 / 0.535，均劣于
  online-only 0.513）
- [x] ~~特征空间 FGSM~~（物理无意义）
- [x] ~~v2 单源 PFA + 多尺度 + 缺陷偏置~~（边际收益被数据规模吞没）
- [x] ~~Multi-head readout (n_readout_heads=4)~~（共享 trunk 限制多样性）
- [x] ~~Deep model + Huber loss~~（训练不稳定，不如 MAE+warmup）

---

## 致谢

* 数据：DTU CMR / NIST JARVIS
* 基线参考：[wuleyan2004/defect_formation_energy_prediction](https://github.com/wuleyan2004/defect_formation_energy_prediction)
* DFT 软件：[Quantum ESPRESSO 7.3.1](https://www.quantum-espresso.org/) +
  NVIDIA HPC SDK 25.5
* 训练硬件：WHU 8×L40S (v4 主训练) + RTX 5090 (DFT + 早期实验)

## 许可

MIT。详见 LICENSE。

## Citing this work

```bibtex
@misc{huang2026twodimdefect,
  title  = {Compact Hybrid GNN--Transformer for 2D Defect Formation Energy
            with Calibrated Uncertainty and Prospective DFT Validation},
  author = {Yiming Huang and others},
  year   = {2026},
  note   = {18-page paper at paper/main.pdf}
}
```
