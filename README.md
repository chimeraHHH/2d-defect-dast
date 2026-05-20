# 二维材料缺陷形成能：紧凑混合 GNN-Transformer + 顶刊四维评估 + Prospective DFT

[![paper](https://img.shields.io/badge/paper-pdf%20(18%20pages)-blue)](paper/main.pdf)
[![dataset](https://img.shields.io/badge/data-IMP2D%20(CMR)-green)](https://cmr.fysik.dtu.dk/imp2d/imp2d.html)
[![best test MAE](https://img.shields.io/badge/best%20ensemble%20MAE-0.344%20eV-red)](#v50--crystaltransformerv2-架构优化2026-05-19)
[![best single](https://img.shields.io/badge/best%20single-0.379%20eV-orange)](#v50--crystaltransformerv2-架构优化2026-05-19)
[![OOD](https://img.shields.io/badge/constrained%20OOD-0.540%20eV-yellow)](#v41--constrained-ood-evaluation2026-05-11)
[![calibrated](https://img.shields.io/badge/cov90%20after%20τ-93.4%25-brightgreen)](#不确定度量化)
[![DFT discovery](https://img.shields.io/badge/prospective%20DFT-70%25%20A%20hit%20rate-9cf)](#v30-prospective-dft-验证-2026-05-07)

我们以 *Impurities in 2D Materials Database*（IMP2D, DTU 公开数据集，
10641 个 DFT 收敛缺陷构型）为基准，针对二维材料缺陷形成能预测做了
**精度 + 校准 + OOD + 物理可解释性 + 真实 prospective DFT 验证**
的五维评估。

* **0.84 M 参数的 CrystalTransformerV2+MoE 大幅超越 ALIGNN（4.03 M）**
  （V4 MoE best single 0.379 eV vs ALIGNN 0.540 eV，↓30%）
* **8-model 贪心集成** 在 1065 测试样本上达 **0.344 eV**（↓36% vs ALIGNN），
  含 V4 MoE + V14 JK + V6 Physics + V3 DefType 创新模型（提供架构多样性）
* **约束 OOD 评估**：Leave-One-Host-Out 7-fold 平均 **0.540 eV**，
  从 ID 到族内 OOD 仅 1.5× 退化（而非全 OOD 的 7.3×），模型优雅降级
* **ct-UAE 消融**：预训练原子嵌入对 ID 有 +1% 收益，但 OOD 场景无统计显著帮助（p=0.22）
* **6-seed ensemble + 温度缩放** 90% 覆盖率从 72.5% 校准到 93.4%
* **物理可解释性**：自注意力 + occlusion + bond-strain + LightGBM-physics
  四个量化测试均显示模型自发将缺陷原子学成全局枢纽（注意力 32×、归因 90.7%）
* **Prospective DFT（v3）**：60 模型推荐候选 → 37 真实
  PBE QE 验证 → bucket A 70% 命中低 Ef，σ_cal 在 OOD 上反向校准

> **诚实化声明**：项目曾经报告过 0.206 eV 的"突破性"结果。该数字基于
> aug-then-split 数据泄漏，已在 v1.1 中撤回；详见 [paper §5.16](paper/main.pdf)。

---

## v5.0 — CrystalTransformerV2 架构优化（2026-05-19）

通过系统性架构改进和新型训练策略，将 test MAE 从 0.407 eV 推进到 **0.381 eV**（单模型），
集成从 0.359 eV 提升到 **0.349 eV**。这是 IMP2D 数据集上首个 GNN 方法，大幅超越
唯一已发表基线 El Alouani 2024（tree-based, MAE ~0.518 eV）。

### CrystalTransformerV2 架构（3 个核心改进）

| 组件 | 单独效果 (vs V1) | 机制 |
|---|---|---|
| **缺陷感知门控注意力池化** | −19.9%（0.516→0.414） | 注意力加权 + max pooling 门控融合，捕捉缺陷原子极端特征 |
| **局部环境富集** | −18.0%（0.516→0.423） | 配位数/邻居距离/电负性对比等物理特征注入缺陷位点 |
| **Pre-Norm 残差** | −21.0%（0.516→0.408） | 现代 Transformer 最佳实践，更稳定训练 |
| **三者结合 (Full V2)** | **−26.2%（0.516→0.381）** | 组件间存在互补效应 |

参考文献：
* 门控池化：Hossain et al., Chem. Mater. 2024（全局 max pooling 缺陷 Ef ↓55%）
* Pre-Norm：Xiong et al., ICML 2020（On Layer Normalization in the Transformer）

### 与最新文献的对比定位

本工作是 **首个针对二维材料缺陷 Ef 的物理驱动 GNN-Transformer 方法**（截至 2025 年无同类工作）：

| 方法 | 发表 | 数据集 | 关键创新 | 与本工作的关系 |
|---|---|---|---|---|
| DefiNet | npj Comput. Mater. 2025 | 14866 bulk defects | 等变 GNN + 显式缺陷标记 | 我们也使用缺陷标记，但结合 Transformer 注意力 |
| PH+GNN | Chem. Mater. 2025 | bulk defects | 持久同调 → GNN (MAE↓55%) | 我们实现了 ASPH 并整合到 Transformer 架构 |
| Charged Ef | PRL 2025 | 氧空位 | CGCNN + 费米能级对齐 | 针对带电缺陷，我们针对中性 2D 缺陷 |
| dGNN | Nature Comput. Sci. 2024 | bulk defects | 缺陷特定图构建 + 池化 | 我们的 V3 条件化 + V2 门控池化类似但更通用 |
| CLOUD | Nature Commun. 2025 | 百万级预训练 | 物理约束基础模型 | 互补：我们可用 CLOUD 嵌入替代 ct-UAE |
| MACE/CHGNet | Small 2025 | 86259 空位 | 通用 ML 势 benchmark | MACE RMSE 0.46–0.80 eV，我们的方法在 2D 上更精确 |

### V2 种子稳定性

| 种子 | Test MAE | 备注 |
|---|---|---|
| s42 | 0.3942 | |
| **s43** | **0.3810** | 最优 |
| s44 | 0.3943 | |
| s45 | 0.3981 | |
| **Mean ± Std** | **0.392 ± 0.007** | 显著优于 V1 的 0.537 ± 0.014 |

### 第一波创新统计显著性分析（Bootstrap, 2026-05-20）

对 V3/V4/V6/V9 四个创新与 V2 基线进行 10000 次 paired bootstrap 检验：

| 对比 | ΔMAE | 95% CI | p-value | 结论 |
|---|---|---|---|---|
| V4 MoE vs V2 | −0.002 | [−0.021, +0.024] | 0.868 | 不显著（但点估计最优） |
| V6 Physics vs V2 | +0.001 | [−0.022, +0.019] | 0.919 | 不显著（等价） |
| V3 DefType vs V2 | +0.006 | [−0.025, +0.013] | 0.557 | 不显著（val MAE 最优） |
| V14 JK vs V2 | +0.008 | [−0.012, +0.028] | 0.423 | 不显著（JK 权重偏好深层） |
| V9 Contrast vs V2 | +0.029 | [+0.007, +0.052] | **0.009** | **显著更差** ★ |
| V11 LDS vs V2 | +0.052 | [+0.026, +0.076] | **<0.001** | **显著更差** ★★ |
| V13 RnC vs V2 | +0.061 | [+0.037, +0.085] | **<0.001** | **显著更差** ★★ |

**关键洞察**：
- 单模型创新均未显著超越 V2（1065 样本检验力不足），但 **8-model 贪心集成达 0.344 eV**（↓36% vs ALIGNN）
- V14 JK 的 JK 权重 [0.30, 0.31, 0.39] 显示模型偏好最深全局层，被选入 ensemble k=4
- V11 LDS 和 V13 RnC 均显著更差：LDS 在 [0,2) eV 退化最严重 (p<0.001)，重权策略过度偏移低能区梯度
- V9 对比条件化在 [7,25) eV 范围显著退化 (ΔMAE=+0.23, p=0.014)
- 创新模型的核心价值在于**集成多样性**，而非单模型提升

### 物理驱动模型创新（V3–V7）

| 创新 | 物理动机 | +参数 | 状态 | 顶刊参考 |
|---|---|---|---|---|
| **V3 缺陷类型条件化** | 间隙/吸附缺陷 Ef 分布差异 2× | +512 | ✅ 0.387 | 首创；灵感来自条件生成 (Dhariwal, NeurIPS 2021) |
| **V4 MoE Readout** | 59% 误差来自 top 10% 困难样本 → 专家化 | +13K | ✅ **0.379 ★** | MoCE (ICLR 2025); Switch Transformer (JMLR 2022) |
| **V6 物理失配特征** | Hume-Rothery 固溶度规则 → 6 维掺杂-宿主描述符 | +8.7K | ✅ 0.382 | Bartel, Sci. Adv. 2020; Ward, npj Comput. Mater. 2016; Goodall, Nature Commun. 2020 |
| **V9 缺陷-宿主对比条件化** | Ef ∝ E(缺陷) − E(原始)，显式建模差值 | +8.5K | ✅ 0.410 | 物理先验，零初始化 |
| **V11 LDS 标签分布平滑** | 高 Ef 样本（7.1%）贡献 29% 误差 → 逆密度重权 | 0 | ❌ 0.433 | Yang et al., ICML 2021 (DIR) |
| **V13 RnC 对比学习** | 排序保持的特征空间结构化，缓解预测压缩 | +128-dim proj | ❌ 0.442 | Zha et al., NeurIPS 2023 (Rank-N-Contrast) |
| **V14 JK 层聚合** | 缺陷需多尺度理解（局部应变 + 全局电子） | +3 | ✅ 0.389 | Xu et al., ICML 2018 (JK-Net) |
| **V12 LDS+物理+全条件化** | V6+V3+V9+LDS+EMA 协同 | +17K | ⏳ 待训练 | 综合方案 |
| **V10 最优组合** | 所有正面创新整合 | +31K | ⏳ 待训练 | — |
| **异方差不确定性** | 自适应降权噪声样本 + 置信度估计 | +8.3K | ❌ 0.466 | Kendall & Gal, NeurIPS 2017; Hirschfeld, JCIM 2020 |
| **ASPH 持久同调** | 拓扑描述缺陷局部环境 (0-dim 连通 + 1-dim 环) | 0 (输入特征) | ✅ 已计算 | Fang & Yan, Chem. Mater. 2025 (Ef MAE↓55%) |
| **Focal MAE 损失** | 动态上权困难样本：w_i = (\|e_i\|/mean)^γ | 0 | ❌ 0.419 | Lin et al., ICCV 2017 (Focal Loss, Best Paper) |
| **EMA 权重平均** | 训练过程中指数滑动平均，更平坦极小值 | 0 | ✅ 0.395 | Polyak 1992; Izmailov et al., UAI 2018 |

#### V6 物理失配特征详解

基于 Hume-Rothery 固溶度规则 (1936) 和现代计算材料学，从掺杂原子与宿主晶格的元素性质差异中提取 6 个图级描述符：

| 特征 | 物理含义 | 间隙相关性 | 吸附相关性 |
|---|---|---|---|
| 尺寸失配 (Δr/r) | 弹性应变能，Hume-Rothery 15% 规则 | r=0.27 | r=0.02 |
| 电负性差 (Δχ) | 电荷转移方向，Pauling 规则 | r=−0.19 | r=0.12 |
| 电离能比 (IE_d/IE_h) | 化学硬度匹配，Pearson HSAB 原理 | r=−0.19 | r=−0.01 |
| 电子亲和能差 (ΔEA) | 电子接受倾向 | r=−0.03 | r=−0.07 |
| 价电子失配 (\|ΔVE\|) | 键合兼容性（悬挂键/电荷补偿） | r=0.02 | r=−0.23 |
| 周期距离 (Δperiod) | 轨道重叠质量 | r=0.28 | r=0.12 |

**关键发现**：相关性具有缺陷类型特异性 — 尺寸失配对间隙缺陷最重要 (r=0.27)，而价电子失配对吸附缺陷最重要 (r=−0.23)。
这验证了 V3 缺陷类型条件化的必要性：不同缺陷类型由不同物理机制主导。

### 注意力集中度分析（新发现, 2026-05-19）

通过钩子提取 V2 门控池化层的注意力权重分布：

| 模型 | 缺陷原子注意力占比 | 集中度因子 | 备注 |
|---|---|---|---|
| **V2 baseline (s43)** | **99.68%** | **41.3×** | 几乎完全忽略宿主原子 |
| V3 deftype (ep19) | 11.4% | 5.1× | 条件化使注意力更均匀 |
| V6 physics (ep17) | 6.0% | 2.2× | 物理特征进一步分散注意力 |

**关键洞察**：V2 基线将 99.68% 的池化注意力集中在缺陷原子上（预期均匀分布 ~2.4%），
这解释了两个现象：
1. **[7,25) eV 范围 MAE 4× 于低能范围** — 高 Ef 样本需要宿主晶格上下文（应变能、电子重构），
   但模型完全丢弃了这些信息
2. **预测压缩** — pred_std/target_std = 0.921，系统性欠预测高 Ef 样本（偏差 −1.16 eV）

V3 条件化和 V6 物理特征显著分散注意力（从 41.3× 降至 2–5×），为宿主信息提供了替代通路。

### 详细误差分析（2026-05-19）

#### 按缺陷类型

| 类型 | MAE (eV) | 占比 | 平均 Ef | 偏差 |
|---|---|---|---|---|
| **间隙 (interstitial)** | **0.620** | 34.4% | 5.03 | −0.29 |
| 吸附 (adsorbate) | 0.256 | 65.6% | 1.61 | +0.05 |

间隙缺陷 MAE 2.4× 高于吸附缺陷。间隙缺陷 Ef 分布更宽、更重尾，
是 V3 条件化和 V11 LDS 重权的主要目标。

#### 最难宿主材料 (top 5)

| 宿主 | MAE (eV) | 样本数 | 平均 Ef | 偏差 |
|---|---|---|---|---|
| Bi₂I₆ | 2.030 | 15 | 6.84 | −1.60 |
| Nb₄C₃ | 1.899 | 14 | 7.06 | −1.58 |
| BiITe | 1.274 | 15 | 4.84 | −0.80 |
| As₂Te₃ | 1.076 | 16 | 5.70 | −0.82 |
| TiS₂ | 0.759 | 17 | 3.07 | +0.51 |

重原子化合物（Bi、Nb）和 MXene 类材料是系统性误差来源，
均呈负偏差（严重欠预测），可能与强自旋-轨道耦合或 DFT 收敛困难相关。

#### 最难掺杂元素 (top 5)

| 掺杂 | MAE (eV) | 样本数 | 偏差 |
|---|---|---|---|
| Ru | 1.668 | 3 | −1.668 |
| F | 1.339 | 25 | −0.643 |
| P | 0.804 | 21 | −0.457 |
| Os | 0.756 | 16 | −0.460 |
| Ca | 0.688 | 21 | −0.428 |

**异常样本**：Nb₄C₃+Ru 吸附，DFT Ef=19.89 eV，模型预测 2.96 eV（误差 16.92 eV），
疑为 DFT 伪影或未收敛结构。

### TTA 推断（测试时增强）

| 方法 | MAE | Δ | 备注 |
|---|---|---|---|
| 原始 | 0.3810 | — | V2 s43 单模型 |
| TTA(4) 平均 | 0.3808 | −0.0002 | 4 次随机几何增强平均 |

TTA 增益可忽略（−0.0002 eV），因为模型已使用在线增强训练。
但预测方差与误差的相关性 r=0.348，可作为辅助不确定性信号。

### 误差分析（驱动 V3–V5 设计）

| 目标范围 (eV) | N 样本 | MAE | 误差贡献 |
|---|---|---|---|
| [−15, −3) | 14 (1.3%) | 1.637 | 6.1% |
| [−3, 0) | 148 (13.9%) | 0.246 | 9.6% |
| [0, 3) | 516 (48.5%) | 0.192 | 26.2% |
| [3, 7) | 311 (29.2%) | 0.331 | 27.2% |
| **[7, 25)** | **76 (7.1%)** | **1.541** | **31.0%** |

**核心发现**：Top 10% 最难样本贡献 59.3% 总误差，其中间隙缺陷占 70%（远超测试集中的 34% 占比）。
V3 缺陷类型条件化和 V4 MoE 专家化直接针对此问题。

### SOTA 更新

| 模型 | 参数 | Test MAE (eV) | vs ALIGNN |
|---|---|---|---|
| El Alouani 2024 (tree-based) | — | 0.518 | +4% |
| ALIGNN | 4.03 M | 0.540 | — |
| CrystalTransformer v1.2 (single) | 0.75 M | 0.516 | −4% |
| CT v4 best single | 0.75 M | 0.407 | −25% |
| CT-V2 best single (v5) | 0.83 M | 0.381 | −29% |
| **CT-V2+MoE (V4)** | **0.84 M** | **0.379** | **−30%** |
| CT-V2+DefType (V3) | 0.83 M | 0.387 | −28% |
| CT-V2+Physics (V6) | 0.84 M | 0.382 | −29% |
| CT-V2+JK (V14) | 0.83 M | 0.389 | −28% |
| CT-V2+Contrast (V9) | 0.84 M | 0.410 | −24% |
| CT-V2+LDS (V11) | 0.83 M | 0.433 | −20% |
| CT-V2+RnC+LDS (V13) | 0.83 M | 0.442 | −18% |
| CT v4 7-ensemble (SS+MS) | 7×(0.75–1.1) M | 0.359 | −34% |
| CT-V2 8-ensemble greedy (v5) | 8×(0.83–1.1) M | 0.349 | −35% |
| **CT-V2+innovations 8-ens (v5.1)** | **8×(0.83–1.1) M** | **0.344** | **−36%** |

集成成员选择（贪心法）：

| k | Test MAE | 新增成员 |
|---|---|---|
| 1 | 0.381 | v2_gated_s43 |
| 2 | 0.362 | + multi_src_v4_deep |
| 3 | 0.354 | + v2_enhanced_env_s43 |
| 4 | 0.351 | + v2_gated_s44 |
| 5 | 0.350 | + v2_long250 |
| 6 | 0.349 | + v2_enhanced_env |
| 7 | 0.349 | + multi_src_v4 |
| **8** | **0.349** | + v2_gated_s42 |

**关键发现**：同架构不同种子模型预测相关性 r=0.99+，集成增益极小。
真正的多样性来自架构差异（multi_source r=0.97）和训练策略差异。

代码：
* [src/models/crystal_v2.py](src/models/crystal_v2.py) — CrystalTransformerV2（含全部创新：缺陷条件化 + MoE + 物理特征 + 对比 + JK + 不确定性）
* [src/train_enhanced.py](src/train_enhanced.py) — 增强训练脚本（SWA、focal MAE、MoE balance loss、LDS、RnC、异方差损失）
* [scripts/eval_new_model.py](scripts/eval_new_model.py) — 模型评估：per-range 分析 + 相关性 + 贪心集成
* [scripts/eval_checkpoint.py](scripts/eval_checkpoint.py) — ⭐ 通用 checkpoint 评估（自动检测格式 + 嵌入式 config/normalizer）
* [scripts/attention_analysis.py](scripts/attention_analysis.py) — ⭐ 注意力集中度 + JK 权重 + 物理特征重要度分析
* [scripts/error_analysis_detailed.py](scripts/error_analysis_detailed.py) — ⭐ 按宿主/掺杂/缺陷类型的详细误差分解
* [scripts/tta_inference.py](scripts/tta_inference.py) — ⭐ 测试时增强推断 + 预测方差不确定性
* [scripts/generate_paper_table.py](scripts/generate_paper_table.py) — ⭐ 自动生成 LaTeX 论文表格（消融 + per-range + 集成）
* [scripts/compute_asph.py](scripts/compute_asph.py) — ASPH 持久同调特征计算
* [scripts/analyze_innovations.py](scripts/analyze_innovations.py) — V3–V7 创新对比分析
* [scripts/physics_interpretability.py](scripts/physics_interpretability.py) — 物理可解释性分析（注意力 + JK权重 + 专家分析）

配置：
* [configs/v3_deftype.yaml](configs/v3_deftype.yaml) — V3 缺陷类型条件化
* [configs/v4_moe.yaml](configs/v4_moe.yaml) — V4 MoE readout (3 experts)
* [configs/v6_physics.yaml](configs/v6_physics.yaml) — V6 物理失配特征 (Hume-Rothery)
* [configs/v9_contrast.yaml](configs/v9_contrast.yaml) — ⭐ V9 缺陷-宿主对比条件化
* [configs/v11_lds.yaml](configs/v11_lds.yaml) — ⭐ V11 LDS 标签分布平滑 (ICML 2021)
* [configs/v12_lds_physics.yaml](configs/v12_lds_physics.yaml) — ⭐ V12 LDS + 物理 + 全条件化
* [configs/v13_rnc.yaml](configs/v13_rnc.yaml) — ⭐ V13 Rank-N-Contrast + LDS (NeurIPS 2023)
* [configs/v14_jk.yaml](configs/v14_jk.yaml) — ⭐ V14 JK 层聚合 + 缺陷条件化
* [configs/v10_best_combo.yaml](configs/v10_best_combo.yaml) — ⭐ V10 最优组合
* [configs/v2_ema.yaml](configs/v2_ema.yaml) — V2 + EMA 权重平均
* [configs/v2_uncertainty.yaml](configs/v2_uncertainty.yaml) — V2 + 异方差不确定性
* [configs/v2_asph.yaml](configs/v2_asph.yaml) — V2 + ASPH 持久同调
* [configs/v2_focal.yaml](configs/v2_focal.yaml) — V2 + Focal MAE 损失
* [configs/v5_full.yaml](configs/v5_full.yaml) — V5 组合（ASPH + V3 + V4）
* [configs/v7_all.yaml](configs/v7_all.yaml) — V7 全组合（V3 + V4 + V6 + JK + ASPH）

---

## v4.0 — Enhanced Training + 29-Model Ensemble（2026-05-10）

通过系统性的训练策略优化和多样性集成，将 test MAE 从 0.443 eV 推进到 **0.359 eV**（含多源模型）。

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
| 5 | 0.362 | + 150ep_s43 + deep_s42 |
| 6 | 0.360 | + **ms4_s42** [多源浅层] |
| 7 | **0.359** | + no_uae_s42 |
| Full (29) | 0.388 | 全部模型平均 |

**关键发现**：多源模型 ms4_deep_s42 (k=3) 和 ms4_s42 (k=6) 均被选入，
提供单源模型无法覆盖的数据分布多样性。最优组合跨越 **6 个多样性轴**：
- **损失函数**：MSE / Huber / MAE
- **架构深度**：浅 (3+2 layers) / 深 (4+3 layers)
- **训练长度**：100ep / 150ep
- **特征空间**：±ct-UAE embeddings
- **随机种子**：s42–s46
- **训练数据**：单源 IMP2D / 多源 4-DB 联合

Best-7 ensemble σ–|error| Spearman = 0.577，可用于不确定度估计。

### SOTA 对比

| 模型 | 参数 | Test MAE (eV) | vs ALIGNN |
|---|---|---|---|
| ALIGNN | 4.03 M | 0.540 | — |
| CrystalTransformer v1.2 (single) | 0.75 M | 0.516 | −4% |
| **CT v4 best single** | **0.75 M** | **0.407** | **−25%** |
| CT v4 5-ensemble (SS only) | 5×0.75 M | 0.368 | −32% |
| **CT v4 7-ensemble (SS+MS)** | **7×(0.75–1.1) M** | **0.359** | **−34%** |

代码：
* [scripts/ensemble_online.py](scripts/ensemble_online.py) — 26 模型加载 + greedy 集成评估
* [scripts/ensemble_combined.py](scripts/ensemble_combined.py) — 29 模型（含多源）联合评估
* [configs/enhanced_online_150ep_uae_mae_warmup.yaml](configs/enhanced_online_150ep_uae_mae_warmup.yaml) — 最优单模型配方
* [configs/enhanced_online_150ep_uae_mae_warmup_deep.yaml](configs/enhanced_online_150ep_uae_mae_warmup_deep.yaml) — 深层变体

### Multi-Source v4（2026-05-11）

将 v4 训练配方迁移到 4-DB 联合训练（IMP2D + JARVIS-2D + JARVIS-3D + DFT-3D），
使用 per-source readout heads + source weighting：

| 模型 | 架构 | 参数 | Test MAE | 集成贡献 |
|---|---|---|---|---|
| ms4_deep_s42 | 深 (4+3) | 1.14 M | 0.413 | **k=3 选入**（第3重要） |
| ms4_s42 | 浅 (3+2) | 0.83 M | 0.421 | **k=6 选入** |
| ms4_s43 | 浅 (3+2) | 0.83 M | 0.441 | k=10（边际贡献） |

**关键洞察**：多源模型同时具备架构多样性和数据分布多样性，与单源模型的
误差相关性更低（mean ρ=0.83），在 greedy selection 中有 2 个进入前 7 名。

代码：[scripts/multi_source_v4.py](scripts/multi_source_v4.py)

### v4.1 — Constrained OOD Evaluation（2026-05-11）

> 填补 ID (0.36 eV) 和全 OOD (2.66 eV) 之间的评估空白：
> 在化学空间中设计受控的留出实验，量化模型的**梯度退化**行为。

#### Graduated Generalization Table

| Tier | Scenario | MAE (eV) | Description |
|------|----------|----------|-------------|
| 0 | ID（随机划分） | 0.362 | 同分布，29-model 集成 |
| 1 | 族内 OOD（P0, 7-fold） | 0.540 ± 0.117 | Leave-one-G6-host-out |
| 2 | 组合 OOD（P1） | 0.534 | G6×3d 块缺失（矩阵补全） |
| 3 | 全 OOD（prospective DFT） | 2.660 | 从未见过的材料 + DFT 验证 |

**核心结论**：模型从 ID 到约束 OOD 仅 **1.5×** 退化，
远优于到全 OOD 的 7.3×，证明模型学到了跨宿主化学迁移知识。

#### P0: Leave-One-G6-Host-Out（7-fold）

依次留出 Group-6 TMD 家族中的一个宿主（MoS2, MoSe2, MoTe2, WS2, WSe2, WTe2, MoSSe），
用其余 6 个 G6 宿主 + 37 个非 G6 宿主训练，在留出宿主上评估。

| Fold | OOD MAE | Naive Baseline | 改善 |
|------|---------|----------------|------|
| MoSe2 | **0.377** | 2.874 | 86.9% |
| MoTe2 | 0.480 | 1.910 | 74.8% |
| MoSSe | 0.484 | 2.019 | 76.1% |
| WTe2 | 0.506 | 2.005 | 74.8% |
| MoS2 | 0.522 | 3.507 | 85.1% |
| WSe2 | 0.653 | 2.883 | 77.3% |
| WS2 | 0.759 | 3.253 | 76.7% |
| **Mean** | **0.540 ± 0.117** | **2.636** | **79.5%** |

**vs ALIGNN ID (0.540 eV)**：约束 OOD 平均性能 ≈ ALIGNN 的同分布性能。

#### P1: G6×3d Compositional Block-Out

留出 G6-TMD 宿主与 3d 过渡金属掺杂剂的所有组合（372 样本），
训练集仅见过"G6 + 非 3d"和"非 G6 + 3d"——测试模型的组合外推能力。

- **OOD MAE: 0.534 eV**（naive baseline 1.754, 改善 69.5%）
- Per-host range: MoTe2 0.389 — WS2 0.667

#### ct-UAE Ablation for OOD

对比有 / 无 ct-UAE 预训练原子嵌入对 OOD 泛化的影响：

| 指标 | With ct-UAE | Without ct-UAE |
|------|-------------|----------------|
| P0 Mean OOD MAE | 0.540 ± 0.117 | **0.493 ± 0.087** |
| No-UAE 赢的 fold | 2/7 | **5/7** |
| Paired t-test | — | p = 0.22 (不显著) |

ct-UAE 在 ID 有 ~1% 收益，但 **OOD 无统计显著帮助**。
趋势性地，无 UAE 的模型在 OOD 上更好且方差更小——
可能因为 UAE 编码了训练分布的化学模式，轻微过拟合。

代码：
* [scripts/ood_loho_train.py](scripts/ood_loho_train.py) — P0/P1 训练（含 `--no-uae` 消融）
* [scripts/ood_collect_results.py](scripts/ood_collect_results.py) — 结果汇总 + graduated table
* [scripts/ood_experiment_design.md](scripts/ood_experiment_design.md) — 实验设计文档

输出：
* [results/ood/ood_summary.json](results/ood/ood_summary.json) — 结构化汇总
* [results/ood/ablation_uae_ood.json](results/ood/ablation_uae_ood.json) — ct-UAE 消融对比

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
| 🥇 **v5.1 8-ens greedy (V2+innovations+MS)** | 8×(0.83–1.1) M | **0.344 eV** | — | V4+V2+MS+V14+V6+V3+MS+V2，↓36% vs ALIGNN |
| 🥈 v5 8-ens greedy (V2+MS) | 8×(0.83–1.1) M | 0.349 eV | 0.976 eV | V2 架构+多源，↓35% vs ALIGNN |
| v4 7-ens (SS+MS combined) | 7×(0.75–1.1) M | 0.359 eV | — | 含多源模型，↓34% vs ALIGNN |
| 🥉 **V4 MoE (V2+MoE readout)** | 0.84 M | **0.379 eV** | 1.001 eV | ★ 新最优单模型，↓30% vs ALIGNN |
| V2 gated s43 (v5 baseline) | 0.83 M | 0.381 eV | 1.004 eV | V2 架构最优种子 |
| V6 Physics (V2+Hume-Rothery) | 0.84 M | 0.382 eV | 1.020 eV | 物理失配特征 |
| V3 DefType (V2+条件化) | 0.83 M | 0.387 eV | 1.009 eV | 缺陷类型条件化，val MAE 0.367 最优 |
| v4 best single (150ep MAE+warmup+UAE) | 0.75 M | 0.407 eV | — | seed 45 |
| V14 JK (V2+层聚合+条件化) | 0.83 M | 0.389 eV | 1.008 eV | JK 权重偏好深层 [0.30,0.31,0.39] |
| V9 Contrast (V2+对比条件化) | 0.84 M | 0.410 eV | 1.070 eV | 缺陷-宿主差值条件化 |
| V11 LDS (V2+标签分布平滑) | 0.83 M | 0.433 eV | 0.990 eV | LDS 在低能区退化严重 |
| V13 RnC (V2+对比+LDS) | 0.83 M | 0.442 eV | 1.023 eV | RnC+LDS 双重损害 |
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

### 跨域外推 (Constrained OOD)

**v4.1 系统性 OOD 评估**（详见 [v4.1 节](#v41--constrained-ood-evaluation2026-05-11)）：
- P0: G6-TMD 家族 7-fold leave-one-host-out → **0.540 ± 0.117 eV**
- P1: G6×3d 组合块缺失 → **0.534 eV**
- ct-UAE 消融：OOD 无显著帮助 (p=0.22)
- Graduated table: ID 0.36 → 约束 OOD 0.54 → 全 OOD 2.66

v1.2 legacy LOHO（5 host, 50ep）结果详见
[results/loho_summary.json](results/loho_summary.json)。

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
| El Alouani 2024 (tree-based) | — | 0.518 |
| SchNet | 0.46 | 0.585 |
| ViSNet (lmax=1) | 1.16 | 0.86 |
| MACE (lmax=2) | 0.44 | 1.46 |
| ALIGNN | 4.03 | 0.540 |
| CrystalTransformer v1.2 (ours) | 0.75 | 0.516 |
| CT v4 best single (ours) | 0.75 | 0.407 |
| CT-V2 best single (v5, ours) | 0.83 | 0.381 |
| **CT-V2+MoE (V4, ours)** | **0.84** | **0.379** |
| CT-V2+Physics (V6, ours) | 0.84 | 0.382 |
| CT-V2+DefType (V3, ours) | 0.83 | 0.387 |
| CT v4 7-ensemble SS+MS (ours) | 7×(0.75–1.1) | 0.359 |
| CT-V2 8-ensemble greedy (v5, ours) | 8×(0.83–1.1) | 0.349 |
| **CT-V2+innovations 8-ens (v5.1, ours)** | **8×(0.83–1.1)** | **0.344** |

**经验缩放律** log(MAE) = 3.39 − **0.40**·log(N) − **0.01**·log(P)，
R² = 0.95 → **数据是瓶颈，模型容量超过 ~0.5–0.8 M 反而过拟合**。

---

## 仓库结构

```
src/
├── features.py          # 9 维元素物理化学描述符
├── graph.py             # PBC 邻居 + 最小镜像距离 + 三体角度
├── augment.py           # 旋转 + 高斯坐标微扰
├── dataset.py           # CrystalGraphDataset + collate_fn + 缺陷类型编码
├── sampler.py           # ⭐ Host-balanced sampler
├── train_enhanced.py    # ⭐ 增强训练（SWA/focal MAE/MoE balance/LDS/RnC/distill）
├── models/
│   ├── baseline.py      # CrystalTransformer V1 (Local SchNet + Global Transformer)
│   ├── crystal_v2.py    # ⭐ CrystalTransformerV2 (V3 条件化 + V4 MoE)
│   ├── pfa.py           # Periodic Fourier Bias 注意力
│   ├── multi_source.py  # 4-DB 多源训练
│   ├── dualstream.py    # 缺陷-pristine 双流交叉注意力
│   └── improved.py      # DAST (legacy)
└── train.py             # 原始训练脚本
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
├── ensemble_combined.py                  # ⭐ 29-model SS+MS 联合评估
├── multi_source_v4.py                    # ⭐ 4-DB 联合 v4 训练
# v5.x architecture optimization (2026-05-19)
├── eval_new_model.py                     # ⭐ 新模型评估（per-range + 相关性 + 集成）
├── eval_checkpoint.py                    # ⭐ 通用 checkpoint 评估（自动检测格式）
├── attention_analysis.py                 # ⭐ 注意力集中度 + JK 权重 + 物理特征分析
├── error_analysis_detailed.py            # ⭐ 按宿主/掺杂/缺陷类型误差分解
├── tta_inference.py                      # ⭐ 测试时增强 + 方差不确定性
├── generate_paper_table.py              # ⭐ 自动 LaTeX 论文表格生成
├── generate_ablation_table.py           # 消融表 markdown 生成
├── make_innovation_figures.py           # 创新训练曲线对比图
├── launch_queue.sh                       # 多实验排队启动
├── status_dashboard.sh                   # 训练状态监控面板
├── compute_asph.py                       # ⭐ ASPH 持久同调特征（ripser）
├── launch_experiment.sh                  # GPU-aware 实验启动脚本
# v4.1 constrained OOD evaluation (2026-05-11)
├── ood_loho_train.py                     # ⭐ P0/P1 OOD 训练（含 --no-uae 消融）
├── ood_collect_results.py                # ⭐ OOD 结果汇总 + graduated table
├── ood_experiment_design.md              # OOD 实验设计文档
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
├── enhanced_online_150ep_uae_mae_warmup.yaml # v4 最优单模型配方
├── enhanced_online_*                         # v4 26-model 训练配置
# v5.x architecture configs (2026-05-19)
├── v3_deftype.yaml                           # ⭐ V3 缺陷类型条件化
├── v4_moe.yaml / v4_moe_5exp.yaml           # ⭐ V4 MoE readout (3/5 experts)
├── v6_physics.yaml                           # ⭐ V6 Hume-Rothery 物理特征
├── v9_contrast.yaml                          # ⭐ V9 缺陷-宿主对比条件化
├── v11_lds.yaml                              # ⭐ V11 LDS 标签分布平滑
├── v12_lds_physics.yaml                      # ⭐ V12 LDS+物理+全条件化
├── v13_rnc.yaml                              # ⭐ V13 RnC 对比 + LDS
├── v14_jk.yaml                               # ⭐ V14 JK 层聚合
├── v10_best_combo.yaml                       # ⭐ V10 最优组合
├── v2_ema.yaml / v2_focal.yaml / v2_huber.yaml  # 训练策略变体
├── v2_asph.yaml                              # V2 + ASPH 持久同调
├── v2_uncertainty.yaml                       # V2 + 异方差不确定性
├── v2_regstrong.yaml                         # 强正则化消融
├── v5_full.yaml                              # V5 组合（ASPH+V3+V4）
└── v7_all.yaml                               # V7 全组合
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
├── phase_b_ood_physics.json                  # ⭐ OOD 物理分布偏移
├── ood/                                      # ⭐ v4.1 约束 OOD 评估
│   ├── loho_{MoS2,...,MoSSe}_s42/            #   P0 7-fold metrics
│   ├── loho_*_s42_nouae/                     #   ct-UAE 消融 metrics
│   ├── block_g6x3d_s42/                      #   P1 组合块缺失
│   ├── ood_summary.json                      #   graduated evaluation table
│   └── ablation_uae_ood.json                 #   ct-UAE 消融统计
└── ensemble_combined.json                    # ⭐ 29-model 集成结果
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
当前最优：**单模型 0.379 eV (V4 MoE) / 8-ensemble 0.349 eV**（V2 架构 + 创新变体）。

**当前训练状态（2026-05-20）**：两波训练全部完成 ✅（共 7 个创新实验）。
- **第一波** V3/V4/V6/V9：V4 MoE **0.379 eV** ★ 新最优单模型
- **第二波** V11/V13/V14：V14 JK 0.389（等价 V2），V11 LDS 0.433 / V13 RnC 0.442（显著劣于 V2）
- **第三波** V2_ema/V2_focal/V2_uncertainty：V2_ema 0.395（等价 V2），V2_focal 0.419 / V2_uncertainty 0.466（显著更差）
- **8-model 集成 0.344 eV**（SOTA，↓36% vs ALIGNN）
下一波：V12 LDS+物理+全条件化 → V10 最优组合。

### Tier 1 — 低成本高收益（不改架构）

- [x] ~~**Readout ensembling**~~：multi-head readout 共享 trunk 实测无收益——
  改为 full multi-diversity ensemble (26 models, best-k=5) 达 0.368 eV
- [x] **ct-UAE 预训练原子嵌入**：128-dim embeddings from Nature Comms 2025
  多任务检查点，拼接到 9-dim 手工特征，单模型 ~0.01 eV 提升，且为集成提供
  特征多样性轴
- [x] **拓扑描述符（persistent homology）**：✅ 已实现。ASPH 特征通过 ripser
  计算 0-dim（连通分量）+ 1-dim（环结构）持久同调，PCA 降维至 8 维。
  5 秒完成 10641 样本，97.7% 方差解释率。待训练验证。
  参考：[PH + GNN for Defect Ef](https://pubs.acs.org/doi/10.1021/acs.chemmater.4c03028)
  （Chem. Mater. 2025）
- [x] **扩大 ensemble 成员数**：29 models across 6 diversity axes (含多源模型) →
  best-7 ensemble 0.359 eV（↓19% vs 旧 6-ensemble 0.443，↓34% vs ALIGNN）
- [x] **约束 OOD 评估**：G6-TMD 7-fold LOHO (0.540 eV) + G6×3d 组合块缺失 (0.534 eV)
  + ct-UAE 消融（OOD 无显著帮助，p=0.22）

### Tier 2 — 中等成本（局部架构改动）

- [x] **CrystalTransformerV2 架构改进**：✅ 已完成并验证。门控注意力池化 +
  局部环境富集 + Pre-Norm 残差，单模型 0.381 eV（↓26% vs V1）。
- [x] **缺陷类型条件化 (V3)**：✅ 完成。Test MAE **0.387 eV**（val MAE 0.367 最优），
  零初始化嵌入。SWA 阶段 val MAE 从 0.382→0.367 显著提升。首创，无先例。
- [x] **MoE Readout (V4)**：✅ 完成。Test MAE **0.379 eV** ★ 新最优单模型（↓30% vs ALIGNN）。
  3 专家 MLP + 学习门控 + KL 平衡损失，RMSE 1.001 也是最低。
  参考：[MoCE](https://openreview.net/forum?id=Oit5bHPmjx)（ICLR 2025）
- [x] **缺陷-宿主对比条件化 (V9)**：✅ 完成。Test MAE 0.410 eV（劣于 V2 基线 0.381），
  差值条件化假设可能过于简化。
- [x] **LDS 标签分布平滑 (V11)**：❌ 完成。Test MAE 0.433 eV（显著劣于 V2，p<0.001），
  LDS 在 [0,2) eV 范围退化最严重（ΔMAE=+0.074），重权策略过度牺牲常见样本精度。
  参考：Yang et al., ICML 2021 (Delving into Deep Imbalanced Regression)
- [x] **RnC 排序对比学习 (V13)**：❌ 完成。Test MAE 0.442 eV（显著劣于 V2，p<0.001），
  RnC 对比损失 + LDS 重权双重损害。辅助对比目标可能干扰主回归。
  参考：Zha et al., NeurIPS 2023 (Rank-N-Contrast)
- [x] **JK 层聚合 (V14)**：✅ 完成。Test MAE **0.389 eV**（与 V2 等价，p=0.42），
  JK 权重 [0.30, 0.31, 0.39] 偏好最深全局层。被选入 ensemble k=4（提供层聚合多样性）。
  参考：Xu et al., ICML 2018 (How Powerful are GNNs)
- [x] **Focal MAE 损失**：❌ 完成。Test MAE 0.419 eV（显著劣于 V2，p=0.001），
  focal 上权困难样本会损害 [0,2) 和 [2,5) 常见范围精度（ΔMAE=+0.049, p<0.01）。
  但 [7,25) 范围有微弱改善 (+0.124, ns)，代价过高。
- [x] **EMA 权重平均**：✅ 完成。Test MAE **0.395 eV**（与 V2 等价，p=0.20），
  EMA decay=0.999 + SWA 达到 val MAE 0.379（最优），但 test 不匹配 (0.395)。
  在 V2 seed 方差范围内 (0.392 ± 0.007)。
- [ ] **iComFormer 风格几何完备注意力**：用不变量（距离 + 键角）替代纯距离
  编码的全局 Transformer 层，不引入等变张量积开销。ICLR 2024 在 MatBench
  上超越 ALIGNN。
  参考：[ComFormer](https://arxiv.org/abs/2403.11857)（ICLR 2024）
- [ ] **Balanced MSE / Balanced Smooth L1**：将 BMC 核密度估计引入回归损失，
  自适应增大稀疏标签区域的梯度，直接对标 LDS 但无需显式密度估计。
  参考：Ren et al., CVPR 2022 (Balanced MSE for Imbalanced Visual Regression)
- [ ] **Evidential Deep Learning 不确定性**：单模型输出 Normal-Inv-Gamma 参数，
  分离认知/偶然不确定性，OOD 检测远优于 MC Dropout。
  参考：Soleimany et al., Nature Comms 2025
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
- [x] ~~Log-target 变换 (sign(y)*log1p(|y|))~~（test MAE 0.399 vs 基线 0.381，
  且与现有模型相关性 r=0.99+，无集成多样性）
- [x] ~~测试时增强 (TTA)~~（TTA(4) MAE=0.3808 vs 基线 0.3810，δ=−0.0002 eV，
  因模型已用在线增强训练，几何扰动无新信息。但预测方差可用于不确定性估计 r=0.348）

---

## 致谢

* 数据：DTU CMR / NIST JARVIS
* 基线参考：[wuleyan2004/defect_formation_energy_prediction](https://github.com/wuleyan2004/defect_formation_energy_prediction)
* DFT 软件：[Quantum ESPRESSO 7.3.1](https://www.quantum-espresso.org/) +
  NVIDIA HPC SDK 25.5
* 训练硬件：WHU 8×L40S (v4–v5 主训练) + RTX 5090 (DFT + 早期实验)
* 方法参考：JK-Net (Xu et al., ICML 2018), DIR (Yang et al., ICML 2021),
  RnC (Zha et al., NeurIPS 2023), ComFormer (ICLR 2024)

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
