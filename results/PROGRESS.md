# 研究进度记录 (v1.2 - 顶刊四维评估)

> 项目目标：基于 IMP2D 数据库构建二维材料缺陷形成能高通量预测平台
> 立项目标：MAE < 0.2 eV
> 最终诚实结论：**与 ALIGNN 持平 (0.537 ± 0.016 eV vs 0.540 eV) 但仅用 1/5 参数；
> 立项 < 0.2 eV 在严格防泄漏对比下未达成**

> **v1.2 升级**：补充了三项材料 ML 顶刊关键评估指标：
> - **OOD/LOHO**：5 host (MoS2/Cr2I6/C2H2/TaSe2/MoSSe) 留一宿主验证
> - **UQ + 校准**：4-seed 深度集成 → MAE 0.464 (−10%)；
>   τ=2.60 温度缩放 → NLL 2.86→1.01；90% 区间覆盖 72.5%→93.4%
> - **物理可解释性**：自注意力 32× 缺陷集中度；
>   occlusion 缺陷原子贡献 90.7% / 残余信号扩散 9 Å > SchNet 5 Å 截断

## 关键里程碑

- 2026-05-02 13:43 项目启动；从 DTU CMR 拉取 IMP2D 10641 样本
- 2026-05-02 14:30 修复 LocalInteractionLayer 收敛问题 (SchNet 风格滤波卷积)
- 2026-05-02 19:00 远端 RTX 5090 GPU 接入
- 2026-05-02 19:30 DAST sparse / dense 实验 → 全部劣于基线 (负面)
- 2026-05-02 20:00 (有泄漏) baseline_aug 取得 0.51 eV → "突破"假象
- 2026-05-02 21:00 (有泄漏) baseline_h128_aug_long 取得 0.21 eV → 发布 v1.0 (后被撤回)
- 2026-05-03 00:00 用户指出 ALIGNN 测试不增强、对比不公平
- 2026-05-03 00:30 进一步发现 aug-then-split 全套数据泄漏 (~99% 测试原样本被训练)
- 2026-05-03 01:00 build_leak_free_aug.py: 先划分后增强, 25536 train + 1064 val + 1065 test
- 2026-05-03 01:30 leak-free 重跑得 baseline_h128_aug_long_safe 0.516
- 2026-05-03 02:00 4-seed 验证 mean 0.537 ± 0.016, 与 ALIGNN 0.540 统计持平
- 2026-05-03 02:30 单 seed h128 100-epoch 得 0.478, 但单 seed 不足以宣告
- 2026-05-03 03:00 v1.1 发布 (诚实化版本)
- 2026-05-03 16:30 启动 v1.2: OOD/UQ/解释性顶刊三件套
- 2026-05-03 16:35 LOHO 基建 commit (5 hosts × build_loho.py)
- 2026-05-03 16:42 UQ 校准脚本 (NLL/CRPS/ECE/温度缩放) → 显著改善
- 2026-05-03 16:48 注意力提取 (CrystalTransformer 多层多头) → 32× 缺陷集中度
- 2026-05-03 16:50 occlusion 归因 → 缺陷原子占总归因 90.7%
- 2026-05-03 16:55 误差分解 (defect_type / size / dopant_block / |Ef|)
- 2026-05-03 17:00 论文 §5.6/8/9/10 + 摘要 + 引言 + 结论重写
- 2026-05-03 17:05 远程构建 5 个 LOHO 数据集 (parallel CPU)
- 2026-05-03 17:08 终止 xlong 队列以优先 LOHO，让 GPU 独占给 LOHO 训练

## 最终结果（IMP2D 测试集 - 与 ALIGNN 同一 1065 样本）

| 配置 | Params | Test MAE | 备注 |
|---|---|---|---|
| baseline_h128_aug_xlong_safe (100ep, single seed=42) | 0.75M | 0.478 | 单 seed, 不作主结论 |
| baseline_h128_aug_long_safe (50ep, seed=42) | 0.75M | 0.516 | best of 4 seeds |
| **4-seed mean (50ep)** | 0.75M | **0.537 ± 0.016** | 主结论数字 |
| ALIGNN (报告引用) | 4.03M | 0.540 | 文献基线 |
| baseline_aug_long_safe (h64, 50ep) | 0.20M | 0.628 | 紧凑版 |
| baseline_h128_long (60ep, no aug) | 0.75M | 0.622 | 无增强长训练 |
| baseline (h64, 30ep, no aug) | 0.20M | 0.862 | 默认基线 |
| improved (DAST sparse, 25ep) | 0.20M | 1.827 | DAST 完整版 (负面) |

## v1.1 核心发现

1. **简单架构 + leak-free aug 与 ALIGNN 在 IMP2D 上统计意义打平**, 仅用 1/5 参数。
2. **DAST 设计 (虚拟锚点 + 晶格自连接 + 星型稀疏) 在 IMP2D 上整体有害**, 退化 0.6-1.0 eV。
3. **aug-then-split 是隐蔽数据泄漏陷阱**: 同一 checkpoint 在 leaky aug-test 上 0.206, 而严格防泄漏从头训练得 0.516, 即 2.5× 虚假改进。

## v1.2 新增 (顶刊三件套)

4. **深度集成 + 温度缩放 = 可信不确定度量化**：4-seed 深度集成把 MAE
   从 0.516 → 0.464 eV（−10% 免训练增益）；原始 σ 欠估计（90% 区间
   实测覆盖 72.5%），单标量 τ=2.60 缩放后 NLL 2.86→1.01、90% 覆盖
   93.4%。脚本 ``scripts/uq_calibration.py``。
5. **跨域外推 (LOHO) 评估**：5 host (MoS2 / Cr2I6 / C2H2 / TaSe2 /
   MoSSe) 留一宿主验证；每 host 留出 ~300 样本，模型重训 50 epoch；
   给出"对未见过化学族"泛化退化的直接量化（详细数字待 GPU 队列完成
   填入）。脚本 ``scripts/build_loho.py`` + ``scripts/loho_summary.py``。
6. **自注意力 + occlusion 联合可解释性诊断**：
   - 模型自发把缺陷原子学成全局枢纽 (attention 32× / attribution 448×)
     → 否定原 DAST"虚拟锚点"动机；
   - 缺陷原子单原子贡献占总归因 **90.7%**（occlusion 测量）；
   - 剩余 ~10% 信号扩散到 9 Å 半径，**超过** SchNet 局部截断 5 Å
     → 直接为"局部 + 全局混合架构"提供物理依据。
   - 脚本 ``scripts/attention_baseline.py`` + ``scripts/occlusion_attribution.py``。

## 撤回声明

v1.0 中"0.21 eV 达成 < 0.2 eV 项目目标 / 击败 ALIGNN 2.6×"的表述基于
有数据泄漏的实验, **不正确**, 已在 v1.1 中诚实化撤回。本项目的真实学术
贡献在于 (i) 紧凑配方的工程价值与 (ii) 揭示并修复了一个常见的隐性
泄漏方法论陷阱。

## 交付物

- 代码: https://github.com/chimeraHHH/2d-defect-dast (public)
- Release v1.1: https://github.com/chimeraHHH/2d-defect-dast/releases/tag/v1.1
- 论文初稿 v1.2: paper/main.md (中文 ~610 行) 与 paper/main.pdf
- 实验图表: paper/figures/
  - 基础: parity / curves / error_dist (基于 leak-free safe 数据)
  - v1.2 新增: fig_uq_reliability.png / fig_uq_calibration.png /
    fig_attention_heads.png / fig_attention_defect_centric.png /
    fig_occlusion_localisation.png / fig_error_by_category.png /
    fig_loho_bars.png (待填入)
- 全部 metrics.json / train.log / test_predictions.npz 在 results/<run>/
- v1.2 新增脚本：
  - ``scripts/build_loho.py`` — leave-one-host-out 数据集构建
  - ``scripts/uq_calibration.py`` — NLL/CRPS/ECE 校准 + 温度缩放
  - ``scripts/attention_baseline.py`` — 多层多头注意力提取
  - ``scripts/occlusion_attribution.py`` — per-atom 占位归因
  - ``scripts/error_decomposition.py`` — 6 维误差分解
  - ``scripts/loho_summary.py`` — LOHO 结果汇总
- leak-free 数据增强脚本: scripts/build_leak_free_aug.py (社区参考)
