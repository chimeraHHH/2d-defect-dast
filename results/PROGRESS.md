# 研究进度记录 (FINAL v1.1 - honest)

> 项目目标：基于 IMP2D 数据库构建二维材料缺陷形成能高通量预测平台
> 立项目标：MAE < 0.2 eV
> 最终诚实结论：**与 ALIGNN 持平 (0.537 ± 0.016 eV vs 0.540 eV) 但仅用 1/5 参数；
> 立项 < 0.2 eV 在严格防泄漏对比下未达成**

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

## 三个核心发现

1. **简单架构 + leak-free aug 与 ALIGNN 在 IMP2D 上统计意义打平**, 仅用 1/5 参数。
2. **DAST 设计 (虚拟锚点 + 晶格自连接 + 星型稀疏) 在 IMP2D 上整体有害**, 退化 0.6-1.0 eV。
3. **aug-then-split 是隐蔽数据泄漏陷阱**: 同一 checkpoint 在 leaky aug-test 上 0.206, 而严格防泄漏从头训练得 0.516, 即 2.5× 虚假改进。

## 撤回声明

v1.0 中"0.21 eV 达成 < 0.2 eV 项目目标 / 击败 ALIGNN 2.6×"的表述基于
有数据泄漏的实验, **不正确**, 已在 v1.1 中诚实化撤回。本项目的真实学术
贡献在于 (i) 紧凑配方的工程价值与 (ii) 揭示并修复了一个常见的隐性
泄漏方法论陷阱。

## 交付物

- 代码: https://github.com/chimeraHHH/2d-defect-dast (public)
- Release v1.1: https://github.com/chimeraHHH/2d-defect-dast/releases/tag/v1.1
- 论文初稿: paper/main.md (中文 ~390 行) 与 paper/main.pdf
- 实验图表: paper/figures/ (parity / curves / error_dist; 全部基于 leak-free safe 数据)
- 全部 metrics.json / train.log / test_predictions.npz 在 results/<run>/
- leak-free 数据增强脚本: scripts/build_leak_free_aug.py (社区参考)
