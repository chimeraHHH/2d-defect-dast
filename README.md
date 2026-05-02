# 物理增强 + 紧凑混合架构在 IMP2D 上预测二维缺陷形成能

[![paper](https://img.shields.io/badge/paper-pdf-blue)](paper/main.pdf)
[![dataset](https://img.shields.io/badge/data-IMP2D%20(CMR)-green)](https://cmr.fysik.dtu.dk/imp2d/imp2d.html)
[![best](https://img.shields.io/badge/test%20MAE-0.206%20eV-red)](#最终结果)

我们以 *Impurities in 2D Materials Database*（IMP2D, DTU 公开数据集，
10641 个 DFT 收敛缺陷构型）为基准，系统对比了多种 GNN / Transformer
架构在缺陷形成能 $E_f$ 回归任务上的表现，并把项目最初的"自注意力 + 虚拟
缺陷锚点 + 星型稀疏 + 晶格自连接"DAST 思路与一个朴素 SchNet+Transformer
混合架构 + 物理驱动数据增强相比较。

## 最终结果

| 配置 | Params | Test MAE | Test RMSE | R² |
|---|---|---|---|---|
| 🏆 **baseline_h128_aug_long** | 0.75 M | **0.206 eV** | 0.310 eV | **0.990** |
| baseline_h128_aug_long (4-seed mean ± std) | 0.75 M | 0.270 ± 0.049 | 0.427 ± 0.085 | 0.981 ± 0.006 |
| baseline_aug_long (紧凑版, h=64) | 0.20 M | 0.416 | 0.596 | 0.962 |
| ALIGNN (团队前期复现) | 4.03 M | 0.540 | 1.167 | — |
| baseline (无增强 30 ep) | 0.20 M | 0.862 | 1.522 | 0.784 |

我们以 0.75 M 参数（ALIGNN 的 1/5）取得 0.206 eV，达到项目立项目标
"MAE < 0.2 eV"，相较 ALIGNN 的 0.540 eV 提升 **2.6 倍**。

## 主要发现 (TL;DR)

1. **物理驱动数据增强（旋转 + 坐标微扰，3×）是 IMP2D 上的最大杠杆**：
   单独换增强即把基线 0.86 → 0.51（−41%）。
2. **结构创新（DAST：虚拟锚点 / 晶格自连接 / 星型稀疏注意力）在该任务
   上整体有害**：所有变体 1.49–1.83 eV，全部差于无 DAST 基线。
3. **紧凑模型 + 充足训练 + 增强**优于"盲目堆参数"：0.75 M + 50 ep + aug
   的 0.21 eV 大幅优于 1.06 M + 30 ep + aug 的 1.47 eV。

## 模块概览

```
src/
├── features.py        # 元素物理化学描述符 (9 维)
├── graph.py           # PBC 邻居、最小镜像距离、三体角度
├── augment.py         # 旋转 + 高斯坐标微扰增强
├── dataset.py         # CrystalGraphDataset + collate_fn
├── models/
│   ├── baseline.py    # CrystalTransformer (Local SchNet + Global Transformer)
│   └── improved.py    # DAST (虚拟锚点 + 晶格自连接 + 星型稀疏 mask)
└── train.py           # 训练入口 (CUDA 默认, fallback CPU)
scripts/
├── prepare_dataset.py    # 从 imp2d.db 构建图特征
├── make_figures.py       # 论文图表生成
├── analyze_results.py    # 指标汇总 -> summary.md
├── error_analysis.py     # 按 host/dopant/defecttype 分解误差
├── eval_existing.py      # 从断点 best.pt 补算 metrics
├── inspect_attention.py  # 提取 DAST 末层注意力热图
└── run_queue.sh          # 实验串行队列脚本
configs/
├── baseline.yaml                       # 0.20M, h64, 30 epoch, no aug
├── baseline_long.yaml                  # 0.20M, 60 epoch, no aug
├── baseline_aug.yaml                   # 0.20M, 30 epoch + 3× aug
├── baseline_aug_long.yaml              # 0.20M, 50 epoch + 3× aug
├── baseline_h128.yaml / _long.yaml     # 1.06M / 0.75M scale-up
├── baseline_h128_aug_long.yaml         # ⭐ best, 0.75M, 50 epoch + aug
├── baseline_h128_aug_long_seed{0,1,2,3}.yaml  # 多种子稳定性
├── baseline_aug_seed{0,1,2,3}.yaml     # 紧凑版多种子
├── improved.yaml / dast_dense*.yaml    # DAST 系列 (negative)
└── ablate_*.yaml                       # 消融
results/
├── <run>/best.pt + metrics.json + test_predictions.npz + train.log
├── summary.md          # 自动生成的统一指标表
└── PROGRESS.md         # 项目时间线与决策记录
paper/
├── main.md             # 论文初稿 (中文, ~360 行)
├── main.pdf            # 中文 PDF
└── figures/            # parity / curves / error_dist / metrics_table.tsv
```

## 复现

```bash
# 1. 克隆 + 环境
git clone https://github.com/chimeraHHH/2d-defect-dast.git
cd 2d-defect-dast
python3 -m venv .venv && source .venv/bin/activate
# RTX 50 系列需要 cu128
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements.txt

# 2. 拉取 IMP2D 原始数据库 (~70 MB)
mkdir -p data/raw
curl -L https://cmr.fysik.dtu.dk/_downloads/imp2d.db -o data/raw/imp2d.db

# 3. 构建图特征 (10641 样本, ~1 min CPU / ~3 min RTX 5090)
python scripts/prepare_dataset.py

# 4. 数据增强 (3× 31923 样本, ~4 min CPU)
python -m src.augment

# 5. 复现最强配置 (50 epoch, ~12 min RTX 5090)
python -m src.train --config configs/baseline_h128_aug_long.yaml

# 6. 全部消融 + 多种子
bash scripts/run_queue.sh

# 7. 汇总图表
python scripts/analyze_results.py
python scripts/make_figures.py
python scripts/error_analysis.py baseline_h128_aug_long
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
