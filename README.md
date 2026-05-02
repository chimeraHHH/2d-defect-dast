# 基于自注意力机制的二维材料缺陷性质高通量预测平台

研究项目代码、数据预处理、训练脚本和论文初稿。基于 IMP2D 数据库（CMR/DTU），目标
任务为缺陷形成能回归。

## 数据来源
- IMP2D database (Computational Materials Repository, DTU): https://cmr.fysik.dtu.dk/imp2d/imp2d.html
- 通过 `scripts/prepare_dataset.py` 解析 imp2d.db → cleaned_dataset.pkl → final_dataset.pkl

## 模块概览
- `src/features.py` — 元素物理化学描述符（9 维），用于节点初始特征
- `src/graph.py` — 周期性图构建（PBC、最小镜像距离、邻居与三元组）
- `src/augment.py` — 物理驱动的数据增强（旋转/坐标微扰）
- `src/dataset.py` — `CrystalGraphDataset` 与 batch collate
- `src/models/baseline.py` — 复现版 CrystalTransformer (Local + Global)
- `src/models/improved.py` — 我们提出的改进模型 (虚拟节点 + 星型稀疏 + 学习偏置)
- `src/train.py` / `src/eval.py` — 训练 / 评估入口

## 运行

```bash
source .venv/bin/activate
# 1. 准备数据
python scripts/prepare_dataset.py
# 2. 训练基线
python -m src.train --config configs/baseline.yaml
# 3. 训练改进模型
python -m src.train --config configs/improved.yaml
# 4. 评估
python -m src.eval --config configs/improved.yaml
```
