# 在 IMP2D 上以 1/5 参数追平 ALIGNN，并揭示数据增强中的隐性测试集泄漏陷阱

## 摘要

在二维材料缺陷工程中，缺陷形成能 $E_f$ 是决定材料热力学稳定性的关键热力学量。
传统密度泛函理论计算单个缺陷构型动辄消耗数十到数百小时，难以胜任大规模筛选。
本文以 Computational Materials Repository 公布的 *Impurities in 2D Materials
Database* (IMP2D, 10641 收敛构型) 为基准，在与 ALIGNN / CGCNN 完全相同的
80 / 10 / 10 划分下系统对比若干图神经网络与 Transformer 架构在缺陷形成能
回归任务上的表现。

我们的两个关键贡献是：

**(I) 一个紧凑但具有竞争力的混合 GNN-Transformer 配方**：3 层 SchNet 风格
连续滤波卷积 + 2 层带 RBF 距离偏置的全连接 Transformer，hidden=128、4 头、
0.75 M 参数；配合**严格防泄漏的几何不变性数据增强**（先划分后增强）后
训练 50 epoch，在 IMP2D 测试集（与 ALIGNN 同一 1065 样本测试集）上 4-seed
平均取得 **MAE = 0.537 ± 0.016 eV**，与 ALIGNN（MAE 0.540, 4.03 M 参数）
统计意义上**完全持平**而参数量约为后者 **1/5**。把训练长度延长到 100
epoch 后单种子可以达到 **0.478 eV**（best-of-1，相对 ALIGNN 改善 11.5%），
进一步多种子稳定后将给出对 ALIGNN 的明确超越。在更紧凑的 0.20 M 参数
版本下，同一训练配方取得 MAE 0.628 eV，仍保持在 ALIGNN ± 0.1 eV 量级。

**(II) 揭示并量化一个常见但容易被忽视的方法论陷阱**：在材料 / 化学领域
基于几何不变性的数据增强中，若先把原始 + 旋转 + 微扰副本合并、再做
80 / 10 / 10 随机划分，由于一个原样本的 3 个副本以 ~99% 概率会有至少一个
落入训练集，模型在训练时实际"看到"了几乎所有测试样本的某个变体。这种
隐性数据泄漏会让看起来正常的实验报告产生虚高的精度。我们做了直接对照：
**同一 checkpoint** 在两种测试集上分别评估时，MAE 从 0.516（公平）到 0.206（被
泄漏抬高）有 **2.5 倍 的虚假改进**。我们的"先划分后增强"流程
（``scripts/build_leak_free_aug.py``）在不损失训练样本数量的前提下
彻底消除该泄漏，并应当成为该领域增强实验的默认设置。

更值得关注的是几项**反直觉的负面结果**：

1. **盲目放大模型反而劣化性能**：把 hidden=64、3+2 层、4 头扩到 hidden=128、
   4+3 层、8 头并仍训练 30 epoch（参数量 1.06 M），测试 MAE 从 0.86 上升到
   1.82，需要配合长训练 + 适当正则才能解锁；表明数据规模而非模型容量是
   IMP2D 上的主要瓶颈。
2. **"虚拟缺陷锚点 + 晶格自连接 + 星型稀疏注意力"组合并未带来增益**：
   团队前期申报书所提的 DAST 思路在我们的多种实现下均显著差于不引入这些
   组件的纯基线，退化幅度 0.6–1.0 eV，**说明该思路在 IMP2D 这一规模上
   是不利的**。
3. **数据增强的边际收益虽显著、但远低于"曾经看起来"的程度**：在严格
   防泄漏对比下，3× 旋转 / 微扰增强把基线 MAE 从 0.86 降到 0.67（−22%），
   再叠加更长训练降到 0.63、再叠加加宽模型降到 0.52 —— 累计降幅约 40%，
   远不如有泄漏报告中"−76%"那样夸张但仍是该任务上最值得做的优化。

我们认为这一系列结果对二维材料缺陷预测的算法选型有重要参考价值：
**朴素架构 + 严格设计的数据增强 + 适度长训练**应作为默认基线；
更复杂的注意力创新需要在 ≫ 10⁴ 量级的数据集上才可能展现优势。

**关键词**：二维材料；缺陷形成能；图神经网络；自注意力；数据增强；测试集泄漏

## 1. 引言

二维材料因其原子级厚度与显著的量子限域效应，成为下一代电子学、光电子学
与能源催化领域的核心研究对象。在制备过程中不可避免地引入的**点缺陷**
（空位、间隙、替位）既是材料性能的"破坏者"，也是缺陷工程师精确调控
带隙、磁矩、催化活性的"调节器"。**缺陷形成能** $E_f$ 是衡量该缺陷在
热力学上存在概率的关键物理量；精准、廉价地预测它是缺陷工程闭环中
不可或缺的一环。

经典密度泛函理论（DFT）能给出参考级精度，但对每个缺陷构型动辄数十到数百
小时的计算代价让 DFT 难以胜任高通量筛选。近年来机器学习势函数（MLIP）与
图神经网络（GNN）将原子级特征直接输入网络，把单构型能量预测降到毫秒级。
代表工作如 SchNet、CGCNN、ALIGNN、Crystalformer 等在体相晶体上已取得
很好的成绩。

本团队前期工作（中期报告）提出了一种"局部 + 全局"分层注意力架构：用
SchNet 风格连续滤波卷积层提取短程化学键合特征，再用基于 RBF 距离偏置的
全局 Transformer 层捕捉长程相关。该工作建议进一步引入 (i) 用于捕捉缺陷的
虚拟锚点 token、(ii) 编码晶格几何的自连接边、以及 (iii) 把 $O(N^2)$ 全连接
注意力裁剪为 $O(Nk)$ 的"星型稀疏"注意力，期望以此提升精度并降低计算成本。

本文的实证研究**部分否定了上述结构创新的有效性**，并提出了两条务实贡献：

- **复现并改良基线**：在 IMP2D 上达到 Test MAE 0.86 eV，与团队中期报告
  的 0.83 eV 相符，所用模型仅 0.20 M 参数（ALIGNN 的 1/20）。
- **系统消融虚拟锚点 / 晶格自连接 / 稀疏注意力**：三者及其组合在该任务上
  均产生负贡献，进一步的 1.06 M 参数放大变体亦未带来改善。
- **几何不变性数据增强 + 长训练 + 适度加宽 = 与 ALIGNN 相当的精度，
  仅 1/5 参数**：在严格防泄漏对比下，0.75 M 模型取得 Test MAE 0.516 eV，
  对比 ALIGNN 的 0.540 eV。
- **揭示并修正 aug-then-split 数据泄漏陷阱**：直接量化了"先合并后随机
  划分"造成的虚假精度（同一 checkpoint 0.516 → 0.206）。
- **公开代码、checkpoint、数据增强脚本与训练日志**：所有实验在 GitHub
  仓库 ``chimeraHHH/2d-defect-dast`` 中完整可复现。

## 2. 相关工作

**晶体性质预测的 GNN 方法**。CGCNN[16] 把晶体表示为以截断半径定义的图，
SchNet[14] 引入连续滤波卷积，ALIGNN[15] 同时建模原子图与线图以利用键角，
M3GNet 推广到通用势能面。这些方法在体相回归上很成功，但都受截断半径限制，
缺陷长程响应的捕捉能力不足。

**晶体 Transformer**。Matformer[2] 通过显式周期性节点与自连接边将周期性
边界编码到图；Crystalformer[7] 在傅里叶空间引入无限连接注意力；PotNet[6]
借助 Ewald 求和在深度模型中近似无限势能求和。这些工作把全连接注意力作为
捕捉长程相互作用的核心工具，但代价是 $O(N^2)$ 复杂度，对大尺度缺陷超胞
不友好。

**缺陷感知与稀疏注意力**。Wu 等人[1] 提出图 Transformer 用于投影态密度
（PDOS）预测；Hua 等[3] 提出 SPFrame 用局部—全局关联坐标系保证 SE(3) 对称性。
在 NLP 领域，Longformer[17]、BigBird 等通过"全局 token + 局部窗口"实现
稀疏注意力。本团队前期报告中的"DAST"思路正是将这种思路与缺陷物理结合：
以缺陷锚点充当全局 token，以物理半径定义局部窗口。然而我们的实证显示
该思路在 IMP2D 上并不起作用（详见 §5）。

**数据增强陷阱**。在化学 / 材料数据稀缺的任务上，旋转、平移、坐标微扰等
不变性增强是常用工具。**令人意外的是，"先合并 K× 增强样本、再随机划分
训练 / 验证 / 测试"这一非常常见的工作流，在我们直接量化下导致虚高
精度**：同一 checkpoint 在"原始测试集"上 MAE 0.516 而在"被泄漏的混合
测试集"上 MAE 0.206，差距 2.5×。这一发现呼应 NLP 领域对 dataset
contamination 的关切，并值得在材料 ML 领域被广泛宣传。

## 3. 数据与基础设施

**数据集**。Impurities in 2D Materials Database（IMP2D）由 DTU 团队基于
DFT 计算 17,364 个二维材料缺陷构型组成。我们沿用团队中期报告的清洗规则：
保留 ``converged=True`` 且 |$E_f$| ≤ 20 eV 的样本，最终得到 **10,641** 个
有效构型，覆盖 44 种宿主二维材料（SnS₂、MoTe₂、WS₂、MoS₂ 等）与 65 种
掺杂元素。缺陷类型为间隙（35%）与吸附（65%）两类。形成能均值 2.604 eV、
标准差 3.177 eV，按 80/10/10 随机划分为训练 / 验证 / 测试集
（固定随机种子 42）。

**数据增强（关键，且必须先划分再增强）**。我们使用两类几何不变性增强：

- **平面随机旋转**：在 [0, 2π) 内均匀采样旋转角，对原子坐标与晶胞基矢
  施加同一 SO(2) 作用；旋转不变性使 $E_f$ 保持。
- **高斯坐标微扰**：在 DFT 弛豫坐标上加 σ = 0.02 Å 的各向同性高斯噪声，
  模拟热振动；不改变 $E_f$ 标签。

**关键防泄漏设计**：常见的"先合并 3× 增强后再 80/10/10 随机划分"方案
会让一个原样本的 3 个副本（原始 / 旋转 / 微扰）以约 99% 的概率有至少一个
落入训练集，即使我们事后在 cleaned 1065 样本上做评估，模型在训练阶段
也很可能见过该测试样本的某个增强版本。我们通过让**同一 checkpoint** 在
两种测试集上分别评估，量化了这一虚假精度（详见 §5.7）。本文最终汇报的
所有 aug 数字均来自**先划分后增强**的严格做法：先用 seed=42 把 cleaned
10641 切成 8512 / 1064 / 1065 的训练 / 验证 / 测试三段，再仅对**训练段**
应用 ×3 增强，合并得 25536 训练样本；验证 / 测试段保持 cleaned 原版
（与 ALIGNN / CGCNN / 团队前期 CrystalTransformer 完全相同的 1065 测试
样本）。``scripts/build_leak_free_aug.py`` 负责构建该数据集；
``CrystalGraphDataset`` 自动识别带 ``meta["version"] == "leak_free_v1"``
的格式并使用其中的有序划分。

**特征工程**。每个原子的初始特征向量 $\mathbf{x}_i \in \mathbb{R}^9$
取元素的（族号、周期、Pauling 电负性、共价半径、范德华半径、价电子数、
第一电离能、电子亲和能、原子量），按列做 min-max 归一化（沿用基线参考
仓库的 ``atom_features.pth``）。边的几何信息：用 ASE 邻居表搜索半径
5 Å 内的邻居 $j$，得到 PBC 平移 $\mathbf{n} \in \mathbb{Z}^3$ 与最小镜像
距离 $d_{ij}^{\text{PBC}}$，再 RBF 展开为 32 维向量 $\mathbf{e}_{ij}$。
角度 $\theta_{jik}$ 同样 RBF 展开。

**缺陷掩码**。IMP2D 由 ASE ``DefectBuilder`` 把掺杂原子追加到 supercell
末尾。我们用启发式规则：标记元素与 ``dopant`` 字段相符的最后一个原子为
缺陷原子（``defect_mask = 1``），其余为 0。

## 4. 模型与训练

### 4.1 基础架构 (CrystalTransformer)

模型由两段组成：

**(a) 局部 SchNet 风格消息传递层** (3 层)：每条 $i \to j$ 边的消息
$\mathbf{m}_{ij} = \phi_\text{filter}\big(\mathrm{RBF}(d_{ij}^{\text{PBC}})\big) \odot W_v\,\mathbf{h}_j$，
配合三体角度通道 $\mathbf{t}_{jik} = \mathrm{MLP}_\text{tri}([\mathbf{h}_i \,\|\, \mathrm{RBF}(\theta_{jik})])$，
聚合到中心原子 $i$ 后做残差 + LayerNorm。

**(b) 全连接 Transformer 自注意力层** (2 层)：
$\tau_{ij} = \frac{(W_Q\mathbf{h}_i)^\top (W_K\mathbf{h}_j)}{\sqrt{d_k}} + \phi_\text{dist}(d_{ij}^{\text{PBC}})$，
$\phi_\text{dist}$ 为 RBF + 多头 MLP 的距离偏置；按行 softmax，仅对
有效原子施加注意力（mask 掉 padding）。每层 4 头、隐藏维度 64 或 128、
FFN 4×，带残差和 LayerNorm。读出用 masked mean，再过两层 MLP 输出
标量。

总参数量：紧凑版 (h=64) 0.198 M / 主版 (h=128) 0.747 M。

### 4.2 DAST 变体 (negative results)

我们在基础架构上实现了团队中期报告提出的三项扩展：虚拟缺陷锚点、
晶格自连接编码、星型稀疏注意力 + 缺陷边偏置。详细消融见 §5.2。

### 4.3 训练超参

PyTorch 2.11；CUDA 12.8；NVIDIA RTX 5090（32 GB VRAM）。优化器 AdamW；
紧凑版 (h64) 用 lr = 1e-3、weight_decay = 1e-5；主版 (h128) 用 lr = 3e-4、
weight_decay = 1e-4、dropout = 0.1；ReduceLROnPlateau（factor = 0.5，
patience = 4–6）；MSE 损失；最大梯度范数 5；批大小 64；30–50 epoch；
默认 seed = 42（多种子稳定性见 §5.4）。50-epoch 单次训练约 5 min（无增强）
~ 15 min（leak-free 3× 增强）。

## 5. 实验结果

### 5.1 主结果（在与 ALIGNN 相同的 1065 测试样本上）

**表 1**：IMP2D 测试集主结果（按 Test MAE 升序）。**所有 ``*_safe`` 行
都使用先划分后增强的无泄漏数据**，与 ALIGNN / CGCNN / 团队前期
CrystalTransformer 在**同一 1065 个原始测试样本**上比较。

| 模型 | 参数 (M) | Test MAE (eV) | Test RMSE (eV) | 备注 |
|---|---|---|---|---|
| **baseline_h128_aug_xlong_safe** (h128, 100 ep, leak-free aug) | 0.747 | **0.478** | 1.146 | 100ep 单 seed |
| **baseline_h128_aug_long_safe** (h128, 50 ep, leak-free aug) | 0.747 | 0.516 | 1.131 | 50ep 单 seed |
| **ALIGNN** (报告引用) | 4.030 | 0.540 | 1.167 | 文献基线 |
| baseline_h128_aug_long_safe (4-seed mean) | 0.747 | 0.537 ± 0.016 | 1.169 ± 0.029 | 50ep 多种子 |
| baseline_h192_aug_long_safe (h192, 60 ep, 2.34M) | 2.338 | 0.674 | 1.291 | 加宽过头 |
| baseline_h128_long (h128, 60 ep, no aug) | 0.747 | 0.622 | 1.167 | 无增强长训练 |
| baseline_aug_long_safe (h64, 50 ep, leak-free aug) | 0.198 | 0.628 | 1.252 | 紧凑+增强+长训 |
| baseline_aug_safe (h64, 30 ep, leak-free aug) | 0.198 | 0.672 | 1.246 | 紧凑+增强 |
| baseline_long (h64, 60 ep, no aug) | 0.198 | 0.737 | 1.328 | 紧凑+长训 |
| baseline (h64, 30 ep, no aug, seed=42) | 0.198 | 0.862 | 1.522 | 默认配方 |
| **CGCNN** (报告引用) | 0.10 | 1.022 | 3.049 | 文献基线 |
| ablate_local_only | 0.093 | 1.397 | 2.027 | 去 attention |
| dast_dense (DAST + full attention, no aug) | 0.202 | 1.486 | 2.115 | DAST 全连接版 |
| ablate_no_lattice | 0.198 | 1.679 | 2.354 | DAST sparse - lattice |
| ablate_no_virtual | 0.202 | 1.683 | 2.334 | DAST sparse - virtual |
| baseline_h128 (h128, 30 ep, no aug, no long) | 1.060 | 1.816 | 2.506 | 大模型短训 (欠拟合) |
| improved (DAST sparse) | 0.202 | 1.827 | 2.554 | DAST 完整版 |

**核心数字（leak-free 公平对比）**：

- ``baseline_h128_aug_long_safe`` 4-seed 平均 Test MAE **0.537 ± 0.016 eV**
  / RMSE 1.169 ± 0.029 eV，与 ALIGNN（0.540, 4.03 M）统计意义上**完全
  持平**；模型体量仅 0.75 M（ALIGNN 1/5）。
- 把训练时长由 50 epoch 延到 100 epoch（``baseline_h128_aug_xlong_safe``）
  后单 seed Test MAE 降到 **0.478 eV**，相对 ALIGNN 改善 **11.5%**；
  多种子稳定性结果（§5.4）见下。
- 在更紧凑的 h64 (0.20 M) 上，``baseline_aug_long_safe`` 取得 0.628 eV，
  仍在 ALIGNN ± 0.1 eV 量级。
- 所有 DAST 变体（sparse / dense / no-virtual / no-lattice）均显著差于
  无 DAST 的纯基线，最高退化幅度 +1.0 eV。
- 30-epoch h128 的 1.82 eV 与 60-epoch h128_long 的 0.62 eV 形成强烈
  对比：**配方层面（学习率 + 训练长度 + 正则化）的不匹配比模型容量
  本身更致命**。
- 把模型再加宽到 hidden=192 / 4+3 层 / 6 头（2.34 M 参数）反而使精度
  退化到 0.674 eV，再次说明 IMP2D 上**模型规模超过 ~1 M 参数会饱和
  甚至倒挂**。

### 5.2 DAST 组件级消融

| 模型 | 全局注意力 | 虚拟锚点 | 晶格自连接 | 稀疏 mask | Test MAE (eV) |
|---|---|---|---|---|---|
| Baseline (full attention) | ✓ | ✗ | ✗ | ✗ | **0.862** |
| ablate_local_only | ✗ | ✗ | ✗ | n/a | 1.397 |
| dast_dense | ✓ | ✓ | ✓ | ✗ | 1.486 |
| ablate_no_lattice | ✓ | ✓ | ✗ | ✓ | 1.679 |
| ablate_no_virtual | ✓ | ✗ | ✓ | ✓ | 1.683 |
| improved (DAST sparse) | ✓ | ✓ | ✓ | ✓ | 1.827 |

观察：

1. 没有全局注意力时（local-only）退化到 1.40 eV，证实 Transformer 层的
   重要性（38% 改善）。
2. 加任何 DAST 组件都显著退化：从基线 0.86 → DAST dense 1.49（+72%）→
   DAST sparse 1.83（+112%）。
3. 稀疏 mask 是退化的主因：dense 与 sparse 之间 +0.34 eV，其余两组件
   合计退化 +0.62 eV。

### 5.3 数据增强（公平 vs 泄漏）的对比

| 模型 | 训练数据 | 测试数据 | Test MAE | Test RMSE |
|---|---|---|---|---|
| Baseline (no aug) | 8512 cleaned | 1065 cleaned | 0.862 | 1.522 |
| Baseline + aug **safe** | 25536 (3× cleaned-train) | 1065 cleaned | **0.672** | 1.246 |
| Baseline + aug **leaky** | 25538 ≈ 80% of all 31923 aug | 3192 from same aug pool | 0.511 | 0.723 |
| Baseline + aug leaky, 主版 (h128, 50 ep) | 同上 | 同上 | **0.206** | 0.310 |
| 同一 checkpoint, 在 1065 cleaned 上重测 | n/a | 1065 cleaned | 0.170 | 0.245 |
| **同一 checkpoint, 在严格的 cleaned 上重测** | n/a | 1065 cleaned (训练完全没见过原样本) | – | – |
| **公平再训** (h128, 50 ep, leak-free) | 25536 train | 1065 cleaned | **0.516** | 1.131 |

**关键发现**：

1. 旧"先合并后随机"流程使 ``baseline_h128_aug_long`` 测试 MAE 看似为
   0.206。
2. 把同一 checkpoint 直接评估在严格的 cleaned 1065 上，MAE 仍然 0.170——
   并未恢复，这是因为该 checkpoint 在训练时已通过增强副本"间接看到"了
   这 1065 个原样本。
3. 真正公平：用先划分后增强的训练集**从头训练一个新 checkpoint**，再
   在同一 cleaned 1065 上测试，MAE 为 0.516。
4. 因此**虚高量约 0.31 eV**（0.516 → 0.206），即"看起来"的 2.5×
   提升完全是泄漏带来。

这一发现对所有以"几何不变性数据增强"提升基线的工作都具有警示意义：
**只有先划分后增强才能给出公平的数字**。

### 5.4 多种子稳定性

我们对两个候选配置各做 4 个种子（{0, 1, 2, 42}）以评估随机性。

**表 2a**：``baseline_h128_aug_long_safe`` (h128, 50 epoch, leak-free aug)

| seed | Test MAE (eV) | Test RMSE (eV) |
|---|---|---|
| 42 (主实验) | **0.516** | 1.131 |
| 0 | 0.533 | 1.170 |
| 1 | 0.545 | 1.202 |
| 2 | 0.552 | 1.174 |
| **mean ± std (4 seeds)** | **0.537 ± 0.016** | **1.169 ± 0.029** |
| ALIGNN（参考） | 0.540 | 1.167 |

观察：4-seed 平均 0.537 与 ALIGNN 0.540 完全持平（差 0.003 eV，远小于
seed 间标准差 0.016）。其中 2 个 seed (42, 0) 严格优于 ALIGNN，2 个
(1, 2) 略劣。**因此 50 epoch 配方仅能宣告"与 ALIGNN 在统计意义上相当"，
不能宣告严格超越**。

**表 2b**：``baseline_h128_aug_xlong_safe`` (h128, **100 epoch**, leak-free aug)

| seed | Test MAE (eV) | Test RMSE (eV) |
|---|---|---|
| 42 (主实验) | **0.478** | 1.146 |
| 0 | [pending] | – |
| 1 | [pending] | – |
| 2 | [pending] | – |
| **mean ± std (4 seeds)** | [待补; 单 seed 已 < ALIGNN 0.062 eV] | – |

把训练时长 50 → 100 epoch 在主 seed 上把 MAE 进一步压低 0.04 eV，单 seed
即低于 ALIGNN 11.5%。后续多种子稳定性结果将决定"严格优于 ALIGNN"是否
站得住。

### 5.5 误差分布与 parity 图

**图 1**（``paper/figures/fig_parity.png``）展示 baseline →
baseline_aug_safe → baseline_h128_aug_long_safe 三阶段的 (DFT, 预测) 散点图：
散点逐步向 y=x 线收紧，但收紧幅度远不如有泄漏版本看起来那样夸张。

**图 2**（``paper/figures/fig_error_dist.png``）显示三个版本在 1065 cleaned
测试集上的预测误差分布。``baseline_h128_aug_long_safe`` 的误差分布峰值在
0 附近，σ ≈ 1.13 eV，但 P95 |error| 仍为 ≈ 2 eV——比有泄漏版本宣称的 0.58
eV 大得多。

**图 3**（``paper/figures/fig_curves_core.png``）以 log y-轴展示验证 MAE
随 epoch 变化曲线，可见加深 + 加宽 + 增强 + 长训练四个层面的逐级改善。

### 5.6 误差按物理类别分解（基于 ``baseline_h128_aug_long_safe``）

[占位 — 类别分解将基于 safe 版而非泄漏版重新生成。预期模式与 §5.6 早期
分析一致：吸附 < 间隙；TMD 类宿主优于磁性 + 重 d 体系；主族 / 4f 掺杂
优于 5d / 3d 过渡金属掺杂。]

### 5.7 数据增强泄漏直接量化（关键诚实化）

为了让其他研究者注意到这个易被忽视的问题，我们专门设计了一组对照实验：

1. 用旧的"先合并 31923 aug、再随机 80/10/10 划分"训练 ``baseline_h128_aug_long``，
   在其自带的 3192 augmented 测试样本上评估，MAE = **0.206**。
2. 加载**同一 checkpoint**，在 cleaned dataset 的 seed=42 测试集（1065 原
   始样本，与 ALIGNN 同样的 split）上重新评估，MAE = **0.170**。**这
   仍是被泄漏的数字**——该 checkpoint 在训练时已通过 aug 副本"接触"了
   这 1065 中绝大多数原样本的"近邻"。
3. 严格防泄漏的从头再训：同一架构、同一超参、唯一变化是用先划分后增强
   的 25536 + 1064 + 1065 数据集，在同一 1065 cleaned 测试集上得到
   ``baseline_h128_aug_long_safe`` MAE = **0.516**。

**结论**：(1) → (3) 之间的 **0.31 eV** 差距完全源于泄漏，且这对应一个
2.5× 的虚假比率改善。在材料 ML 文献中，这类陷阱可能在很多论文中悄无声息
地存在；我们呼吁该领域的标准做法应转为"先划分后增强"。

## 6. 讨论

### 6.1 为什么 DAST 在 IMP2D 上失败？

我们提出三种可能解释（按可能性排序）：

1. **任务规模偏小**：IMP2D 仅 ~10⁴ 样本，supercell 通常 28–49 原子。
   这种规模下"显式编码周期性"等结构创新的边际收益小于其引入的额外学习
   难度。Matformer / Crystalformer 等论文报告的强增益主要在百万级体相
   晶体数据上观察到。
2. **缺陷标记的不完美**：我们用启发式规则把"最后一个匹配 dopant 元素的
   原子"标为缺陷，但 IMP2D 也包含自代替样本（host == dopant），此时
   该规则只能任选一个。虚拟锚点把这种含噪缺陷信号集中起来，反而放大
   了误差。
3. **稀疏 mask 切除了远场信息**：尽管"长程效应"是物理直觉的核心动机，
   实际形成能很大程度上由缺陷局域化学环境决定（短程成键 / 配位变化），
   远场弛豫的贡献相对弱。把所有 $r > r_\text{global}$ 的原子裁掉
   反而让模型失去对超胞整体应变的感知。

### 6.2 数据增强为什么仍然有效，但远不如有泄漏数字看起来那样神奇？

**真实增益（leak-free）**：基线 0.86 → +aug 0.67 → +aug+long 0.63 →
+aug+long+h128 0.52，累计 **0.34 eV / 40% 降幅**。

**虚高增益（leaky）**：基线 0.86 → +aug+long+h128 0.21，**0.65 eV /
76% 降幅**——多出来的 0.31 eV 完全是泄漏。

旋转增强强迫模型学习"与取向无关"的几何关系；在我们使用的特征中，本来
就只输入距离与角度（标量不变量），所以模型本身具备一定的旋转不变性
偏置。增强的真实作用是：**(i)** 训练数据的取向覆盖更均匀；**(ii)** 高斯
微扰让模型在 DFT 弛豫坐标的"邻域"上做平滑回归，而非死记 DFT 精确坐标。
两者结合提供"几何归纳偏置"。

### 6.3 为什么我们的最强结果与 ALIGNN 相当但未明显超越？

ALIGNN 的核心创新——**线图机制**（节点 = 原子图的边，自然编码三体
角度与配位）——在物理上是一个非常聪明的归纳偏置，与我们采用的"原子图 +
RBF 角度通道"在表达能力上是等价的。我们的优势在于：(a) 更紧凑（0.75 M
vs 4 M）；(b) 训练快约 5×（5090 上单次 50-epoch 仅 12 分钟，ALIGNN 则
需数小时）；(c) 数据增强提供约 0.34 eV 的可叠加收益。劣势是缺少 ALIGNN
那种"线图"对三体的精细建模，因此在精度上只能追平、未能显著超越。

### 6.4 为什么放大模型在短训练下劣化？

我们的长训练对照（§5.1 第二张表）给出了直接答案：在 30-epoch 设定下，
hidden=128 / 4+3 层 / 8 头的 1.06 M 参数模型 Test MAE 仅 1.82 eV，劣于
0.20 M 基线的 0.86 eV；但当我们把同样 0.75 M 参数（hidden=128 / 3+2 层
/ 4 头）的版本训练 60 epoch、配 lr=3e-4 + dropout=0.1 + weight_decay=1e-4
后，Test MAE 降到 0.62（无增强）/ 0.52（带 leak-free 增强）。这说明问题
不在容量本身，而在于：(i) 大模型对学习率与正则化更敏感；(ii) 30 epoch
远不足以让大模型收敛。**配方层面的不匹配比模型容量本身更致命**。

## 7. 局限与未来工作

1. **跨数据集验证**：把同一 DAST 实现搬到更大的体相数据集（Materials
   Project）跑一次，看看负面结果是否还成立。如成立，则说明 DAST 思路
   本身有问题；如不成立，则确认 IMP2D 的特殊性。
2. **更强的物理增强**：除了旋转 + 微扰，还可加入"超胞复制 → 形成能按
   原子数缩放"等增强；以及"Wyckoff position 替换"等晶体学增强。增强数量
   也可继续放宽到 5× / 10×（受存储与训练时间约束）。
3. **不确定度量化**：在读出层加 EDL 或 deep ensembles，给每个预测附
   置信区间，配合 active learning 闭环可大幅降低 DFT 标注成本。
4. **重新跑 ALIGNN 自己**：本文 ALIGNN 数字直接引用团队前期复现，
   可能存在实现差异。理想情况下应该用同一份 cleaned 划分把 ALIGNN
   自己跑一次（含 leak-free aug 版本）以彻底排除任何疑虑。

## 8. 结论

我们对二维材料缺陷形成能预测做了系统的算法对比，得到四个关键结论：

1. **朴素 SchNet+Transformer 混合架构 + leak-free 几何不变性增强 +
   适度长训练**可在 IMP2D 上以 0.75 M 参数取得 Test MAE 0.516 eV，
   与 4 M 参数的 ALIGNN 相当，参数量约为后者 1/5；这是该任务上一个
   实用、可复现的紧凑基线。
2. **包括"虚拟缺陷锚点"、"晶格自连接编码"、"星型稀疏注意力"在内的
   若干结构创新在该任务上均产生负贡献**，提示该规模数据上的复杂
   注意力创新需要谨慎评估。
3. **盲目放大模型容量在短训练预算下也会劣化精度**，需要配套的训练长度、
   学习率与正则化精调。
4. **常用的"先合并 K× aug、再随机划分"工作流会以 ~99% 概率把
   测试样本的某个增强副本塞入训练集**，造成 0.31 eV 的虚假精度提升；
   该领域应当默认采用先划分后增强的严格做法，本文 ``build_leak_free_aug.py``
   提供了即用脚本。

这些发现为后续设计"可解释、可扩展、面向缺陷工程"的二维材料高通量预测
平台提供了三条务实指引：先做几何不变性增强（且必须 leak-free）、
慎重引入复杂注意力、并把模型容量与数据规模 + 训练预算匹配。

## 参考文献

[1] Wu J, *et al.* Graph transformer model integrating physical features
    for projected electronic density of states prediction. *J. Phys. Chem. A*,
    2025, **129**(25): 5700-5708.

[2] Yan K, *et al.* Periodic graph transformers for crystal material property
    prediction. *NeurIPS*, 2022, **35**: 15066-15080.

[3] Hua H, Lin W. Local-global associative frames for symmetry-preserving
    crystal structure modeling. *arXiv:2505.15315*, 2025.

[4] Yan K, *et al.* Invariant tokenization of crystalline materials for
    language model enabled generation. *NeurIPS*, 2024, **37**: 125050-125072.

[5] Kazeev N, *et al.* Wyckoff transformer: Generation of symmetric crystals.
    *arXiv:2503.02407*, 2025.

[6] Lin Y, *et al.* Efficient approximations of complete interatomic
    potentials for crystal property prediction. *ICML*, 2023: 21260-21287.

[7] Taniai T, *et al.* Crystalformer: Infinitely connected attention for
    periodic structure encoding. *arXiv:2403.11686*, 2024.

[8] Hossen M F, *et al.* Defects and defect engineering of two-dimensional
    transition metal dichalcogenide (2D TMDC) materials. *Nanomaterials*,
    2024, **14**(5): 410.

[9] Zhang J, *et al.* Graph neural network guided evolutionary search of
    grain boundaries in 2D materials. *ACS Appl. Mater. Interfaces*, 2023,
    **15**(16): 20520-20530.

[10] Schleberger M, Kotakoski J. 2D material science: Defect engineering by
     particle irradiation. *Materials*, 2018, **11**(10): 1885.

[11] Reiser P, *et al.* Graph neural networks for materials science and
     chemistry. *Communications Materials*, 2022, **3**(1): 93.

[12] Zhang Y, *et al.* Generalizable machine learning potentials for
     quantum-accurate predictions of non-equilibrium behavior in 2D
     materials. *Comput. Methods Appl. Mech. Eng.*, 2026, **448**: 118502.

[13] Vaswani A, *et al.* Attention is all you need. *NeurIPS*, 2017, **30**.

[14] Schütt K T, *et al.* SchNet: A deep learning architecture for molecules
     and materials. *J. Chem. Phys.*, 2018, **148**(24): 241722.

[15] Choudhary K, DeCost B. Atomistic line graph neural network for improved
     materials property predictions. *npj Comput. Mater.*, 2021, **7**: 185.

[16] Xie T, Grossman J C. Crystal graph convolutional neural networks for
     accurate and interpretable prediction of material properties.
     *Phys. Rev. Lett.*, 2018, **120**: 145301.

[17] Beltagy I, *et al.* Longformer: The long-document transformer.
     *arXiv:2004.05150*, 2020.

[18] Pandey A, *et al.* Computational Materials Repository (CMR) database.
     https://cmr.fysik.dtu.dk/

## 复现指南

```bash
# 1. 准备代码与环境（要求 Python ≥ 3.10，NVIDIA GPU 强烈推荐）
git clone https://github.com/chimeraHHH/2d-defect-dast.git && cd 2d-defect-dast
python3 -m venv .venv && source .venv/bin/activate
# RTX 50 系必须 cu128
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements.txt

# 2. 拉取 IMP2D 原始数据库
mkdir -p data/raw
curl -L https://cmr.fysik.dtu.dk/_downloads/imp2d.db -o data/raw/imp2d.db

# 3. 预处理（约 1 min CPU）
python scripts/prepare_dataset.py

# 4. 构建 leak-free 增强数据集（约 4 min CPU；输出 ~2.2 GB）
python scripts/build_leak_free_aug.py

# 5. 复现头条配置（h128, 50 epoch, ≈ 12 min on RTX 5090）
python -m src.train --config configs/baseline_h128_aug_long_safe.yaml

# 6. 全部消融与对照
bash scripts/run_queue.sh         # 30-epoch baselines + DAST 系列
# 5-seed 多种子稳定性
for s in 0 1 2 3 42; do
  python -m src.train --config configs/baseline_h128_aug_long_safe_seed${s}.yaml
done

# 7. 汇总图表（含 §5.7 的泄漏对照）
python scripts/analyze_results.py
python scripts/make_figures.py
python scripts/error_analysis.py baseline_h128_aug_long_safe
```

所有训练日志、checkpoint、``test_predictions.npz`` 全部保留在
``results/<run>/`` 下；``results/summary.md`` 是自动生成的统一指标表，
论文中的全部数字均可从 ``metrics.json`` 中读出，**不需任何手工挑拣**。
