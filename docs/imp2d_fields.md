# IMP2D 数据库字段清点（roadmap WS0.3，部分完成）

> 核对日期：2026-11-27
> 数据来源：`https://wiki.fysik.dtu.dk/cmr-files/imp2d.db`（README 里旧的 `cmr.fysik.dtu.dk/_downloads/imp2d.db` 已经 404）
> 文档：<https://cmr.fysik.dtu.dk/imp2d/imp2d.html>
> 版本：2022-07-12 初版，17 364 行

## 1. 计算设置（来自 CMR 页面）

| 步骤 | 设置 |
|---|---|
| 初始弛豫 | **VASP**，PBE，**仅 Γ 点**，自旋极化，PW 截断 600 eV，动能截断 900 eV，电子收敛 10⁻⁴，离子收敛 0.005 eV，FFT 网格 3/2 |
| 最终弛豫 | 同上，电子收敛 10⁻⁶，离子收敛 10⁻⁴ eV，FFT 网格 2 |
| 缺陷构型生成 | ASE 的 DefectBuilder 模块（**当前 ASE 3.30 中没有** `ase.build.defects`，需要找到原版本或 ASR 中的实现） |
| 工作流 | httk High-Throughput Toolkit |

**注意**：IMP2D 用的是 **VASP，不是 GPAW**。roadmap v1.1 中写成了 GPAW，v1.2 已更正。DFT 验证（WS6）的设置对齐和 Gate D 都要按 VASP 来设计。

## 2. 键值对

| 键 | 含义 | 单位 | 备注 |
|---|---|---|---|
| `host` | 宿主化学式 | | 过滤后剩 44 种宿主（CMR 页面称 53 种单层） |
| `dopant` | 杂质元素 | | |
| `defecttype` | `adsorbate` / `interstitial` | | |
| `site` | 构型生成时的位点编号 | | `ads0`–`ads5`、`int0`–`int5`，是 DefectBuilder 的枚举序号，**没有几何含义**，必须用同一个 DefectBuilder 重新生成才能对应 |
| `depth` | 深度参数 | | DefectBuilder 的输入参数，重建初始结构时需要 |
| `extension_factor` | 扩展因子 | | 同上 |
| `supercell` | 计算超胞 | | 编码形式：`331`、`441`、`221`，分别有 6255、3328、1058 个样本（过滤后） |
| `eform` | 形成能 | eV | |
| `en1` / `en2` | 第一步 / 最终总能 | eV | |
| `conv1` / `conv2` | 对应的能量收敛 | eV | |
| `hostenergy` | 宿主能量 | eV | |
| `dopant_chemical_potential` | 杂质化学势 | eV | |
| `spin` | **磁矩** | μB | **整数**，为超胞总磁矩，有正有负 |
| `host_spacegroup` | 宿主空间群 | | |
| `converged` | 弛豫是否收敛 | | |
| `name` | 体系名称 | | 例如 `NbS2_Tm_ads0` |

数据库中**没有**：初始（未弛豫）结构、弛豫轨迹、逐原子磁矩（`row.data` 为空）、pristine 宿主结构。

## 3. 磁矩标签 `spin` 的统计（过滤后 10 641 行）

过滤条件与审计链第一步相同：`converged`、`eform` 有限、|eform| ≤ 20 eV。统计脚本为 `scripts/prm/ws4_magmom_baseline.py`。

- **全部是整数**。分布：|μ| = 1 的有 3658 个，0 有 2209 个，2 有 1796 个，3 有 1005 个，4 有 607 个，5 有 559 个，6 有 322 个，≥ 7 的较少。
- **只有约 21% 无磁**，吸附位 17.9%，间隙位 26.2%。也就是说"有磁"是多数情况，"|μ| > 0"这样的筛选判据几乎没有区分度。
- 负值约占 4%，按 |μ| 处理。
- **本征磁性宿主**：Cr₂I₆（CrI₃）的总磁矩中位数为 25 μB，主要来自宿主本身。磁矩任务中先排除这个宿主。Nb₄C₃、Mo₂CO₂ 等宿主的中位数也偏高（4–5 μB），需要核查宿主本身是否带磁。
- **电子数奇偶规则**：整数总磁矩的奇偶性应当与超胞总电子数（∑Z）的奇偶性相同。数据中 **96.1%** 的样本满足这一点。不满足的 4% 集中在 Cr₂I₆、PbI₂、TiO₂、Hf₂Te₆ 等宿主，推测是金属性或部分占据导致小数磁矩被取整，需要查原文确认 `spin` 是取整值还是固定自旋态（NUPDOWN）计算的结果。

## 4. 待确认

- [ ] `spin` 是否为取整后的总磁化，还是固定总磁矩计算的结果（查 Davidsson et al. 2023 原文和 SI）。
- [ ] 各宿主 pristine 状态的磁矩（可从 C2DB 获取），用于定义"缺陷诱导磁矩" Δμ = μ(缺陷) − μ(宿主)。
- [ ] 金属性宿主（NbS₂、NbSe₂、MXene 等）上的磁矩是否有物理意义；仅 Γ 点采样对金属体系的影响。
- [ ] DefectBuilder 的原始实现和版本，WS1.1 位点生成器要依赖它。
