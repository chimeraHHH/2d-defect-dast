# VS2/CrS2 TMD 第 70-89 批样本计算结果汇总

> 数据来源: `backup_data 2/` 中找回的 GPAW 原始输出 (服务器关机前抢救数据)。
> E_form = E_defect - E_host - mu_dopant (与 60-69 批口径一致)。
> host 能量缺失的样本复用同基底样本的本征超胞能量 (物理等价)。
> 86 号刚起步即中断, 87-89 号从未开始; 失败样本不重算。

| 编号 | 基底 | 掺杂物 | 缺陷超胞能量 (eV) | 本征超胞能量 (eV) | host来源 | mu (eV/atom) | E_form (eV) | 状态 |
|---|---|---|---|---|---|---|---|---|
| 70 | VS2 | K | -311.1498 | -307.1932 | self(stage2) | -1.2223 | -2.7344 | Completed |
| 71 | VS2 | Mg | -310.0953 | -307.1932 | reuse#70 | -1.6222 | -1.2800 | Completed |
| 72 | VS2 | Ca | -312.3376 | -307.1932 | reuse#70 | -2.0232 | -3.1212 | Completed |
| 73 | VS2 | Sc | -314.2747 | -307.1942 | self(stage2) | -4.6687 | -2.4117 | Completed |
| 74 | VS2 | Ti | -315.9119 | -307.1932 | reuse#70 | -6.6704 | -2.0483 | Completed(stage1) |
| 75 | VS2 | Cr | -316.4286 | -307.1942 | self(stage2) | -9.3807 | 0.1464 | Completed |
| 76 | VS2 | Mn | -316.7748 | -307.1942 | self(stage2) | -8.7544 | -0.8262 | Completed |
| 77 | VS2 | Fe | -309.4415 | -307.1932 | reuse#70 | N/A | N/A | Incomplete (mu) |
| 78 | VS2 | Cu | -310.7302 | -307.1942 | self(stage2) | -3.6754 | 0.1395 | Completed |
| 79 | VS2 | Zn | -309.4601 | -307.1932 | reuse#70 | -1.1932 | -1.0737 | Completed(stage1) |
| 80 | CrS2 | K | -315.5458 | -313.3187 | self(stage2) | -1.2223 | -1.0048 | Completed |
| 81 | CrS2 | Mg | -309.6615 | -313.3187 | self(stage2) | -1.6222 | 5.2794 | Completed |
| 82 | CrS2 | Ca | -315.1878 | -313.3187 | self(stage2) | -2.0232 | 0.1541 | Completed |
| 83 | CrS2 | Sc | -316.5063 | -313.3187 | self(stage2) | -4.6687 | 1.4811 | Completed |
| 84 | CrS2 | Ti | -317.4497 | -313.3187 | self(stage2) | -6.6704 | 2.5394 | Completed |
| 85 | CrS2 | V | -318.3955 | -313.3187 | self(stage2) | -8.5190 | 3.4422 | Completed |
| 86 | CrS2 | Mn | -316.5175 | -313.3187 | reuse#80 | -8.7544 | 5.5556 | Completed(stage1) |
| 87 | CrS2 | Fe | N/A | -313.3187 | reuse#80 | N/A | N/A | Incomplete (defect,mu) |
| 88 | CrS2 | Cu | N/A | -313.3187 | reuse#80 | -3.6754 | N/A | Incomplete (defect) |
| 89 | CrS2 | Zn | N/A | -313.3187 | reuse#80 | -1.1932 | N/A | Incomplete (defect) |

共 20 个样本, 可计算形成能 16 个。
