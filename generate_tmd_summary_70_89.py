#!/usr/bin/env python3
"""
从 backup_data 2/ 中找回的 GPAW 原始输出提取第 70-89 批
(new_tmd_samples_20.db: VS2/CrS2 x 10 种掺杂) 的能量并计算形成能。

- 数据来源: backup_data 2/gpaw_sample_{idx}_{defect,host}_stage{2,1}.txt
- 化学势:   backup_data 2/gpaw_mu_{EL}_stage{2,1}.txt (每原子能量)
- 失败/未完成的样本直接标 N/A, 不重算 (用户指示)
- host 能量缺失时, 回退使用同批次同基底其他样本的 host 能量
  (本征超胞对同一基底完全相同, 物理上等价)

形成能约定与此前批次一致: E_form = E_defect - E_host - mu_dopant
输出: backup_data/tmd_samples_70_89_summary.md
"""
import os
import re
import sqlite3

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup_data 2")
DB = os.path.join(BASE, "new_tmd_samples_20.db")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "backup_data", "tmd_samples_70_89_summary.md")
START_IDX = 70  # run_tmd_samples_70_20.log => 批次从 70 号开始


def last_extrapolated(path):
    """返回 GPAW txt 输出中最后一次 'Extrapolated:' 能量 (eV)。"""
    if not os.path.exists(path):
        return None
    energy = None
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.search(r"Extrapolated:\s+(-?\d+\.\d+)", line)
            if m:
                energy = float(m.group(1))
    return energy


def natoms_in_output(path):
    """从 GPAW txt 中解析原子数 (positions 块行数)。"""
    if not os.path.exists(path):
        return None
    n = None
    with open(path, errors="ignore") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("Positions:"):
            cnt = 0
            for l in lines[i + 1:]:
                if re.match(r"\s*\d+\s+[A-Z][a-z]?\s+-?\d", l):
                    cnt += 1
                else:
                    break
            if cnt:
                n = cnt
    return n


def sample_energy(idx, kind):
    """kind in {defect, host}; 优先 stage2, 回退 stage1。"""
    for stage in ("stage2", "stage1"):
        p = os.path.join(BASE, f"gpaw_sample_{idx}_{kind}_{stage}.txt")
        e = last_extrapolated(p)
        if e is not None:
            return e, stage
    return None, None


def mu_per_atom(el):
    for stage in ("stage2", "stage1"):
        p = os.path.join(BASE, f"gpaw_mu_{el}_{stage}.txt")
        e = last_extrapolated(p)
        if e is not None:
            n = natoms_in_output(p) or 1
            return e / n
    return None


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    rows = {}
    for sid, key, val in cur.execute(
            "SELECT id, key, value FROM text_key_values"):
        rows.setdefault(sid, {})[key] = val
    # 化学式(从 systems 表拿不到简式, 用 host+dopant 描述即可)
    con.close()

    records = []
    host_cache = {}   # host 材料 -> (energy, 来源样本号)
    for i, sid in enumerate(sorted(rows)):
        idx = START_IDX + i
        host_mat = rows[sid].get("host", "?")
        dopant = rows[sid].get("dopant", "?")

        en2, en2_st = sample_energy(idx, "defect")
        host_e, host_st = sample_energy(idx, "host")
        host_src = f"self({host_st})" if host_e is not None else None
        if host_e is None and host_mat in host_cache:
            host_e, src_idx = host_cache[host_mat]
            host_src = f"reuse#{src_idx}"
        if host_e is not None and host_mat not in host_cache and host_st == "stage2":
            host_cache[host_mat] = (host_e, idx)

        mu = mu_per_atom(dopant)

        if en2 is not None and host_e is not None and mu is not None:
            eform = en2 - host_e - mu
            status = "Completed" if en2_st == "stage2" else "Completed(stage1)"
        else:
            eform = None
            missing = []
            if en2 is None: missing.append("defect")
            if host_e is None: missing.append("host")
            if mu is None: missing.append("mu")
            status = "Incomplete (" + ",".join(missing) + ")"

        records.append((idx, host_mat, dopant, en2, host_e, host_src, mu,
                        eform, status))

    # 第二遍: host_cache 补全 (前面样本 host 缺失但后面样本算出了同基底 host)
    for j, r in enumerate(records):
        idx, host_mat, dopant, en2, host_e, host_src, mu, eform, status = r
        if host_e is None and host_mat in host_cache:
            host_e, src_idx = host_cache[host_mat]
            host_src = f"reuse#{src_idx}"
            if en2 is not None and mu is not None:
                eform = en2 - host_e - mu
                status = "Completed(host-reused)"
            records[j] = (idx, host_mat, dopant, en2, host_e, host_src, mu,
                          eform, status)

    fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# VS2/CrS2 TMD 第 70-89 批样本计算结果汇总\n\n")
        f.write("> 数据来源: `backup_data 2/` 中找回的 GPAW 原始输出 "
                "(服务器关机前抢救数据)。\n")
        f.write("> E_form = E_defect - E_host - mu_dopant (与 60-69 批口径一致)。\n")
        f.write("> host 能量缺失的样本复用同基底样本的本征超胞能量 (物理等价)。\n")
        f.write("> 86 号刚起步即中断, 87-89 号从未开始; 失败样本不重算。\n\n")
        f.write("| 编号 | 基底 | 掺杂物 | 缺陷超胞能量 (eV) | 本征超胞能量 (eV) "
                "| host来源 | mu (eV/atom) | E_form (eV) | 状态 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in records:
            idx, host_mat, dopant, en2, host_e, host_src, mu, eform, status = r
            f.write(f"| {idx} | {host_mat} | {dopant} | {fmt(en2)} | "
                    f"{fmt(host_e)} | {host_src or 'N/A'} | {fmt(mu)} | "
                    f"{fmt(eform)} | {status} |\n")
        done = sum(1 for r in records if r[7] is not None)
        f.write(f"\n共 {len(records)} 个样本, 可计算形成能 {done} 个。\n")

    print(f"written: {OUT}")
    for r in records:
        print(r)


if __name__ == "__main__":
    main()
