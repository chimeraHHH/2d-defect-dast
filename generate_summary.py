import os
from ase.io import read
from ase.db import connect

def get_final_energy(filename):
    if not os.path.exists(filename):
        return None
    try:
        atoms = read(filename)
        return atoms.get_potential_energy()
    except Exception:
        return None

def get_mu(dopant):
    mu_s2 = f"/root/autodl-tmp/2d-defect-dast/gpaw_mu_{dopant}_stage2.txt"
    mu_s1 = f"/root/autodl-tmp/2d-defect-dast/gpaw_mu_{dopant}_stage1.txt"
    mu = get_final_energy(mu_s2)
    if mu is None:
        mu = get_final_energy(mu_s1)
        
    if mu is not None:
        # Check number of atoms to get per-atom mu
        try:
            atoms = read(mu_s2 if os.path.exists(mu_s2) else mu_s1)
            return mu / len(atoms)
        except:
            return None
    return None

db1 = connect('/root/autodl-tmp/2d-defect-dast/new_samples.db')
db2 = connect('/root/autodl-tmp/2d-defect-dast/new_samples_batch2.db')

rows_40 = list(db1.select())
rows_50 = list(db2.select())

with open('/root/autodl-tmp/2d-defect-dast/recent_samples_summary.md', 'w') as f:
    f.write("# 最新计算样本结果汇总\n\n")
    f.write("本汇总包含近期通过 GPAW 进行的两批新样本计算结果（编号 40-49，及 50-59）。\n\n")
    f.write("## 计算结果表\n\n")
    f.write("| 编号 | 化学式 | 掺杂物 | 缺陷超胞能量 (en2, eV) | 本征超胞能量 (host, eV) | 掺杂物化学势 (mu, eV) | 形成能 (E_form, eV) | 状态 |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    
    def process_row(idx, row):
        en2_s2 = f"/root/autodl-tmp/2d-defect-dast/gpaw_sample_{idx}_defect_stage2.txt"
        en2_s1 = f"/root/autodl-tmp/2d-defect-dast/gpaw_sample_{idx}_defect_stage1.txt"
        host_s2 = f"/root/autodl-tmp/2d-defect-dast/gpaw_sample_{idx}_host_stage2.txt"
        host_s1 = f"/root/autodl-tmp/2d-defect-dast/gpaw_sample_{idx}_host_stage1.txt"
        
        en2 = get_final_energy(en2_s2)
        if en2 is None: en2 = get_final_energy(en2_s1)
            
        host = get_final_energy(host_s2)
        if host is None: host = get_final_energy(host_s1)
            
        mu = get_mu(row.dopant)
        
        eform = "N/A"
        if en2 is not None and host is not None and mu is not None:
            eform_val = en2 - host - mu
            eform = f"{eform_val:.4f}"
            
        status = "Completed" if eform != "N/A" else "Terminated/Incomplete"
        
        # Format strings
        en2_str = f"{en2:.4f}" if en2 is not None else "N/A"
        host_str = f"{host:.4f}" if host is not None else "N/A"
        mu_str = f"{mu:.4f}" if mu is not None else "N/A"
        
        f.write(f"| {idx} | {row.formula} | {row.dopant} | {en2_str} | {host_str} | {mu_str} | {eform} | {status} |\n")

    for i, row in enumerate(rows_40):
        process_row(40 + i, row)
        
    for i, row in enumerate(rows_50):
        process_row(50 + i, row)

print("Summary generated at /root/autodl-tmp/2d-defect-dast/recent_samples_summary.md")
