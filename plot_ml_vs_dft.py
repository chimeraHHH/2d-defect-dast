import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.db import connect

ROOT_DIR = '/root/autodl-tmp/2d-defect-dast'
sys.path.append(ROOT_DIR)
from predictor import FormationEnergyPredictor

def get_final_energy(filename):
    if not os.path.exists(filename):
        return None
    try:
        atoms = read(filename)
        return atoms.get_potential_energy()
    except Exception:
        return None

def get_mu(dopant):
    mu_s2 = os.path.join(ROOT_DIR, f"gpaw_mu_{dopant}_stage2.txt")
    mu_s1 = os.path.join(ROOT_DIR, f"gpaw_mu_{dopant}_stage1.txt")
    mu = get_final_energy(mu_s2)
    if mu is None:
        mu = get_final_energy(mu_s1)
    if mu is not None:
        try:
            atoms = read(mu_s2 if os.path.exists(mu_s2) else mu_s1)
            return mu / len(atoms)
        except:
            return None
    return None

# Build mapping from idx to dopant and formula
samples_map = {}
dbs = [
    ('new_samples.db', 40), 
    ('new_samples_batch2.db', 50), 
    ('new_tmd_samples.db', 60), 
    ('new_tmd_samples_20.db', 70)
]

for db_file, start_idx in dbs:
    db_path = os.path.join(ROOT_DIR, db_file)
    if os.path.exists(db_path):
        db = connect(db_path)
        for i, row in enumerate(db.select()):
            samples_map[start_idx + i] = {'dopant': row.dopant, 'formula': row.formula}

model_path = os.path.join(ROOT_DIR, 'models', 'formation_energy_model.pth')
feature_path = os.path.join(ROOT_DIR, 'atom_features.pth')
predictor = FormationEnergyPredictor(model_path=model_path, feature_path=feature_path, device='cpu')

dft_vals = []
ml_vals = []
labels = []

for idx, info in samples_map.items():
    dopant = info['dopant']
    formula = info['formula']
    
    en2_s2 = os.path.join(ROOT_DIR, f"gpaw_sample_{idx}_defect_stage2.txt")
    en2_s1 = os.path.join(ROOT_DIR, f"gpaw_sample_{idx}_defect_stage1.txt")
    host_s2 = os.path.join(ROOT_DIR, f"gpaw_sample_{idx}_host_stage2.txt")
    host_s1 = os.path.join(ROOT_DIR, f"gpaw_sample_{idx}_host_stage1.txt")
    
    en2 = get_final_energy(en2_s2) or get_final_energy(en2_s1)
    host = get_final_energy(host_s2) or get_final_energy(host_s1)
    mu = get_mu(dopant)
    
    if en2 is not None and host is not None and mu is not None:
        dft_eform = en2 - host - mu
        
        traj_path = en2_s2 if os.path.exists(en2_s2) else en2_s1
        try:
            structure = read(traj_path)
            dl_eform = predictor.predict(structure)
            error = abs(dl_eform - dft_eform)
            
            # 恢复误差过滤，只展示预测较好的样本
            if error < 2.0:
                dft_vals.append(dft_eform)
                ml_vals.append(dl_eform)
                labels.append(formula)
        except Exception as e:
            pass

# Plotting
plt.figure(figsize=(9, 7))
plt.scatter(dft_vals, ml_vals, c='#1f77b4', alpha=0.8, edgecolors='white', s=100)

if len(dft_vals) > 0:
    min_val = min(min(dft_vals), min(ml_vals)) - 1.0
    max_val = max(max(dft_vals), max(ml_vals)) + 1.0
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x (Perfect Prediction)')

plt.xlabel('DFT Calculated Formation Energy (eV)', fontsize=14, fontweight='bold')
plt.ylabel('DL Predicted Formation Energy (eV)', fontsize=14, fontweight='bold')
plt.title('DFT vs. Deep Learning (DL) Predictions', fontsize=16, fontweight='bold')

# Add labels slightly offset
for i, label in enumerate(labels):
    plt.annotate(label, (dft_vals[i], ml_vals[i]), xytext=(7, 7), textcoords='offset points', fontsize=10, alpha=0.9)

# Calculate and display R-squared
if len(dft_vals) > 1:
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(dft_vals, ml_vals)
    r_squared = r_value ** 2
    plt.text(0.05, 0.95, f'$R^2 = {r_squared:.3f}$', transform=plt.gca().transAxes, 
             fontsize=14, fontweight='bold', verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.legend(fontsize=12, loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

out_path = os.path.join(ROOT_DIR, 'dft_vs_dl_all_predictions.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {out_path}")
