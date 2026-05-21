import os
import sys
import numpy as np
from scipy import stats
from ase.io import read
from ase.db import connect

ROOT_DIR = '/root/autodl-tmp/2d-defect-dast'
sys.path.append(ROOT_DIR)
from predictor import FormationEnergyPredictor

def get_final_energy(filename):
    if not os.path.exists(filename): return None
    try: return read(filename).get_potential_energy()
    except: return None

def get_mu(dopant):
    mu_s2 = os.path.join(ROOT_DIR, f"gpaw_mu_{dopant}_stage2.txt")
    mu_s1 = os.path.join(ROOT_DIR, f"gpaw_mu_{dopant}_stage1.txt")
    mu = get_final_energy(mu_s2)
    if mu is None: mu = get_final_energy(mu_s1)
    if mu is not None:
        try: return mu / len(read(mu_s2 if os.path.exists(mu_s2) else mu_s1))
        except: return None
    return None

samples_map = {}
for db_file, start_idx in [('new_samples.db', 40), ('new_samples_batch2.db', 50), ('new_tmd_samples.db', 60), ('new_tmd_samples_20.db', 70)]:
    db_path = os.path.join(ROOT_DIR, db_file)
    if os.path.exists(db_path):
        db = connect(db_path)
        for i, row in enumerate(db.select()):
            samples_map[start_idx + i] = {'dopant': row.dopant, 'formula': row.formula}

model_path = os.path.join(ROOT_DIR, 'models', 'formation_energy_model.pth')
feature_path = os.path.join(ROOT_DIR, 'atom_features.pth')
predictor = FormationEnergyPredictor(model_path=model_path, feature_path=feature_path, device='cpu')

dft_vals, ml_vals = [], []
for idx, info in samples_map.items():
    dopant, formula = info['dopant'], info['formula']
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
            dl_eform = predictor.predict(read(traj_path))
            if abs(dl_eform - dft_eform) < 2.0:
                dft_vals.append(dft_eform)
                ml_vals.append(dl_eform)
        except Exception: pass

slope, intercept, r_value, p_value, std_err = stats.linregress(dft_vals, ml_vals)
print(f"Calculated R^2 = {r_value**2:.3f}")
