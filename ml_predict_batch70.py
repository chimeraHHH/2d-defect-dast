import os
import sys
import torch
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

def main():
    model_path = os.path.join(ROOT_DIR, 'models', 'formation_energy_model.pth')
    feature_path = os.path.join(ROOT_DIR, 'atom_features.pth')
    
    print("Loading predictor...")
    predictor = FormationEnergyPredictor(model_path=model_path, feature_path=feature_path, device='cpu')
    
    db_path = os.path.join(ROOT_DIR, 'new_tmd_samples_20.db')
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
        
    db = connect(db_path)
    rows = list(db.select())
    
    print("\n| ID | Host | Dopant | Formula | DFT E_form (eV) | ML E_form (eV) | Abs Error (eV) |")
    print("|---|---|---|---|---|---|---|")
    
    maes = []
    
    for i, row in enumerate(rows):
        idx = 70 + i
        en2_s2 = os.path.join(ROOT_DIR, f"gpaw_sample_{idx}_defect_stage2.txt")
        en2_s1 = os.path.join(ROOT_DIR, f"gpaw_sample_{idx}_defect_stage1.txt")
        host_s2 = os.path.join(ROOT_DIR, f"gpaw_sample_{idx}_host_stage2.txt")
        host_s1 = os.path.join(ROOT_DIR, f"gpaw_sample_{idx}_host_stage1.txt")
        
        en2 = get_final_energy(en2_s2)
        if en2 is None: en2 = get_final_energy(en2_s1)
            
        host = get_final_energy(host_s2)
        if host is None: host = get_final_energy(host_s1)
            
        mu = get_mu(row.dopant)
        
        if en2 is not None and host is not None and mu is not None:
            dft_eform = en2 - host - mu
            
            # Predict using the defect structure
            traj_path = en2_s2 if os.path.exists(en2_s2) else en2_s1
            try:
                structure = read(traj_path)
                ml_eform = predictor.predict(structure)
                error = abs(ml_eform - dft_eform)
                maes.append(error)
                print(f"| {idx:2d} | {row.host:4s} | {row.dopant:6s} | {row.formula:10s} | {dft_eform:15.4f} | {ml_eform:14.4f} | {error:14.4f} |")
            except Exception as e:
                print(f"| {idx:2d} | Error reading or predicting: {e} |")
        else:
            # 也可以提示哪些尚未完成
            pass
            
    if maes:
        print(f"\n**Mean Absolute Error (MAE) for these {len(maes)} samples:** {sum(maes)/len(maes):.4f} eV")
    else:
        print("\nNo completed samples found or missing chemical potentials.")

if __name__ == '__main__':
    main()
