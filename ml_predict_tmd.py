import os
import sys
import torch
from ase.db import connect

ROOT_DIR = '/root/autodl-tmp/2d-defect-dast'
sys.path.append(ROOT_DIR)
from predictor import FormationEnergyPredictor

def main():
    model_path = os.path.join(ROOT_DIR, 'models', 'formation_energy_model.pth')
    feature_path = os.path.join(ROOT_DIR, 'atom_features.pth')
    
    print("Loading predictor...")
    predictor = FormationEnergyPredictor(model_path=model_path, feature_path=feature_path, device='cpu')
    
    db = connect('/root/autodl-tmp/2d-defect-dast/new_tmd_samples.db')
    
    print("\n| ID | Host | Dopant | Formula | ML Predicted E_form (eV) |")
    print("|---|---|---|---|---|")
    
    for row in db.select():
        atoms = row.toatoms()
        try:
            ml_eform = predictor.predict(atoms)
            print(f"| {row.id:2d} | {row.host:5s} | {row.dopant:6s} | {row.formula:10s} | {ml_eform:14.4f} |")
        except Exception as e:
            print(f"| {row.id:2d} | {row.host:5s} | {row.dopant:6s} | {row.formula:10s} | Error: {e} |")

if __name__ == '__main__':
    main()
