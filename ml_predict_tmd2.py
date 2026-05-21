import os
import sys
import torch
from ase.io import read

ROOT_DIR = '/root/autodl-tmp/2d-defect-dast'
sys.path.append(ROOT_DIR)
from predictor import FormationEnergyPredictor

def main():
    model_path = os.path.join(ROOT_DIR, 'models', 'formation_energy_model.pth')
    feature_path = os.path.join(ROOT_DIR, 'atom_features.pth')
    
    predictor = FormationEnergyPredictor(model_path=model_path, feature_path=feature_path, device='cpu')
    
    dft_results = {}
    with open(os.path.join(ROOT_DIR, 'tmd_samples_summary.md'), 'r') as f:
        for line in f:
            if line.startswith('|') and 'Completed' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) > 8:
                    idx = int(parts[1])
                    formula = parts[2]
                    host = parts[3]
                    dopant = parts[4]
                    eform = float(parts[8])
                    dft_results[idx] = {'formula': formula, 'host': host, 'dopant': dopant, 'eform': eform}
                    
    print("\n| ID | Host | Dopant | Formula | DFT E_form (eV) | ML Predicted E_form (eV) | Abs Error (eV) |")
    print("|---|---|---|---|---|---|---|")
    
    maes = []
    
    for idx in sorted(dft_results.keys()):
        info = dft_results[idx]
        
        traj_path = os.path.join(ROOT_DIR, f'gpaw_sample_{idx}_defect_stage2.txt')
        if not os.path.exists(traj_path):
            traj_path = os.path.join(ROOT_DIR, f'gpaw_sample_{idx}_defect_stage1.txt')
            
        if not os.path.exists(traj_path):
            continue
            
        try:
            structure = read(traj_path)
            ml_eform = predictor.predict(structure)
            
            dft_eform = info['eform']
            error = abs(ml_eform - dft_eform)
            maes.append(error)
            
            print(f"| {idx:2d} | {info['host']:4s} | {info['dopant']:6s} | {info['formula']:10s} | {dft_eform:15.4f} | {ml_eform:24.4f} | {error:14.4f} |")
        except Exception as e:
            print(f"| {idx:2d} | Error reading or predicting: {e} |")
            
    if maes:
        print(f"\n**Mean Absolute Error (MAE) for these {len(maes)} samples:** {sum(maes)/len(maes):.4f} eV")

if __name__ == '__main__':
    main()
