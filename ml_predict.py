import os
import sys
import torch
from ase.io import read

# Setup paths
ROOT_DIR = '/root/autodl-tmp/2d-defect-dast'
sys.path.append(ROOT_DIR)
from predictor import FormationEnergyPredictor

def main():
    model_path = os.path.join(ROOT_DIR, 'models', 'formation_energy_model.pth')
    feature_path = os.path.join(ROOT_DIR, 'atom_features.pth')
    
    print("Loading predictor...")
    predictor = FormationEnergyPredictor(model_path=model_path, feature_path=feature_path, device='cpu')
    
    # Read the summary to get DFT energies
    dft_results = {}
    with open(os.path.join(ROOT_DIR, 'recent_samples_summary.md'), 'r') as f:
        for line in f:
            if line.startswith('|') and 'Completed' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) > 7:
                    idx = int(parts[1])
                    formula = parts[2]
                    dopant = parts[3]
                    eform = float(parts[7])
                    dft_results[idx] = {'formula': formula, 'dopant': dopant, 'eform': eform}
                    
    print(f"\nFound {len(dft_results)} completed samples. Running predictions...")
    
    print("\n| ID | Formula | Dopant | DFT E_form (eV) | ML E_form (eV) | Abs Error (eV) |")
    print("|---|---|---|---|---|---|")
    
    maes = []
    
    # Sort for output display
    for idx in sorted(dft_results.keys()):
        info = dft_results[idx]
        
        # Try to load stage 2, fallback to stage 1
        traj_path = os.path.join(ROOT_DIR, f'sample_{idx}_defect_s2.traj')
        if not os.path.exists(traj_path):
            traj_path = os.path.join(ROOT_DIR, f'sample_{idx}_defect_s1.traj')
            
        if not os.path.exists(traj_path):
            continue
            
        try:
            structure = read(traj_path)
            ml_eform = predictor.predict(structure)
            
            dft_eform = info['eform']
            error = abs(ml_eform - dft_eform)
            maes.append(error)
            
            print(f"| {idx:2d} | {info['formula']:7s} | {info['dopant']:6s} | {dft_eform:15.4f} | {ml_eform:14.4f} | {error:14.4f} |")
        except Exception as e:
            print(f"| {idx:2d} | Error reading or predicting: {e} |")
            
    if maes:
        print(f"\n**Mean Absolute Error (MAE) for these {len(maes)} samples:** {sum(maes)/len(maes):.4f} eV")

if __name__ == '__main__':
    main()
