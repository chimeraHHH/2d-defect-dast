import os
import glob

def get_final_energy(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'Free energy:' in line or 'Energy:' in line or 'total energy' in line:
                pass # Just a quick check, but let's parse more robustly
    
    # Better to use ASE to read it
    try:
        from ase.io import read
        atoms = read(filename)
        return atoms.get_potential_energy()
    except Exception as e:
        return None

results = {}

# We are interested in samples 40-59
for i in range(40, 60):
    en2_s2 = f"gpaw_sample_{i}_defect_stage2.txt"
    en2_s1 = f"gpaw_sample_{i}_defect_stage1.txt"
    
    host_s2 = f"gpaw_sample_{i}_host_stage2.txt"
    host_s1 = f"gpaw_sample_{i}_host_stage1.txt"
    
    en2 = get_final_energy(en2_s2)
    if en2 is None:
        en2 = get_final_energy(en2_s1)
        
    host = get_final_energy(host_s2)
    if host is None:
        host = get_final_energy(host_s1)
        
    results[i] = {'en2': en2, 'host': host}

for i in range(40, 60):
    print(f"Sample {i}: en2={results[i]['en2']}, host={results[i]['host']}")
