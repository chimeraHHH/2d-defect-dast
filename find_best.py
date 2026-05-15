import os
import json
import torch

best_mae = float('inf')
best_model_path = ""
best_config = ""

results_dir = 'results'
for root, dirs, files in os.walk(results_dir):
    for f in files:
        if f == 'metrics.json':
            path = os.path.join(root, f)
            try:
                with open(path, 'r') as jf:
                    metrics = json.load(jf)
                    if 'best_val_mae' in metrics and metrics['best_val_mae'] < best_mae:
                        best_mae = metrics['best_val_mae']
                        model_file = os.path.join(root, 'best_model.pth')
                        if os.path.exists(model_file):
                            best_model_path = model_file
                            best_config = path
            except Exception:
                pass

print(f"Best model path: {best_model_path}")
print(f"Best MAE: {best_mae}")
print(f"Config path: {best_config}")
