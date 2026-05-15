import torch
try:
    ckpt = torch.load('models/formation_energy_model.pth', map_location='cpu')
    print("Config:", ckpt.get('config', {}))
    print("Metrics:", ckpt.get('metrics', {}))
    print("Best validation MAE might be:", ckpt.get('best_val_mae', 'Not Found'))
except Exception as e:
    print(e)
