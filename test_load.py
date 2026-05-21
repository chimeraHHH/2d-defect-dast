import torch
ckpt = torch.load('/root/autodl-tmp/2d-defect-dast/models/formation_energy_model.pth', map_location='cpu', weights_only=False)
print("Config:", ckpt['config'])
print("Normalizer:", ckpt['normalizer'])
print("Val MAE:", ckpt.get('val_mae'))
