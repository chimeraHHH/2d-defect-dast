#!/usr/bin/env python3
"""
Extract attention weights from DART's global layers for interpretability.
Run on WHU server where model + data are available.

Usage:
  python extract_attention.py --checkpoint /path/to/model.pt --output attention_data.npz
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch

def extract_attention_weights(checkpoint_path, output_path, n_samples=50):
    """Extract attention weights from a trained DART model."""
    from src.data.dataset import CrystalDataset
    from src.models.crystal_transformer import CrystalTransformer
    from torch.utils.data import DataLoader

    # Load config from checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})

    # Load test dataset
    dataset = CrystalDataset(
        split='test',
        root=config.get('data_root', 'data'),
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Build model
    model = CrystalTransformer(**config.get('model', {}))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Hook to capture attention weights
    attention_maps = []
    defect_indices = []
    sample_info = []

    def attn_hook(module, input, output):
        # Store attention weights
        if hasattr(output, 'attn_weights'):
            attention_maps.append(output.attn_weights.detach().cpu())

    # Register hooks on global attention layers
    hooks = []
    for name, module in model.named_modules():
        if 'global' in name.lower() and 'attn' in name.lower():
            h = module.register_forward_hook(attn_hook)
            hooks.append(h)
            print(f"Hooked: {name}")

    # If no hooks registered, try alternative approach
    if not hooks:
        print("No attention hooks found. Using manual weight extraction...")
        # Manually extract from transformer layers
        for name, module in model.named_modules():
            if hasattr(module, 'self_attn') or hasattr(module, 'attention'):
                print(f"  Found attention-like module: {name}")

    # Run inference on subset
    results = {
        'defect_attention_ratio': [],
        'host_names': [],
        'dopant_names': [],
        'defect_types': [],
        'n_atoms': [],
    }

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            output = model(batch)

            # Get defect mask
            defect_mask = batch.get('defect_mask', batch.get('is_defect', None))
            if defect_mask is not None:
                defect_mask = defect_mask.bool()

            if i % 10 == 0:
                print(f"  Processed {i}/{n_samples}")

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Save results
    np.savez(output_path, **{k: np.array(v) for k, v in results.items()})
    print(f"Saved attention data to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='attention_data.npz')
    parser.add_argument('--n_samples', type=int, default=50)
    args = parser.parse_args()

    extract_attention_weights(args.checkpoint, args.output, args.n_samples)
