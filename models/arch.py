import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RBFExpansion(nn.Module):
    """
    Gaussian Radial Basis Function expansion for continuous distances.
    Maps a scalar distance d to a vector of size n_rbf.
    """
    def __init__(self, dmin=0.0, dmax=8.0, n_rbf=32):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(dmin, dmax, n_rbf), requires_grad=False)
        self.sigma = (dmax - dmin) / n_rbf
        
    def forward(self, d):
        # d: (..., ) -> (..., n_rbf)
        return torch.exp(-((d.unsqueeze(-1) - self.centers)**2) / self.sigma**2)

class LocalInteractionLayer(nn.Module):
    """
    GNN layer incorporating 2-body (bond) and 3-body (angle) interactions.
    Inspired by CGCNN/SchNet but adding angle information explicitly.
    """
    def __init__(self, hidden_dim, n_rbf_edge=32, n_rbf_angle=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Edge Update (2-body)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + n_rbf_edge, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Angle Update (3-body)
        # We encode angles (0 to pi) using RBFs as well
        self.angle_rbf = RBFExpansion(dmin=0.0, dmax=math.pi, n_rbf=n_rbf_angle)
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_dim + n_rbf_angle, hidden_dim), # Input: edge_feat + angle_rbf
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Node Update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim), # Input: old_node + aggregated_message
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, edge_index, edge_attr_rbf, triplet_index, angles):
        """
        x: (N_total, hidden_dim) - flattened batch features
        edge_index: (2, E_total)
        edge_attr_rbf: (E_total, n_rbf_edge)
        triplet_index: (M_total, 3) - [j, i, k] (neighbors j, k around center i)
        angles: (M_total, ) - angle values in radians
        """
        row, col = edge_index
        
        # --- 1. Compute Edge Features (m_ij) ---
        # m_ij = MLP([x_i, x_j, rbf(d_ij)])
        # x[row] is target node features, x[col] is source node features
        edge_cat = torch.cat([x[row], x[col], edge_attr_rbf], dim=-1)
        edge_messages = self.edge_mlp(edge_cat) # (E, hidden_dim)
        
        # --- 2. Compute Triplet Interactions (Optional but requested) ---
        # This part is tricky because we need to aggregate triplet info back to edges or nodes.
        # Here we model it as modulating the edge messages or creating new messages.
        # Strategy: 
        #   For each triplet (j, i, k), we compute a feature t_jik based on angle.
        #   We then aggregate these t_jik to the center node i.
        
        # Encode angles
        angle_rbf = self.angle_rbf(angles) # (M, n_rbf_angle)
        
        # To make it efficient, we can project the center node i's feature 
        # and combine with angle. Or use the edge messages involved.
        # For simplicity in this demo: t_jik = MLP(x_i || rbf(theta))
        # Note: A more complex version would use the incoming edge features m_ji and m_ki.
        
        center_nodes = triplet_index[:, 1] # i
        triplet_input = torch.cat([x[center_nodes], angle_rbf], dim=-1)
        triplet_messages = self.triplet_mlp(triplet_input) # (M, hidden_dim)
        
        # --- 3. Aggregation ---
        # Aggregate edge messages to target nodes (row)
        # We use scatter_add equivalent (index_add_)
        aggr_messages = torch.zeros_like(x)
        aggr_messages.index_add_(0, row, edge_messages)
        
        # Aggregate triplet messages to center nodes
        aggr_messages.index_add_(0, center_nodes, triplet_messages)
        
        # --- 4. Node Update ---
        new_x = self.node_mlp(torch.cat([x, aggr_messages], dim=-1))
        
        # Residual connection
        return x + new_x

class GeometricTransformerBlock(nn.Module):
    """
    Transformer block with geometric bias based on PBC distances.
    Attention(Q, K, V) = softmax( (QK^T)/sqrt(d) + phi(d_ij) ) * V
    """
    def __init__(self, hidden_dim, num_heads=4, n_rbf_dist=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Geometric Bias MLP: maps distance RBF -> scalar bias (per head)
        self.dist_rbf = RBFExpansion(dmin=0.0, dmax=10.0, n_rbf=n_rbf_dist)
        self.bias_mlp = nn.Sequential(
            nn.Linear(n_rbf_dist, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads) # One bias value per head per pair
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, x, dist_matrix, mask=None):
        """
        x: (Batch, N_max, hidden_dim)
        dist_matrix: (Batch, N_max, N_max)
        mask: (Batch, N_max) - True for valid atoms
        """
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        residual = x
        x = self.norm1(x)
        
        # Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N, D)
        
        # Standard Attention Scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D) # (B, H, N, N)
        
        # Geometric Bias
        # dist_matrix: (B, N, N) -> RBF -> (B, N, N, n_rbf)
        rbf_feat = self.dist_rbf(dist_matrix) 
        # bias: (B, N, N, H)
        geom_bias = self.bias_mlp(rbf_feat) 
        # Permute to (B, H, N, N) to add to scores
        geom_bias = geom_bias.permute(0, 3, 1, 2)
        
        scores = scores + geom_bias
        
        # Apply Mask (Padding)
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N) broadcastable
            mask_expanded = mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, N) target dim
            # We want to mask out attention to padding tokens (keys)
            scores = scores.masked_fill(~mask_expanded, float('-1e9'))
            
        attn = F.softmax(scores, dim=-1)
        
        # Aggregate
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        # Residual 1
        x = residual + out
        
        # FFN + Residual 2
        x = x + self.ffn(self.norm2(x))
        
        return x

class CrystalTransformer(nn.Module):
    def __init__(self, atom_fea_len=9, hidden_dim=128, n_local_layers=3, n_global_layers=2):
        super().__init__()
        
        # 1. Embedding
        self.embedding = nn.Linear(atom_fea_len, hidden_dim)
        
        # 2. Local Path (GNN)
        self.edge_rbf = RBFExpansion(dmin=0.0, dmax=5.0, n_rbf=32)
        self.local_layers = nn.ModuleList([
            LocalInteractionLayer(hidden_dim, n_rbf_edge=32, n_rbf_angle=32)
            for _ in range(n_local_layers)
        ])
        
        # 3. Global Path (Transformer)
        self.global_layers = nn.ModuleList([
            GeometricTransformerBlock(hidden_dim, num_heads=4, n_rbf_dist=32)
            for _ in range(n_global_layers)
        ])
        
        # 4. Readout
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, batch_dict):
        # Unpack batch
        # x: (B, N, 9)
        # mask: (B, N)
        # dist_matrix: (B, N, N)
        # edge_index_list, edge_dist_list, etc. (Lists of length B)
        
        x = batch_dict['x']
        mask = batch_dict['atom_mask']
        dist_matrix = batch_dict['dist_matrix']
        
        B, N, _ = x.shape
        
        # --- 1. Embedding ---
        h = self.embedding(x) # (B, N, H)
        
        # --- 2. Local Layers (Flattened Batch Processing) ---
        # Because GNNs work best on sparse graphs, we flatten the batch into a single big graph
        # This requires re-indexing edge indices
        
        # Prepare flattened features
        # We only take valid atoms based on mask
        flat_h = h[mask] # (N_total_valid, H)
        
        # Prepare flattened graph connectivity
        # We need to shift indices in edge_index_list based on cumulative atom counts
        device = h.device
        num_atoms = batch_dict['num_atoms']
        cum_atoms = torch.cumsum(torch.cat([torch.tensor([0], device=device), num_atoms[:-1]]), dim=0)
        
        all_edge_indices = []
        all_edge_dists = []
        all_triplet_indices = []
        all_angles = []
        
        for i in range(B):
            offset = cum_atoms[i].item()
            
            # Edges
            e_idx = batch_dict['edge_index_list'][i] + offset
            all_edge_indices.append(e_idx)
            all_edge_dists.append(batch_dict['edge_dist_list'][i])
            
            # Triplets
            if len(batch_dict['triplet_index_list'][i]) > 0:
                t_idx = batch_dict['triplet_index_list'][i] + offset
                all_triplet_indices.append(t_idx)
                all_angles.append(batch_dict['angles_list'][i])
        
        flat_edge_index = torch.cat(all_edge_indices, dim=1)
        flat_edge_dist = torch.cat(all_edge_dists, dim=0)
        
        if len(all_triplet_indices) > 0:
            flat_triplet_index = torch.cat(all_triplet_indices, dim=0)
            flat_angles = torch.cat(all_angles, dim=0)
        else:
            # Handle case with no triplets (rare but possible)
            flat_triplet_index = torch.empty((0, 3), dtype=torch.long, device=device)
            flat_angles = torch.empty((0,), dtype=torch.float32, device=device)
            
        # RBF expansion for edge distances
        edge_attr_rbf = self.edge_rbf(flat_edge_dist)
        
        # Run Local GNN Layers
        for layer in self.local_layers:
            flat_h = layer(flat_h, flat_edge_index, edge_attr_rbf, flat_triplet_index, flat_angles)
            
        # Un-flatten back to (B, N, H) for Transformer
        # We scatter flat_h back to a zero-filled tensor
        h_updated = torch.zeros_like(h)
        h_updated[mask] = flat_h
        
        # --- 3. Global Layers (Transformer) ---
        # Now we operate on (B, N, H) with padding masked out
        for layer in self.global_layers:
            h_updated = layer(h_updated, dist_matrix, mask)
            
        # --- 4. Readout ---
        # Global Average Pooling
        # Mask out padding nodes before averaging
        # sum(h * mask) / sum(mask)
        mask_float = mask.float().unsqueeze(-1) # (B, N, 1)
        h_masked = h_updated * mask_float
        
        global_feat = h_masked.sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9) # (B, H)
        
        # Regression
        out = self.readout_mlp(global_feat) # (B, 1)
        
        return out.squeeze(-1)

if __name__ == "__main__":
    # Test the model with dummy data
    import sys
    import os
    # Add parent directory to path to allow importing WLY module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from WLY.data_loader import CrystalGraphDataset, DataLoader, collate_fn
    
    DATA_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/cleaned_dataset.pkl'
    FEATURE_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/atom_features.pth'
    
    dataset = CrystalGraphDataset(DATA_PATH, FEATURE_PATH)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    model = CrystalTransformer(atom_fea_len=9, hidden_dim=64, n_local_layers=2, n_global_layers=2)
    
    for batch in loader:
        print("\n--- Model Forward Test ---")
        output = model(batch)
        print(f"Output shape: {output.shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"Prediction: {output}")
        print(f"Ground Truth: {batch['target']}")
        break
