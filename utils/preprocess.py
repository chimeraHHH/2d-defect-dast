import torch
import numpy as np
import os
from ase import Atoms
from ase.neighborlist import neighbor_list

class CrystalPreprocessor:
    def __init__(self, feature_path='/Users/wuleyan/Desktop/2D_Platform/atom_features.pth', cutoff_radius=5.0, device='cpu'):
        """
        Initialize the preprocessor.
        
        Args:
            feature_path (str): Path to the atom_features.pth file.
            cutoff_radius (float): Cutoff radius for graph edges.
            device (str): Device to run computation on ('cpu' or 'cuda').
        """
        self.device = device
        self.cutoff_radius = cutoff_radius
        
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Atom features not found at {feature_path}")
            
        self.atom_features = torch.load(feature_path, map_location='cpu').to(self.device)
        
    def compute_pbc_distance_matrix(self, positions, cell):
        """
        Compute N*N distance matrix with PBC using Minimum Image Convention (MIC).
        """
        # 1. Compute pairwise difference vectors (Cartesian)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # 2. Convert to fractional coordinates
        try:
            cell_inv = torch.linalg.inv(cell)
        except RuntimeError:
            return torch.norm(diff, dim=-1)
            
        diff_frac = diff @ cell_inv
        
        # 3. Apply MIC: shift fractional coordinates to [-0.5, 0.5]
        diff_frac_mic = diff_frac - torch.round(diff_frac)
        
        # 4. Convert back to Cartesian
        diff_cart_mic = diff_frac_mic @ cell
        
        # 5. Compute Euclidean norm
        dist_matrix = torch.norm(diff_cart_mic, dim=-1)
        
        return dist_matrix

    def process(self, structure):
        """
        Convert a user-input crystal structure into a batch_dict format for the model.
        
        Args:
            structure: Can be an ase.Atoms object or a pymatgen Structure object.
            
        Returns:
            batch_dict: A dictionary ready to be fed into the model.
        """
        # Convert pymatgen Structure to ase.Atoms if needed
        if hasattr(structure, "lattice"):
            try:
                from pymatgen.io.ase import AseAtomsAdaptor
                atoms = AseAtomsAdaptor().get_atoms(structure)
            except ImportError:
                raise ImportError("pymatgen is not installed but a pymatgen-like structure was provided.")
        elif isinstance(structure, Atoms):
            atoms = structure
        else:
            raise TypeError("Unsupported structure type. Please provide an ase.Atoms or pymatgen Structure object.")
            
        # 1. Extract basic info
        numbers = atoms.get_atomic_numbers()
        positions_np = atoms.get_positions()
        cell_np = atoms.get_cell()[:]
        n_atoms = len(atoms)
        
        # 2. Build Neighbor List (Edges)
        idx_i, idx_j, d_ij, D_ij = neighbor_list('ijdD', atoms, cutoff=self.cutoff_radius)
        
        if len(idx_i) > 0:
            edge_index_np = np.vstack([idx_i, idx_j])
            edge_dist_np = d_ij
        else:
            edge_index_np = np.empty((2, 0), dtype=np.int64)
            edge_dist_np = np.empty((0,), dtype=np.float32)
        
        # 3. Build Triplets (Angles)
        triplets = []
        angles = []
        
        if len(idx_i) > 0:
            sorted_indices = np.argsort(idx_i)
            idx_i_sorted = idx_i[sorted_indices]
            idx_j_sorted = idx_j[sorted_indices]
            D_ij_sorted = D_ij[sorted_indices]
            
            unique_center_atoms, split_indices = np.unique(idx_i_sorted, return_index=True)
            split_indices = np.append(split_indices, len(idx_i_sorted))
            
            for k, center_atom in enumerate(unique_center_atoms):
                start = split_indices[k]
                end = split_indices[k+1]
                
                neighbors_j = idx_j_sorted[start:end]
                neighbors_D = D_ij_sorted[start:end]
                
                num_neighbors = len(neighbors_j)
                if num_neighbors < 2:
                    continue
                    
                for m in range(num_neighbors):
                    for n in range(num_neighbors):
                        if m == n:
                            continue
                        
                        vec1 = neighbors_D[m]
                        vec2 = neighbors_D[n]
                        
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)
                        
                        if norm1 < 1e-6 or norm2 < 1e-6:
                            continue
                            
                        dot_prod = np.dot(vec1, vec2)
                        cos_theta = dot_prod / (norm1 * norm2)
                        
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        theta = np.arccos(cos_theta)
                        
                        triplets.append([neighbors_j[m], center_atom, neighbors_j[n]])
                        angles.append(theta)
                        
        if len(triplets) > 0:
            triplet_index_np = np.array(triplets)
            angles_np = np.array(angles)
        else:
            triplet_index_np = np.empty((0, 3), dtype=np.int64)
            angles_np = np.empty((0,), dtype=np.float32)
            
        # 4. Convert to PyTorch Tensors
        numbers_t = torch.tensor(numbers, dtype=torch.long, device=self.device)
        x = self.atom_features[numbers_t] # (N, 9)
        
        edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=self.device)
        edge_dist = torch.tensor(edge_dist_np, dtype=torch.float32, device=self.device)
        triplet_index = torch.tensor(triplet_index_np, dtype=torch.long, device=self.device)
        angles_t = torch.tensor(angles_np, dtype=torch.float32, device=self.device)
        
        positions = torch.tensor(positions_np, dtype=torch.float32, device=self.device)
        cell = torch.tensor(cell_np, dtype=torch.float32, device=self.device)
        
        dist_matrix = self.compute_pbc_distance_matrix(positions, cell)
        
        # 5. Format as batch_dict (batch_size = 1)
        batch_dict = {
            'x': x.unsqueeze(0), # (1, N, 9)
            'dist_matrix': dist_matrix.unsqueeze(0), # (1, N, N)
            'atom_mask': torch.ones((1, n_atoms), dtype=torch.bool, device=self.device), # (1, N)
            'num_atoms': torch.tensor([n_atoms], device=self.device),
            'edge_index_list': [edge_index],
            'edge_dist_list': [edge_dist],
            'triplet_index_list': [triplet_index],
            'angles_list': [angles_t]
        }
        
        return batch_dict
