"""
E(n)-Equivariant Graph Neural Network for Protein-Ligand Binding Affinity Prediction
Supports multi-task learning: affinity, ligand coordinates, protein-ligand interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter
import numpy as np


class EGNNLayer(MessagePassing):
    """E(n)-Equivariant Graph Neural Network Layer"""
    def __init__(self, in_channels, out_channels, edge_attr_dim=2):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_attr_dim = edge_attr_dim
        
        # Node update MLP
        self.phi_e = nn.Sequential(
            nn.Linear(2 * in_channels + edge_attr_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, out_channels)
        )
        
        # Coordinate update MLP
        self.phi_x = nn.Sequential(
            nn.Linear(out_channels + edge_attr_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
        # Node aggregation MLP
        self.phi_h = nn.Sequential(
            nn.Linear(in_channels + out_channels, 64),
            nn.SiLU(),
            nn.Linear(64, out_channels)
        )
    
    def forward(self, x, pos, edge_index, edge_attr):
        """
        Args:
            x: Node features [N, in_channels]
            pos: Node positions [N, 3]
            edge_index: Edge indices [2, E]
            edge_attr: Edge attributes [E, edge_attr_dim]
        Returns:
            x_new: Updated node features [N, out_channels]
            pos_new: Updated positions [N, 3]
        """
        # Message passing for edge updates
        x_new = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)
        
        # Coordinate updates (equivariant to rotations/translations)
        row, col = edge_index
        edge_vectors = pos[col] - pos[row]  # [E, 3]
        edge_dist = torch.norm(edge_vectors, dim=1, keepdim=True) + 1e-8
        edge_dir = edge_vectors / edge_dist  # Unit direction [E, 3]
        
        # Edge-based coordinate contribution
        edge_contrib = self.phi_x(torch.cat([x_new[row], edge_attr], dim=1))  # [E, 1]
        
        # Aggregate coordinate updates
        coord_updates = scatter(
            edge_contrib * edge_dir,  # [E, 3]
            row, dim=0, dim_size=x_new.shape[0], reduce='mean'
        )
        pos_new = pos + 0.1 * coord_updates
        
        return x_new, pos_new
    
    def message(self, x_i, x_j, edge_attr):
        """Compute edge messages"""
        edge_feat = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.phi_e(edge_feat)
    
    def aggregate(self, aggr_out, index, ptr, dim_size, x_j, x_i, edge_attr):
        """Aggregate messages and update node features"""
        aggregated = scatter(aggr_out, index, dim=0, dim_size=dim_size, reduce='mean')
        x_new = self.phi_h(torch.cat([x_i, aggregated], dim=1))
        return x_new


class ProteinLigandGNN(nn.Module):
    """Multi-task GNN for protein-ligand complex analysis"""
    def __init__(self, num_node_features=18, num_edge_features=2, hidden_dim=128, num_layers=4, num_residues=400):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_residues = num_residues
        
        # Initial node embedding
        self.node_embed = nn.Linear(num_node_features, hidden_dim)
        self.edge_embed = nn.Linear(num_edge_features, hidden_dim // 2)
        
        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, hidden_dim // 2)
            for _ in range(num_layers)
        ])
        
        # HEAD 1: Affinity prediction
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # HEAD 2: Ligand coordinate prediction (per atom)
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # HEAD 3: Protein residue interaction prediction
        self.interaction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, pos, edge_index, edge_attr, batch, is_ligand_mask, residue_indices):
        """
        Args:
            x: Node features [N, num_node_features]
            pos: Node coordinates [N, 3]
            edge_index: Edge indices [2, E]
            edge_attr: Edge attributes [E, num_edge_features]
            batch: Batch assignment [N]
            is_ligand_mask: Binary mask for ligand atoms [N]
            residue_indices: Residue index per atom [N]
        Returns:
            dict with affinity, ligand_coords, interaction_logits
        """
        # Embed input features
        x = self.node_embed(x)
        edge_attr_emb = self.edge_embed(edge_attr)
        
        # Pass through EGNN layers
        for i, egnn_layer in enumerate(self.egnn_layers):
            x, pos = egnn_layer(x, pos, edge_index, edge_attr_emb)
            x = F.relu(x)
        
        # HEAD 1: Global affinity score
        graph_pool = global_mean_pool(x, batch)  # [B, hidden_dim]
        affinity_logits = self.affinity_head(graph_pool)  # [B, 1]
        
        # HEAD 2: Per-atom ligand coordinates (only for ligand atoms)
        coord_logits = self.coord_head(x)  # [N, 3]
        ligand_coords = coord_logits[is_ligand_mask]  # [n_ligand_atoms, 3]
        
        # HEAD 3: Per-residue interaction probabilities
        # Average features per residue
        residue_features = []
        num_residues_actual = residue_indices.max().item() + 1
        for res_idx in range(min(num_residues_actual, self.num_residues)):
            res_mask = residue_indices == res_idx
            if res_mask.sum() > 0:
                res_feat = x[res_mask].mean(dim=0, keepdim=True)
                interaction_prob = self.interaction_head(res_feat)
                residue_features.append(interaction_prob)
        
        if residue_features:
            interaction_logits = torch.cat(residue_features, dim=0)  # [num_residues, 1]
        else:
            interaction_logits = torch.zeros(1, 1, device=x.device)
        
        return {
            'affinity': affinity_logits,
            'ligand_coords': ligand_coords,
            'interactions': interaction_logits
        }


def load_model(model_path, device='cpu'):
    """Load pre-trained GNN model"""
    model = ProteinLigandGNN(
        num_node_features=18,
        num_edge_features=2,
        hidden_dim=128,
        num_layers=4,
        num_residues=400
    )
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    # Quick test
    model = ProteinLigandGNN()
    x = torch.randn(100, 18)
    pos = torch.randn(100, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.randn(3, 2)
    batch = torch.zeros(100, dtype=torch.long)
    is_ligand = torch.zeros(100, dtype=torch.bool)
    is_ligand[:20] = True
    residues = torch.randint(0, 50, (100,))
    
    output = model(x, pos, edge_index, edge_attr, batch, is_ligand, residues)
    print(f"Affinity shape: {output['affinity'].shape}")
    print(f"Ligand coords shape: {output['ligand_coords'].shape}")
    print(f"Interactions shape: {output['interactions'].shape}")
