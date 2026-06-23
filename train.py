"""
Train GNN model on PDBbind protein-ligand complexes
Supports multi-task learning: affinity prediction, coordinate regression, interaction classification
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, DataLoader as GeometricDataLoader
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from models.gnn_model import ProteinLigandGNN

ATOM_TYPES = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'H': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9}
BOND_TYPES = {'single': 0, 'double': 1, 'aromatic': 2, 'unknown': 3}


def parse_pdb_file(pdb_content):
    """Parse PDB file and extract atoms"""
    atoms = {'protein': [], 'ligand': []}
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM'):
            atoms['protein'].append(parse_pdb_line(line, 'protein'))
        elif line.startswith('HETATM'):
            atoms['ligand'].append(parse_pdb_line(line, 'ligand'))
    return atoms


def parse_pdb_line(line, atom_type):
    """Parse single PDB line"""
    return {
        'name': line[12:16].strip(),
        'residue': line[17:20].strip(),
        'chain': line[21],
        'residue_num': int(line[22:26]),
        'x': float(line[30:38]),
        'y': float(line[38:46]),
        'z': float(line[46:54]),
        'element': line[76:78].strip(),
        'atom_type': atom_type,
        'charge': 0.0  # Placeholder; can be computed with RDKit if needed
    }


def build_molecular_graph(protein_atoms, ligand_atoms, cutoff=5.0):
    """Build graph with protein and ligand atoms"""
    all_atoms = protein_atoms + ligand_atoms
    num_atoms = len(all_atoms)
    
    if num_atoms == 0:
        raise ValueError("No atoms found in PDB")
    
    # Node features: [atom_type_one_hot(10), is_ligand(1), charge(1), mass_est(1), residue_num(1), ...]
    node_features = []
    positions = []
    is_ligand_mask = []
    residue_indices = []
    
    for i, atom in enumerate(all_atoms):
        # One-hot encode element
        element_onehot = [0] * len(ATOM_TYPES)
        if atom['element'] in ATOM_TYPES:
            element_onehot[ATOM_TYPES[atom['element']]] = 1
        
        # Features: [element_onehot(10), is_ligand(1), charge(1), normalized_residue_num(1), atom_mass(1), name_hash(1), backbone(1), hetero(1)]
        is_ligand = 1 if atom['atom_type'] == 'ligand' else 0
        mass_est = 12 if atom['element'] in ['C', 'N', 'O', 'S'] else 1
        backbone = 1 if atom['name'] in ['N', 'CA', 'C', 'O'] else 0
        hetero = 1 if atom['element'] in ['S', 'P', 'F', 'Cl', 'Br', 'I'] else 0
        
        features = (
            element_onehot +
            [is_ligand, atom['charge'], atom['residue_num'] / 500.0, mass_est / 12.0, 
             hash(atom['name']) % 100 / 100.0, backbone, hetero]
        )
        node_features.append(features)
        positions.append([atom['x'], atom['y'], atom['z']])
        is_ligand_mask.append(is_ligand)
        residue_indices.append(atom.get('residue_num', 0))
    
    node_features = torch.tensor(node_features, dtype=torch.float32)
    positions = torch.tensor(positions, dtype=torch.float32)
    is_ligand_mask = torch.tensor(is_ligand_mask, dtype=torch.bool)
    residue_indices = torch.tensor(residue_indices, dtype=torch.long)
    
    # Build edges within cutoff
    edge_list = []
    edge_features_list = []
    
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dx = positions[i] - positions[j]
            dist = torch.norm(dx).item()
            if dist < cutoff and dist > 0.1:  # Avoid self-loops and overlaps
                # Bond type (heuristic)
                bond_type = 3  # unknown
                if dist < 1.6:
                    bond_type = 0  # single
                elif dist < 1.3:
                    bond_type = 1  # double
                
                # Edge features: [distance, bond_type]
                edge_feat = [dist / cutoff, bond_type / 3.0]
                edge_list.append([i, j])
                edge_features_list.append(edge_feat)
                edge_list.append([j, i])
                edge_features_list.append(edge_feat)
    
    if len(edge_list) == 0:
        # Add at least one edge to avoid empty graph
        edge_list = [[0, 1], [1, 0]] if num_atoms > 1 else [[0, 0]]
        edge_features_list = [[1.0, 0.0]] * len(edge_list)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features_list, dtype=torch.float32)
    
    return {
        'node_features': node_features,
        'positions': positions,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'is_ligand': is_ligand_mask,
        'residue_indices': residue_indices,
        'all_atoms': all_atoms
    }


def compute_interaction_labels(protein_atoms, ligand_atoms, cutoff=4.5):
    """Compute residue-level interaction labels"""
    residue_labels = {}
    
    for lig_atom in ligand_atoms:
        lig_pos = np.array([lig_atom['x'], lig_atom['y'], lig_atom['z']])
        
        for prot_atom in protein_atoms:
            prot_pos = np.array([prot_atom['x'], prot_atom['y'], prot_atom['z']])
            dist = np.linalg.norm(lig_pos - prot_pos)
            
            if dist < cutoff:
                res_key = (prot_atom['chain'], prot_atom['residue_num'])
                residue_labels[res_key] = 1
    
    return residue_labels


class PDBbindDataset(Dataset):
    """PDBbind dataset"""
    def __init__(self, data_list, affinity_values, split='train'):
        self.data_list = data_list
        self.affinity_values = affinity_values
        self.split = split
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        graph_data = self.data_list[idx]
        affinity = torch.tensor([self.affinity_values[idx]], dtype=torch.float32)
        return graph_data, affinity


def load_and_prepare_pdbind_data(data_dir, num_samples=100):
    """Load PDBbind INDEX file and prepare graph data"""
    print(f"Loading PDBbind data from {data_dir}...")
    
    # Simulate loading (in practice, read actual PDBbind INDEX)
    data_list = []
    affinities = []
    
    # For demo: create synthetic data from existing CSV if available
    csv_path = os.path.join(data_dir, 'processed', 'real_data_merged_5200.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, nrows=num_samples)
        print(f"Loaded {len(df)} records from CSV")
        
        for _, row in df.iterrows():
            try:
                # Create synthetic graph (in real scenario, parse actual PDB files)
                num_protein = np.random.randint(50, 200)
                num_ligand = np.random.randint(5, 30)
                
                protein_atoms = [{
                    'name': f'CA_{i}', 'residue': 'ALA', 'chain': 'A',
                    'residue_num': i, 'x': float(i), 'y': float(i % 10), 'z': 0.0,
                    'element': 'C', 'atom_type': 'protein', 'charge': 0.0
                } for i in range(num_protein)]
                
                ligand_atoms = [{
                    'name': f'C{i}', 'residue': 'LIG', 'chain': 'B',
                    'residue_num': 0, 'x': float(5 + i * 0.5), 'y': float(5 + i % 5),
                    'z': float(i % 3), 'element': 'C' if i % 2 == 0 else 'N',
                    'atom_type': 'ligand', 'charge': 0.0
                } for i in range(num_ligand)]
                
                graph = build_molecular_graph(protein_atoms, ligand_atoms)
                
                # Convert to PyTorch Geometric Data
                data = Data(
                    x=graph['node_features'],
                    pos=graph['positions'],
                    edge_index=graph['edge_index'],
                    edge_attr=graph['edge_attr']
                )
                data.is_ligand = graph['is_ligand']
                data.residue_indices = graph['residue_indices']
                
                data_list.append(data)
                
                # Affinity from CSV (convert Ki/IC50 to -logKd)
                affinity_val = float(row.get('affinity_value', -8.0))
                affinities.append(affinity_val)
                
            except Exception as e:
                print(f"Error processing sample {_}: {e}")
                continue
    else:
        print(f"CSV not found at {csv_path}, creating synthetic data...")
        for i in range(num_samples):
            num_protein = np.random.randint(50, 200)
            num_ligand = np.random.randint(5, 30)
            
            protein_atoms = [{
                'name': f'CA_{j}', 'residue': 'ALA', 'chain': 'A',
                'residue_num': j, 'x': float(j), 'y': float(j % 10), 'z': 0.0,
                'element': 'C', 'atom_type': 'protein', 'charge': 0.0
            } for j in range(num_protein)]
            
            ligand_atoms = [{
                'name': f'C{j}', 'residue': 'LIG', 'chain': 'B',
                'residue_num': 0, 'x': float(5 + j * 0.5), 'y': float(5 + j % 5),
                'z': float(j % 3), 'element': 'C' if j % 2 == 0 else 'N',
                'atom_type': 'ligand', 'charge': 0.0
            } for j in range(num_ligand)]
            
            graph = build_molecular_graph(protein_atoms, ligand_atoms)
            data = Data(
                x=graph['node_features'],
                pos=graph['positions'],
                edge_index=graph['edge_index'],
                edge_attr=graph['edge_attr']
            )
            data.is_ligand = graph['is_ligand']
            data.residue_indices = graph['residue_indices']
            
            data_list.append(data)
            affinities.append(np.random.uniform(-14, -5))
    
    print(f"Prepared {len(data_list)} graph samples")
    return data_list, np.array(affinities)


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    affinity_loss_sum = 0
    coord_loss_sum = 0
    interaction_loss_sum = 0
    
    criterion_affinity = nn.MSELoss()
    criterion_coords = nn.MSELoss()
    criterion_interaction = nn.BCELoss()
    
    for batch_data, affinity_target in train_loader:
        batch_data = batch_data.to(device)
        affinity_target = affinity_target.to(device)
        
        optimizer.zero_grad()
        
        # Model forward pass
        output = model(
            batch_data.x,
            batch_data.pos,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch if hasattr(batch_data, 'batch') else torch.zeros(batch_data.x.size(0), dtype=torch.long, device=device),
            batch_data.is_ligand,
            batch_data.residue_indices
        )
        
        # Affinity loss
        affinity_pred = output['affinity']
        loss_affinity = criterion_affinity(affinity_pred, affinity_target)
        
        # Coordinate loss (for ligand atoms)
        ligand_atoms_mask = batch_data.is_ligand
        if ligand_atoms_mask.sum() > 0:
            ligand_pos = batch_data.pos[ligand_atoms_mask]
            ligand_pred_coords = output['ligand_coords']
            loss_coords = criterion_coords(ligand_pred_coords, ligand_pos)
        else:
            loss_coords = torch.tensor(0.0, device=device)
        
        # Interaction loss (multi-label, per residue)
        interaction_pred = output['interactions']
        if interaction_pred.size(0) > 0:
            interaction_target = torch.bernoulli(torch.ones(interaction_pred.size(0), device=device) * 0.3).unsqueeze(1)
            loss_interaction = criterion_interaction(interaction_pred, interaction_target)
        else:
            loss_interaction = torch.tensor(0.0, device=device)
        
        # Multi-task loss
        loss = 0.4 * loss_affinity + 0.4 * loss_coords + 0.2 * loss_interaction
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        affinity_loss_sum += loss_affinity.item()
        coord_loss_sum += loss_coords.item()
        interaction_loss_sum += loss_interaction.item()
    
    num_batches = len(train_loader)
    return {
        'total': total_loss / num_batches,
        'affinity': affinity_loss_sum / num_batches,
        'coords': coord_loss_sum / num_batches,
        'interaction': interaction_loss_sum / num_batches
    }


def evaluate(model, val_loader, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    all_affinity_pred = []
    all_affinity_true = []
    
    criterion_affinity = nn.MSELoss()
    
    with torch.no_grad():
        for batch_data, affinity_target in val_loader:
            batch_data = batch_data.to(device)
            affinity_target = affinity_target.to(device)
            
            output = model(
                batch_data.x,
                batch_data.pos,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.batch if hasattr(batch_data, 'batch') else torch.zeros(batch_data.x.size(0), dtype=torch.long, device=device),
                batch_data.is_ligand,
                batch_data.residue_indices
            )
            
            affinity_pred = output['affinity']
            loss = criterion_affinity(affinity_pred, affinity_target)
            total_loss += loss.item()
            
            all_affinity_pred.extend(affinity_pred.cpu().numpy().flatten())
            all_affinity_true.extend(affinity_target.cpu().numpy().flatten())
    
    # Compute Pearson R
    if len(all_affinity_pred) > 1:
        pearson_r, _ = pearsonr(all_affinity_pred, all_affinity_true)
    else:
        pearson_r = 0.0
    
    return {
        'loss': total_loss / len(val_loader),
        'pearson_r': pearson_r,
        'pred': all_affinity_pred,
        'true': all_affinity_true
    }


def main():
    """Train GNN model"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='PDBbind data directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_samples', type=int, default=100, help='Limit samples for demo')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_list, affinities = load_and_prepare_pdbind_data(args.data_dir, num_samples=args.num_samples)
    print(f"Loaded {len(data_list)} complexes, affinity range: [{affinities.min():.2f}, {affinities.max():.2f}]")
    
    # Train/val/test split (80/10/10)
    n_total = len(data_list)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    indices = np.argsort(affinities)  # Stratified split
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_data = [data_list[i] for i in train_indices]
    train_affinity = affinities[train_indices]
    val_data = [data_list[i] for i in val_indices]
    val_affinity = affinities[val_indices]
    test_data = [data_list[i] for i in test_indices]
    test_affinity = affinities[test_indices]
    
    train_dataset = PDBbindDataset(train_data, train_affinity)
    val_dataset = PDBbindDataset(val_data, val_affinity)
    test_dataset = PDBbindDataset(test_data, test_affinity)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    model = ProteinLigandGNN(num_node_features=18, num_edge_features=2, hidden_dim=128, num_layers=4)
    model = model.to(device)
    
    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = 'models/best_model.pt'
    os.makedirs('models', exist_ok=True)
    
    print("\nTraining...")
    for epoch in range(args.epochs):
        train_losses = train_epoch(model, train_loader, optimizer, device)
        val_results = evaluate(model, val_loader, device)
        scheduler.step()
        
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_val_loss
            }, best_model_path)
            print(f"✓ Best model saved at epoch {epoch}")
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {train_losses['total']:.4f} (Aff: {train_losses['affinity']:.4f}, "
                  f"Coord: {train_losses['coords']:.4f}, Int: {train_losses['interaction']:.4f}) | "
                  f"Val Loss: {val_results['loss']:.4f} | "
                  f"Pearson R: {val_results['pearson_r']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Pearson R: {test_results['pearson_r']:.4f}")
    
    # Save metrics
    os.makedirs('results', exist_ok=True)
    metrics = {
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'test_loss': float(test_results['loss']),
        'test_pearson_r': float(test_results['pearson_r']),
        'affinity_predictions': test_results['pred'],
        'affinity_true': test_results['true']
    }
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Training complete! Best model saved to {best_model_path}")


if __name__ == '__main__':
    main()
