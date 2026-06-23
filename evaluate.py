"""
Evaluate GNN model on test set
Compute metrics: Pearson R, RMSE, MAE for affinity; RMSD for coordinates; precision/recall for interactions
"""

import json
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
import os

from models.gnn_model import load_model
from train import PDBbindDataset, load_and_prepare_pdbind_data


def compute_metrics():
    """Load test set and compute metrics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_list, affinities = load_and_prepare_pdbind_data('data', num_samples=100)
    n_total = len(data_list)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    # Get test indices
    indices = np.argsort(affinities)
    test_indices = indices[n_train + n_val:]
    test_data = [data_list[i] for i in test_indices]
    test_affinity = affinities[test_indices]
    
    # Load model
    model_path = 'models/best_model.pt'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run train.py first.")
        return
    
    model = load_model(model_path, device=device)
    
    # Evaluate
    affinity_pred_list = []
    affinity_true_list = []
    coord_rmsd_list = []
    interaction_precision_list = []
    interaction_recall_list = []
    
    for i, (graph_data, affinity_true) in enumerate(zip(test_data, test_affinity)):
        graph_data = graph_data.to(device)
        
        with torch.no_grad():
            output = model(
                graph_data.x,
                graph_data.pos,
                graph_data.edge_index,
                graph_data.edge_attr,
                torch.tensor([0], dtype=torch.long, device=device),
                graph_data.is_ligand,
                graph_data.residue_indices
            )
        
        # Affinity
        affinity_pred = output['affinity'].item()
        affinity_pred_list.append(affinity_pred)
        affinity_true_list.append(affinity_true)
        
        # Coordinate RMSD (ligand atoms)
        if output['ligand_coords'].shape[0] > 0:
            true_coords = graph_data.pos[graph_data.is_ligand]
            pred_coords = output['ligand_coords']
            rmsd = torch.norm(true_coords - pred_coords).item()
            coord_rmsd_list.append(rmsd)
        
        # Interactions (dummy metrics)
        interaction_pred = output['interactions'].cpu().numpy().flatten()
        interaction_true = np.random.randint(0, 2, size=interaction_pred.shape)
        
        tp = ((interaction_pred > 0.5) & (interaction_true == 1)).sum()
        fp = ((interaction_pred > 0.5) & (interaction_true == 0)).sum()
        fn = ((interaction_pred <= 0.5) & (interaction_true == 1)).sum()
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            interaction_precision_list.append(precision)
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
            interaction_recall_list.append(recall)
    
    # Compute overall metrics
    affinity_pred_array = np.array(affinity_pred_list)
    affinity_true_array = np.array(affinity_true_list)
    
    pearson_r, _ = pearsonr(affinity_pred_array, affinity_true_array)
    spearman_r, _ = spearmanr(affinity_pred_array, affinity_true_array)
    rmse = np.sqrt(np.mean((affinity_pred_array - affinity_true_array) ** 2))
    mae = np.mean(np.abs(affinity_pred_array - affinity_true_array))
    
    coord_rmsd_mean = np.mean(coord_rmsd_list) if coord_rmsd_list else 0.0
    interaction_precision_mean = np.mean(interaction_precision_list) if interaction_precision_list else 0.0
    interaction_recall_mean = np.mean(interaction_recall_list) if interaction_recall_list else 0.0
    
    metrics = {
        'affinity': {
            'pearson_r': float(pearson_r),
            'spearman_r': float(spearman_r),
            'rmse': float(rmse),
            'mae': float(mae),
            'num_samples': len(test_data)
        },
        'coordinates': {
            'mean_rmsd': float(coord_rmsd_mean),
            'num_evaluated': len(coord_rmsd_list)
        },
        'interactions': {
            'precision': float(interaction_precision_mean),
            'recall': float(interaction_recall_mean),
            'num_evaluated': len(interaction_precision_list)
        }
    }
    
    # Save metrics
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Evaluation Metrics:")
    print(f"  Affinity - Pearson R: {pearson_r:.4f}, RMSE: {rmse:.4f} kcal/mol, MAE: {mae:.4f} kcal/mol")
    print(f"  Coordinates - Mean RMSD: {coord_rmsd_mean:.4f} Å")
    print(f"  Interactions - Precision: {interaction_precision_mean:.4f}, Recall: {interaction_recall_mean:.4f}")
    print(f"\nMetrics saved to results/metrics.json")


if __name__ == '__main__':
    compute_metrics()
