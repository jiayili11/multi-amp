"""
MultiAMP Prediction Script
Based on amppre/predict.py, for model evaluation
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
import pandas as pd
import numpy as np

from model import PeptideTriStreamModel
from dataset import PeptideDataset, custom_collate_fn
from config import MultiAMPConfig


def predict(model, data_loader, device):
    """Generate predictions"""
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            # Move to device
            sequences = batch['sequence_str']
            labels = batch['label'].cpu().numpy()
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            contact_maps = batch.get('contact_maps')
            if contact_maps is not None:
                contact_maps = contact_maps.to(device, non_blocking=True)
                
            node_geometric_feat = batch.get('node_geometric_feat')
            if node_geometric_feat is not None:
                node_geometric_feat = node_geometric_feat.to(device, non_blocking=True)
                
            edge_index = batch.get('edge_index')
            if edge_index is not None:
                edge_index = edge_index.to(device, non_blocking=True)
                
            edge_attr = batch.get('edge_attr')
            if edge_attr is not None:
                edge_attr = edge_attr.to(device, non_blocking=True)
                
            node_coords = batch.get('node_coords')
            if node_coords is not None:
                node_coords = node_coords.to(device, non_blocking=True)
            
            # Forward
            outputs = model(
                sequences,
                attention_mask,
                contact_maps,
                node_geometric_feat,
                edge_index,
                edge_attr,
                node_coords
            )
            
            # Get predictions
            class_logits = outputs['class_logits']
            probs = torch.sigmoid(class_logits).cpu().numpy()
            
            # Store results
            for i in range(len(sequences)):
                results.append({
                    'sequence': sequences[i],
                    'label': int(labels[i]),
                    'probability': float(probs[i]),
                    'prediction': int(probs[i] > 0.5)
                })
    
    return results


def evaluate_predictions(results):
    """Calculate metrics from predictions"""
    labels = [r['label'] for r in results]
    probs = [r['probability'] for r in results]
    preds = [r['prediction'] for r in results]
    
    metrics = {
        'acc': accuracy_score(labels, preds),
        'auc': roc_auc_score(labels, probs),
        'f1': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'mcc': matthews_corrcoef(labels, preds)
    }
    
    return metrics


def main():
    config = MultiAMPConfig()
    device = torch.device(config.DEVICE)
    
    print(f"Using device: {device}")
    
    # Load validation dataset
    print("\n=== Loading Validation Dataset ===")
    pdb_dirs_map = {
        1: config.AMP_VALID_PDB_DIR,
        0: config.NONAMP_VALID_PDB_DIR
    }
    
    valid_dataset = PeptideDataset(
        data_path=config.VALID_DATA_PATH,
        pdb_dirs=pdb_dirs_map,
        max_len=config.MAX_SEQ_LEN,
        is_training=False,
        config=config
    )
    
    print(f"Validation samples: {len(valid_dataset)}")
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    print("\n=== Loading Model ===")
    model = PeptideTriStreamModel(config).to(device)
    
    model_path = f"{config.SAVE_DIR}/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Model loaded from {model_path}")
    
    # Generate predictions
    print("\n=== Generating Predictions ===")
    results = predict(model, valid_loader, device)
    
    # Evaluate
    metrics = evaluate_predictions(results)
    
    print("\n=== Results ===")
    print(f"Accuracy:  {metrics['acc']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"MCC:       {metrics['mcc']:.4f}")
    
    # Save predictions
    output_path = f"{config.SAVE_DIR}/predictions.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Predictions saved to {output_path}")


if __name__ == '__main__':
    main()
