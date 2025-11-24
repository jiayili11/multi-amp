"""
MultiAMP Training Script
Based on amppre/train.py, simplified for training phase only
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
import pandas as pd

from model import PeptideTriStreamModel
from dataset import PeptideDataset, custom_collate_fn
from losses import SupConLoss, StructureAwareLoss, compute_ss_class_weights
from config import MultiAMPConfig


def train_epoch(model, train_loader, optimizer, criterion_class, criterion_contrast, 
                criterion_ss, device, config, scaler, epoch):
    """Single training epoch"""
    model.train()
    
    total_loss = 0
    all_labels = []
    all_preds = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        sequences = batch['sequence_str']
        labels = batch['label'].to(device, non_blocking=True).float()
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
        with torch.amp.autocast(config.device_type, enabled=config.USE_AMP):
            outputs = model(
                sequences,
                attention_mask,
                contact_maps,
                node_geometric_feat,
                edge_index,
                edge_attr,
                node_coords
            )
            
            # Classification loss
            class_logits = outputs['class_logits']
            loss_class = criterion_class(class_logits, labels)
            
            # Contrastive loss
            loss_contrast = torch.tensor(0.0, device=device)
            if 'contrast_features' in outputs:
                contrast_features = outputs['contrast_features']
                loss_contrast = criterion_contrast(contrast_features, labels.long())
            
            # SS reconstruction loss
            loss_ss = torch.tensor(0.0, device=device)
            if 'ss_emission_scores' in outputs and config.USE_SS_RECON:
                ss_logits = outputs['ss_emission_scores']  # [B, L_actual, 3]
                ss_target_full = batch['ss_target'].to(device, non_blocking=True)  # [B, max_len]
                mask_full = attention_mask  # [B, max_len]
                
                # Align to actual sequence length (model dynamically adjusts)
                B, L_actual, C = ss_logits.shape
                ss_target = ss_target_full[:, :L_actual]  # [B, L_actual]
                mask = mask_full[:, :L_actual]  # [B, L_actual]
                
                ss_loss_dict = criterion_ss(ss_logits, ss_target, mask)
                loss_ss = ss_loss_dict['loss']
            
            # Total loss
            loss = (config.W_CLASS * loss_class + 
                   config.W_CONTRAST * loss_contrast +
                   config.W_SS_RECON * loss_ss)
        
        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Metrics
        total_loss += loss.item()
        probs = torch.sigmoid(class_logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
        
        # Progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{loss_class.item():.4f}',
            'con': f'{loss_contrast.item():.4f}',
            'ss': f'{loss_ss.item():.4f}'
        })
    
    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc


def validate(model, valid_loader, criterion_class, device, config):
    """Validation"""
    model.eval()
    
    total_loss = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating"):
            # Move to device
            sequences = batch['sequence_str']
            labels = batch['label'].to(device, non_blocking=True).float()
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
            
            # Loss
            class_logits = outputs['class_logits']
            loss = criterion_class(class_logits, labels)
            total_loss += loss.item()
            
            # Predictions
            probs = torch.sigmoid(class_logits).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    # Metrics
    avg_loss = total_loss / len(valid_loader)
    all_preds = (np.array(all_probs) > 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc
    }


def main():
    config = MultiAMPConfig()
    device = torch.device(config.DEVICE)
    
    print(f"Using device: {device}")
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    pdb_dirs_map = {
        1: config.AMP_PDB_DIR,
        0: config.NONAMP_PDB_DIR
    }
    train_dataset = PeptideDataset(
        data_path=config.TRAIN_DATA_PATH,
        pdb_dirs=pdb_dirs_map,
        max_len=config.MAX_SEQ_LEN,
        is_training=True,
        config=config
    )
    
    pdb_dirs_map_valid = {
        1: config.AMP_VALID_PDB_DIR,
        0: config.NONAMP_VALID_PDB_DIR
    }
    valid_dataset = PeptideDataset(
        data_path=config.VALID_DATA_PATH,
        pdb_dirs=pdb_dirs_map_valid,
        max_len=config.MAX_SEQ_LEN,
        is_training=False,
        config=config
    )
    
    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
    
    # Compute class weights for SS
    print("\n=== Computing SS Class Weights ===")
    ss_class_weights = compute_ss_class_weights(
        train_dataset,
        num_classes=3,
        ignore_index=-1
    ).to(device)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    print("\n=== Initializing Model ===")
    model = PeptideTriStreamModel(config).to(device)
    
    # Loss functions
    criterion_class = nn.BCEWithLogitsLoss()
    criterion_contrast = SupConLoss(temperature=config.TEMPERATURE)
    criterion_ss = StructureAwareLoss(
        num_classes=config.NUM_SS,
        class_weights=ss_class_weights,
        gamma=config.SS_FOCAL_GAMMA,
        continuity_weight=config.SS_CONTINUITY_WEIGHT,
        use_crf=config.SS_USE_CRF,
        ignore_index=config.RECON_IGNORE_INDEX
    ).to(device)
    
    # Optimizer and scheduler
    config.update_for_epoch(0)
    model._freeze_plm_layers()
    optimizer = model.get_optimizer()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.PLATEAU_FACTOR,
        patience=config.PLATEAU_PATIENCE,
        verbose=True
    )
    
    scaler = torch.amp.GradScaler(config.device_type, enabled=config.USE_AMP)
    
    # Training loop
    print("\n=== Starting Training ===")
    best_auc = 0
    best_epoch = -1
    
    for epoch in range(config.EPOCHS):
        # Update config for current epoch
        config.update_for_epoch(epoch)
        if epoch == config.WARMUP_EPOCHS:
            print(f"\n🔥 Unfreezing last {config.UNFREEZE_LAST_N} PLM layers")
            model._unfreeze_last_n_layers(config.UNFREEZE_LAST_N)
            optimizer = model.get_optimizer()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_class,
            criterion_contrast, criterion_ss, device, config, scaler, epoch
        )
        
        # Validate
        valid_metrics = validate(model, valid_loader, criterion_class, device, config)
        
        # Scheduler step
        scheduler.step(valid_metrics['auc'])
        
        # Print results
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Valid - Loss: {valid_metrics['loss']:.4f}, AUC: {valid_metrics['auc']:.4f}, "
              f"MCC: {valid_metrics['mcc']:.4f}, F1: {valid_metrics['f1']:.4f}, "
              f"Acc: {valid_metrics['acc']:.4f}")
        
        # Save best model
        if valid_metrics['auc'] > best_auc:
            best_auc = valid_metrics['auc']
            best_epoch = epoch
            torch.save(model.state_dict(), f"{config.SAVE_DIR}/best_model.pth")
            print(f"✅ Best model saved (AUC: {best_auc:.4f})")
    
    print(f"\n🎉 Training completed! Best AUC: {best_auc:.4f} at epoch {best_epoch+1}")


if __name__ == '__main__':
    import numpy as np
    main()
