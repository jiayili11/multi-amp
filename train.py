"""
MultiAMP Training Script
"""
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='MultiAMP Training')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: from config)')
    parser.add_argument('--lr', type=float, default=None, help='Head learning rate (default: from config)')
    parser.add_argument('--save_dir', type=str, default=None, help='Checkpoint save directory (default: from config)')
    parser.add_argument('--gpu', type=str, default='1', help='GPU device ID (default: 1)')
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
import pandas as pd

from model import PeptideTriStreamModel
from dataset import PeptideDataset, custom_collate_fn, SS_MAP
from losses import SupConLoss, StructureAwareLoss, compute_ss_class_weights
from config import MultiAMPConfig


def calculate_reconstruction_accuracy(logits, targets, ignore_index=-1):
    """Compute per-class secondary structure prediction accuracy."""
    preds = torch.argmax(logits, dim=-1)
    valid_mask = (targets != ignore_index) & (targets >= 0) & (targets < 3)
    
    if valid_mask.sum().item() == 0:
        return 0.0, {'H': 0.0, 'C': 0.0, 'E': 0.0}
    
    correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
    total = valid_mask.sum().item()
    overall_acc = correct / total
    
    class_names = ['H', 'C', 'E']
    class_acc = {}
    for c, name in enumerate(class_names):
        class_mask = (targets == c) & valid_mask
        if class_mask.sum().item() > 0:
            class_correct = (preds[class_mask] == targets[class_mask]).sum().item()
            class_total = class_mask.sum().item()
            class_acc[name] = class_correct / class_total
        else:
            class_acc[name] = 0.0
    
    return overall_acc, class_acc


def main():
    config = MultiAMPConfig()
    
    # Override config with CLI args
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.head_lr = args.lr
    if args.save_dir is not None:
        config.SAVE_DIR = args.save_dir
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # === 1. Data Loading ===
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
    
    # Compute SS class weights
    print("\n=== Computing SS Class Weights ===")
    ss_class_weights = compute_ss_class_weights(
        train_dataset, num_classes=3, ignore_index=-1
    ).to(device)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=custom_collate_fn, num_workers=4, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=4, pin_memory=True
    )
    
    # === 2. Model, Loss, Optimizer, Scheduler ===
    print("\n=== Initializing Model ===")
    model = PeptideTriStreamModel(config).to(device)
    
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
    
    scaler = torch.amp.GradScaler(config.device_type, enabled=config.USE_AMP)
    
    config.update_for_epoch(0)
    model._freeze_plm_layers()
    optimizer = model.get_optimizer()
    
    scheduler_type = getattr(config, 'SCHEDULER_TYPE', 'plateau')
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=config.PLATEAU_FACTOR,
            patience=config.PLATEAU_PATIENCE, verbose=True
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(config, 'COSINE_T_0', 10),
            T_mult=getattr(config, 'COSINE_T_MULT_RESTARTS', 1),
            eta_min=getattr(config, 'MIN_LR', 1e-7)
        )
    
    # === 3. Training Loop ===
    print("\n=== Starting Training ===")
    best_auc = 0
    best_epoch = -1
    w_contrast = config.W_CONTRAST
    w_ss = config.W_SS_RECON
    
    aux_decay_start = getattr(config, 'AUX_WEIGHT_DECAY_START_EPOCH', 25)
    aux_decay_rate = getattr(config, 'AUX_WEIGHT_DECAY_RATE', 0.90)
    
    for epoch in range(config.EPOCHS):
        config.update_for_epoch(epoch)
        
        # Warmup -> finetune transition
        if config.finetune_plm and epoch == config.WARMUP_EPOCHS:
            print(f"\nWarmup finished. Unfreezing last {config.UNFREEZE_LAST_N} PLM layers.")
            model._freeze_plm_layers()
            optimizer = model.get_optimizer()
            if scheduler_type == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=config.PLATEAU_FACTOR,
                    patience=config.PLATEAU_PATIENCE, verbose=True
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=getattr(config, 'COSINE_T_0', 10),
                    T_mult=getattr(config, 'COSINE_T_MULT_RESTARTS', 1),
                    eta_min=getattr(config, 'MIN_LR', 1e-7)
                )
        
        # Auxiliary loss weight decay in later epochs
        if epoch >= aux_decay_start:
            w_contrast *= aux_decay_rate
            w_ss *= aux_decay_rate
        
        # --- Training phase ---
        model.train()
        train_loss_total = 0.0
        train_loss_cls = 0.0
        train_loss_ss = 0.0
        train_loss_con = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for batch in pbar:
            seq_strs = batch['sequence_str']
            labels_float = batch['label'].to(device, non_blocking=True).float()
            labels_int = batch['label'].to(device, non_blocking=True).long()
            ss_targets = batch['ss_target'].to(device, non_blocking=True).long()
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            contact_maps = batch['contact_maps'].to(device, non_blocking=True)
            node_geometric_feat = batch['node_geometric_feat'].to(device, non_blocking=True)
            edge_index = batch['edge_index'].to(device, non_blocking=True)
            edge_attr = batch['edge_attr'].to(device, non_blocking=True)
            node_coords = batch['node_coords'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=config.device_type, dtype=torch.float16, enabled=config.USE_AMP):
                outputs = model(
                    seq_strs, attention_mask, contact_maps,
                    node_geometric_feat, edge_index, edge_attr, node_coords
                )
                
                loss_cls = criterion_class(outputs['class_logits'], labels_float)
                
                # SS loss
                loss_ss_dict = {'loss': torch.tensor(0.0, device=device), 'focal_loss': 0.0, 'crf_loss': 0.0}
                if 'ss_emission_scores' in outputs:
                    ss_emission = outputs['ss_emission_scores']
                    L_actual = ss_emission.size(1)
                    ss_mask = attention_mask[:, :L_actual]
                    loss_ss_dict = criterion_ss(ss_emission, ss_targets[:, :L_actual], ss_mask)
                loss_ss = loss_ss_dict['loss']
                
                # Contrastive loss
                loss_contrast = criterion_contrast(outputs['cls_embedding'], labels_int)
                
                total_loss = (
                    config.W_CLASS * loss_cls +
                    w_ss * loss_ss +
                    w_contrast * loss_contrast
                )
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_total += total_loss.item()
            train_loss_cls += loss_cls.item()
            train_loss_ss += loss_ss.item()
            train_loss_con += loss_contrast.item()
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'cls': f"{loss_cls.item():.4f}",
                'ss': f"{loss_ss.item():.4f}",
                'con': f"{loss_contrast.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
            })
        
        avg_train = {
            'total': train_loss_total / len(train_loader),
            'cls': train_loss_cls / len(train_loader),
            'ss': train_loss_ss / len(train_loader),
            'con': train_loss_con / len(train_loader),
        }
        
        # --- Validation phase ---
        model.eval()
        all_probs, all_labels = [], []
        val_loss_cls = 0.0
        val_loss_ss = 0.0
        total_ss_acc = 0.0
        all_class_acc = {'H': 0.0, 'C': 0.0, 'E': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                seq_strs = batch['sequence_str']
                labels = batch['label'].to(device, non_blocking=True).float()
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                contact_maps = batch['contact_maps'].to(device, non_blocking=True)
                node_geometric_feat = batch['node_geometric_feat'].to(device, non_blocking=True)
                edge_index = batch['edge_index'].to(device, non_blocking=True)
                edge_attr = batch['edge_attr'].to(device, non_blocking=True)
                node_coords = batch['node_coords'].to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=config.device_type, dtype=torch.float16, enabled=config.USE_AMP):
                    outputs = model(
                        seq_strs, attention_mask, contact_maps,
                        node_geometric_feat, edge_index, edge_attr, node_coords
                    )
                
                loss_cls = criterion_class(outputs['class_logits'], labels)
                val_loss_cls += loss_cls.item()
                
                if 'ss_emission_scores' in outputs:
                    ss_targets = batch['ss_target'].to(device, non_blocking=True).long()
                    ss_emission = outputs['ss_emission_scores']
                    L_actual = ss_emission.size(1)
                    ss_mask = attention_mask[:, :L_actual]
                    loss_ss_dict = criterion_ss(ss_emission, ss_targets[:, :L_actual], ss_mask)
                    val_loss_ss += loss_ss_dict['loss'].item()
                    
                    overall_acc, class_acc_dict = calculate_reconstruction_accuracy(
                        ss_emission, ss_targets[:, :L_actual]
                    )
                    total_ss_acc += overall_acc
                    for k, v in class_acc_dict.items():
                        all_class_acc[k] += v
                
                probs = torch.sigmoid(outputs['class_logits']).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch['label'].cpu().numpy())
        
        n_batches = len(valid_loader)
        avg_val_cls = val_loss_cls / n_batches
        avg_val_ss = val_loss_ss / n_batches
        avg_ss_acc = total_ss_acc / n_batches
        avg_class_acc = {k: v / n_batches for k, v in all_class_acc.items()}
        
        val_labels_np = np.array(all_labels)
        val_probs_np = np.array(all_probs)
        val_preds = (val_probs_np > 0.5).astype(int)
        
        val_acc = accuracy_score(val_labels_np, val_preds)
        val_auc = roc_auc_score(val_labels_np, val_probs_np)
        val_f1 = f1_score(val_labels_np, val_preds)
        val_precision = precision_score(val_labels_np, val_preds, zero_division=0)
        val_recall = recall_score(val_labels_np, val_preds, zero_division=0)
        val_mcc = matthews_corrcoef(val_labels_np, val_preds)
        
        # Scheduler step
        if scheduler_type == 'plateau':
            scheduler.step(val_auc)
        else:
            scheduler.step()
        
        # Print results
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} Summary ---")
        print(f"Train Loss -> Total: {avg_train['total']:.4f} | Cls: {avg_train['cls']:.4f} | SS: {avg_train['ss']:.4f} | Con: {avg_train['con']:.4f}")
        print(f"Valid Loss -> Cls: {avg_val_cls:.4f} | SS: {avg_val_ss:.4f}")
        print(f"Valid Metrics -> AUC: {val_auc:.4f} | MCC: {val_mcc:.4f} | F1: {val_f1:.4f} | Recall: {val_recall:.4f} | Precision: {val_precision:.4f} | Acc: {val_acc:.4f}")
        print(f"SS Recon Acc -> Overall: {avg_ss_acc:.4f} | {' | '.join([f'{k}: {v:.4f}' for k, v in avg_class_acc.items()])}")
        
        # Save best model
        # if val_auc > best_auc:
        #     best_auc = val_auc
        #     best_epoch = epoch + 1
        #     save_path = os.path.join(config.SAVE_DIR, "best_model_overall.pth")
        #     torch.save(model.state_dict(), save_path)
        #     print(f"New best model saved (AUC: {val_auc:.4f} at Epoch {best_epoch})")


if __name__ == '__main__':
    main()
