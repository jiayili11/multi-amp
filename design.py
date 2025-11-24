"""
MultiAMP Design Script
Based on amppre/generate_figure5_data.py
Includes de novo design and motif-guided design
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd

from model import PeptideTriStreamModel
from config import MultiAMPConfig


AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def create_batch_for_design(sequences, device):
    """Create batch for design (sequence-only mode)"""
    batch_size = len(sequences)
    max_len = max(len(s) for s in sequences)
    
    # Attention mask
    attention_mask = []
    for seq in sequences:
        mask = [1] * len(seq) + [0] * (max_len - len(seq))
        attention_mask.append(mask)
    attention_mask = torch.tensor(attention_mask).float().to(device)
    
    # Edge index (simple sequential connections)
    all_edges = []
    for i, seq in enumerate(sequences):
        edges = [[j, j+1] for j in range(len(seq)-1)]
        if edges:
            edges = torch.tensor(edges).T
            edges = edges + i * max_len
            all_edges.append(edges)
    
    if all_edges:
        edge_index = torch.cat(all_edges, dim=1).to(device)
        edge_attr = torch.zeros(edge_index.shape[1], 27).to(device)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long).to(device)
        edge_attr = torch.zeros(0, 27).to(device)
    
    # Zero structure features
    contact_maps = torch.zeros(batch_size, max_len, max_len).to(device)
    node_geometric_feat = torch.zeros(batch_size, max_len, 15).to(device)
    node_coords = torch.zeros(batch_size, max_len, 3).to(device)
    
    return {
        'sequence_str': sequences,
        'attention_mask': attention_mask,
        'contact_maps': contact_maps,
        'node_geometric_feat': node_geometric_feat,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'node_coords': node_coords
    }


def optimize_sequence(model, initial_seq, device, n_iterations=100, lr=0.015, 
                     temp=0.5, reg_weight=0.6, motif_positions=None):
    """
    Optimize sequence using Gumbel-Softmax
    
    Args:
        model: Trained model
        initial_seq: Initial sequence string
        device: Device
        n_iterations: Number of optimization iterations
        lr: Learning rate
        temp: Gumbel-Softmax temperature
        reg_weight: Diversity regularization weight
        motif_positions: Dict {pos: aa} for fixed positions
    """
    model.train()  # Need gradients
    
    # Encode initial sequence
    seq_indices = [AA_TO_IDX[aa] for aa in initial_seq]
    seq_len = len(seq_indices)
    
    # Initialize logits
    logits = torch.zeros(seq_len, len(AA_LIST), device=device, requires_grad=True)
    for i, idx in enumerate(seq_indices):
        logits.data[i, idx] = 5.0  # Initialize around initial sequence
    
    optimizer = torch.optim.Adam([logits], lr=lr)
    
    best_score = -float('inf')
    best_seq = initial_seq
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        # Gumbel-Softmax sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        soft_seq = F.softmax((logits + gumbel_noise) / temp, dim=-1)  # [L, 20]
        
        # Hard sequence for forward pass
        hard_indices = torch.argmax(soft_seq, dim=-1)
        hard_seq = ''.join([AA_LIST[idx.item()] for idx in hard_indices])
        
        # Apply motif constraints
        if motif_positions:
            hard_seq_list = list(hard_seq)
            for pos, aa in motif_positions.items():
                if 0 <= pos < len(hard_seq_list):
                    hard_seq_list[pos] = aa
            hard_seq = ''.join(hard_seq_list)
        
        # Forward pass
        batch = create_batch_for_design([hard_seq], device)
        outputs = model(
            batch['sequence_str'],
            batch['attention_mask'],
            batch['contact_maps'],
            batch['node_geometric_feat'],
            batch['edge_index'],
            batch['edge_attr'],
            batch['node_coords']
        )
        
        amp_logit = outputs['class_logits']
        amp_score = torch.sigmoid(amp_logit)
        
        # Diversity regularization
        diversity_loss = -torch.sum(soft_seq ** 2)
        
        # Total loss (negative because we maximize)
        loss = -amp_score + reg_weight * diversity_loss
        
        loss.backward()
        optimizer.step()
        
        # Track best
        current_score = amp_score.item()
        if current_score > best_score:
            best_score = current_score
            best_seq = hard_seq
    
    model.eval()
    return best_seq, best_score


def design_de_novo(model, device, n_sequences=500, select_top=5, 
                  seq_length_range=(15, 45), n_iterations=100):
    """
    De novo AMP design
    
    Args:
        model: Trained model
        device: Device
        n_sequences: Number of candidates to generate
        select_top: Number of top candidates to return
        seq_length_range: (min_len, max_len) for sequences
        n_iterations: Optimization iterations per candidate
    """
    print(f"\n🧬 De Novo Design: Generating {n_sequences} candidates")
    
    results = []
    
    for i in tqdm(range(n_sequences), desc="Generating"):
        # Random initialization
        seq_len = np.random.randint(seq_length_range[0], seq_length_range[1])
        init_seq = ''.join(np.random.choice(AA_LIST, seq_len))
        
        # Optimize
        opt_seq, score = optimize_sequence(
            model, init_seq, device,
            n_iterations=n_iterations,
            lr=0.015,
            temp=0.5,
            reg_weight=0.6
        )
        
        results.append({
            'sequence': opt_seq,
            'amp_score': score,
            'length': len(opt_seq)
        })
    
    # Select top
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('amp_score', ascending=False)
    top_candidates = df_sorted.head(select_top)
    
    print(f"\n✅ Top {select_top} de novo designs:")
    for idx, row in top_candidates.iterrows():
        print(f"  {row['sequence'][:50]}... (score: {row['amp_score']:.4f}, len: {row['length']})")
    
    return top_candidates


def design_with_motif(model, device, motif_seq, motif_start, 
                      n_variants=100, select_top=5, total_length=25, 
                      n_iterations=100):
    """
    Motif-guided AMP design
    
    Args:
        model: Trained model
        device: Device
        motif_seq: Motif sequence (e.g., 'KKK')
        motif_start: Starting position of motif
        n_variants: Number of variants to generate
        select_top: Number of top variants to return
        total_length: Total sequence length
        n_iterations: Optimization iterations per variant
    """
    print(f"\n🎯 Motif-Guided Design: Motif '{motif_seq}' at position {motif_start}")
    
    # Build motif position dict
    motif_positions = {}
    for i, aa in enumerate(motif_seq):
        motif_positions[motif_start + i] = aa
    
    results = []
    
    for i in tqdm(range(n_variants), desc="Generating variants"):
        # Random initialization
        init_seq = ''.join(np.random.choice(AA_LIST, total_length))
        
        # Optimize with motif constraint
        opt_seq, score = optimize_sequence(
            model, init_seq, device,
            n_iterations=n_iterations,
            lr=0.015,
            temp=0.5,
            reg_weight=0.6,
            motif_positions=motif_positions
        )
        
        results.append({
            'sequence': opt_seq,
            'motif': motif_seq,
            'motif_start': motif_start,
            'amp_score': score,
            'length': len(opt_seq)
        })
    
    # Select top
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('amp_score', ascending=False)
    top_variants = df_sorted.head(select_top)
    
    print(f"\n✅ Top {select_top} motif-guided designs:")
    for idx, row in top_variants.iterrows():
        print(f"  {row['sequence'][:50]}... (score: {row['amp_score']:.4f})")
    
    return top_variants


def main():
    config = MultiAMPConfig()
    device = torch.device(config.DEVICE)
    
    print(f"Using device: {device}")
    
    # Load model
    print("\n=== Loading Model ===")
    model = PeptideTriStreamModel(config).to(device)
    
    model_path = f"{config.SAVE_DIR}/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    print(f"✅ Model loaded from {model_path}")
    
    # Create output directory
    output_dir = "./design_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Experiment 1: De novo design
    print("\n" + "="*70)
    print("Experiment 1: De Novo AMP Design")
    print("="*70)
    
    # Quick test version (reduced numbers for fast testing)
    denovo_results = design_de_novo(
        model, device,
        n_sequences=10,  # Reduced from 500 to 10
        select_top=5,
        n_iterations=50  # Reduced from 100 to 50
    )
    denovo_results.to_csv(f"{output_dir}/denovo_top5.csv", index=False)
    
    # Experiment 2: Motif-guided design
    print("\n" + "="*70)
    print("Experiment 2: Motif-Guided AMP Design")
    print("="*70)
    
    motifs = [
        ('KKK', 5),
        ('KRK', 8),
        ('KLLKL', 3),
    ]
    
    all_motif_results = []
    for motif_seq, motif_start in motifs:
        results = design_with_motif(
            model, device,
            motif_seq=motif_seq,
            motif_start=motif_start,
            n_variants=10,  # Reduced from 100 to 10
            select_top=3,   # Reduced from 5 to 3
            total_length=25,
            n_iterations=50  # Reduced from 100 to 50
        )
        all_motif_results.append(results)
    
    # Save all motif results
    motif_combined = pd.concat(all_motif_results, ignore_index=True)
    motif_combined.to_csv(f"{output_dir}/motif_designs.csv", index=False)
    
    print(f"\n💾 Results saved to {output_dir}/")
    print("✅ Design completed!")


if __name__ == '__main__':
    main()
