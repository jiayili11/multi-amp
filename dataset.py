"""
MultiAMP Dataset
=================
PeptideDataset: Loads peptide sequences with FASTA files and PDB structures.
Extracts geometric features (dihedrals, curvature, SASA, etc.) and builds
graph edges (KNN, sequential, contact, SS-motif) for GVP-GNN.
"""
import os
import torch
import numpy as np
import random
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import calc_dihedral, Vector
import warnings
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

# --- Constants ---
AA_MAP = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX-")}
SS_MAP = {ss: i for i, ss in enumerate("HCE-")}
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

PDB_TO_CANONICAL = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# ============ Geometric Feature Extraction ============

def calculate_dihedrals(coords):
    """Compute backbone dihedral angles phi, psi, omega"""
    L = len(coords['CA'])
    phi = np.zeros(L)
    psi = np.zeros(L)
    omega = np.zeros(L)
    
    for i in range(1, L-1):
        try:
            phi[i] = calc_dihedral(
                Vector(coords['C'][i-1]),
                Vector(coords['N'][i]),
                Vector(coords['CA'][i]),
                Vector(coords['C'][i])
            )
            psi[i] = calc_dihedral(
                Vector(coords['N'][i]),
                Vector(coords['CA'][i]),
                Vector(coords['C'][i]),
                Vector(coords['N'][i+1])
            )
            omega[i] = calc_dihedral(
                Vector(coords['CA'][i]),
                Vector(coords['C'][i]),
                Vector(coords['N'][i+1]),
                Vector(coords['CA'][i+1])
            )
        except:
            pass
    
    return phi, psi, omega


def calculate_local_curvature(coords_ca):
    """
    Compute local curvature (based on three-point angle).
    For each residue i, compute the angle formed by i-1, i, i+1.
    """
    L = len(coords_ca)
    curvature = np.zeros(L)
    
    for i in range(1, L-1):
        v1 = coords_ca[i] - coords_ca[i-1]
        v2 = coords_ca[i+1] - coords_ca[i]
        
        # Normalize
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Compute angle (radians)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Curvature = pi - angle (0 for straight, increases with bending)
        curvature[i] = np.pi - angle
    
    return curvature


def get_full_coords_from_pdb(pdb_path):
    """Extract backbone atom coordinates (N, CA, C, O) and CB"""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("peptide", pdb_path)
        coords = {'N': [], 'CA': [], 'C': [], 'O': [], 'CB': []}
        
        model = next(structure.get_models())
        chain = next(model.get_chains())
        
        for residue in chain.get_residues():
            if residue.get_resname() not in PDB_TO_CANONICAL:
                continue
            
            try:
                coords['N'].append(residue['N'].get_coord())
                coords['CA'].append(residue['CA'].get_coord())
                coords['C'].append(residue['C'].get_coord())
                coords['O'].append(residue['O'].get_coord())
                
                # CB (use virtual CB for glycine)
                if 'CB' in residue:
                    coords['CB'].append(residue['CB'].get_coord())
                else:
                    ca = residue['CA'].get_coord()
                    n = residue['N'].get_coord()
                    virtual_cb = ca + 0.5 * (ca - n)
                    coords['CB'].append(virtual_cb)
            except:
                for key in coords:
                    if coords[key] and len(coords[key]) > len(coords['CA']) - 1:
                        coords[key].pop()
                continue
        
        for key in coords:
            coords[key] = np.array(coords[key])
        
        return coords if len(coords['CA']) > 0 else None
    
    except Exception as e:
        return None


def calculate_sasa_simple(coords_ca, neighbor_cutoff=10.0):
    """Simplified SASA (solvent-accessible surface area) estimation"""
    dist_matrix = squareform(pdist(coords_ca, 'euclidean'))
    neighbor_counts = (dist_matrix < neighbor_cutoff).sum(axis=1) - 1
    max_neighbors = neighbor_counts.max() if neighbor_counts.max() > 0 else 1
    rsa = 1.0 - (neighbor_counts / max_neighbors)
    return rsa


# Amino acid physicochemical properties table
AA_PROPERTIES = {
    'A': [1.8, 0, 0, 89.1], 'C': [2.5, 0, 1, 121.2],
    'D': [-3.5, -1, 1, 133.1], 'E': [-3.5, -1, 1, 147.1],
    'F': [2.8, 0, 0, 165.2], 'G': [-0.4, 0, 0, 75.1],
    'H': [-3.2, 0.5, 1, 155.2], 'I': [4.5, 0, 0, 131.2],
    'K': [-3.9, 1, 1, 146.2], 'L': [3.8, 0, 0, 131.2],
    'M': [1.9, 0, 0, 149.2], 'N': [-3.5, 0, 1, 132.1],
    'P': [-1.6, 0, 0, 115.1], 'Q': [-3.5, 0, 1, 146.2],
    'R': [-4.5, 1, 1, 174.2], 'S': [-0.8, 0, 1, 105.1],
    'T': [-0.7, 0, 1, 119.1], 'V': [4.2, 0, 0, 117.1],
    'W': [-0.9, 0, 0, 204.2], 'Y': [-1.3, 0, 1, 181.2],
    'X': [0, 0, 0, 0],
}

def get_aa_properties(sequence):
    """Return [L, 4] physicochemical property matrix (hydropathy, charge, polarity, MW)"""
    return np.array([AA_PROPERTIES.get(aa, [0,0,0,0]) for aa in sequence])


# ============ Graph Construction ============

def build_knn_edges(coords_ca, k=20):
    """Build KNN graph"""
    L = len(coords_ca)
    dist_matrix = squareform(pdist(coords_ca, 'euclidean'))
    
    edges = []
    for i in range(L):
        neighbors = np.argsort(dist_matrix[i])[1:k+1]
        for j in neighbors:
            edges.append([i, j])
    
    return np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int64)


def build_sequential_edges(length, window=2):
    """Build sequential neighbor edges"""
    edges = []
    for i in range(length):
        for delta in range(1, window + 1):
            if i + delta < length:
                edges.append([i, i + delta])
                edges.append([i + delta, i])
    return np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int64)


def build_contact_edges(coords_ca, threshold=5.0):
    """
    Build contact graph edges (5 Angstrom threshold)
    """
    L = len(coords_ca)
    dist_matrix = squareform(pdist(coords_ca, 'euclidean'))
    
    edges = []
    for i in range(L):
        for j in range(i+1, L):
            if dist_matrix[i, j] < threshold:
                edges.append([i, j])
                edges.append([j, i])  # Bidirectional
    
    return np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int64)


def build_ss_motif_edges(ss_labels):
    """
    Build secondary structure motif graph.
    - Fully connect residues within the same helix/sheet segment
    """
    L = len(ss_labels)
    edges = []
    
    # Identify contiguous SS segments
    segments = []
    current_seg = [0]
    current_type = ss_labels[0]
    
    for i in range(1, L):
        if ss_labels[i] == current_type:
            current_seg.append(i)
        else:
            if current_type in ['H', 'E']:  # Only care about helix and sheet
                segments.append((current_type, current_seg))
            current_seg = [i]
            current_type = ss_labels[i]
    
    if current_type in ['H', 'E']:
        segments.append((current_type, current_seg))
    
    # Fully connect residues within the same segment
    for ss_type, seg in segments:
        for i in seg:
            for j in seg:
                if i != j:
                    edges.append([i, j])
    
    return np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int64)


def detect_hbonds(coords, distance_cutoff=3.5, angle_cutoff=120):
    """
    Simplified hydrogen bond detection.
    Detects N-H...O=C hydrogen bonds.
    Criteria: N...O distance < 3.5 Angstrom.
    """
    L = len(coords['N'])
    hbonds = []
    
    for i in range(L):
        for j in range(L):
            if abs(i - j) < 3:  # Skip residues too close in sequence
                continue
            
            # Compute N(i) to O(j) distance
            n_o_dist = np.linalg.norm(coords['N'][i] - coords['O'][j])
            
            if n_o_dist < distance_cutoff:
                # Simplified: consider as potential hydrogen bond
                hbonds.append((i, j))
    
    return hbonds


# ============ Edge Feature Computation ============

def calculate_edge_features_advanced(coords, edge_index, ss_labels, phi, psi, omega):
    """
    Compute enhanced edge features.
    Includes: distance, direction, dihedral angle difference, hydrogen bond features.
    """
    E = edge_index.shape[1]
    edge_features = []
    
    # Pre-compute hydrogen bonds
    hbonds_set = set(detect_hbonds(coords))
    
    for e in range(E):
        i, j = edge_index[:, e]
        
        # 1. Distance features
        ca_dist = np.linalg.norm(coords['CA'][i] - coords['CA'][j])
        cb_dist = np.linalg.norm(coords['CB'][i] - coords['CB'][j])
        
        # 2. Direction vector (normalized) for GVP
        direction = coords['CA'][j] - coords['CA'][i]
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
        
        # 3. Sequence distance
        seq_dist = abs(i - j)
        
        # 4. RBF distance encoding
        rbf_min, rbf_max, rbf_bins = 0.0, 20.0, 16
        rbf_centers = np.linspace(rbf_min, rbf_max, rbf_bins)
        rbf_gamma = 1.0
        rbf_encoded = np.exp(-rbf_gamma * (ca_dist - rbf_centers)**2)
        
        # 5. SS pairing
        ss_same = 1.0 if ss_labels[i] == ss_labels[j] else 0.0
        
        # 6. Dihedral angle differences
        delta_phi = phi[j] - phi[i]
        delta_psi = psi[j] - psi[i]
        delta_omega = omega[j] - omega[i]
        
        # 7. Hydrogen bond feature
        is_hbond = 1.0 if (i, j) in hbonds_set or (j, i) in hbonds_set else 0.0
        
        # Concatenate
        feat = np.concatenate([
            [ca_dist, cb_dist],          # 2
            direction_norm,               # 3 (for GVP)
            [seq_dist],                  # 1
            rbf_encoded,                 # 16
            [ss_same],                   # 1
            [delta_phi, delta_psi, delta_omega],  # 3
            [is_hbond]                   # 1
        ])
        edge_features.append(feat)
    
    return np.array(edge_features)  # [E, 27]


def get_coords_from_pdb(pdb_path):
    """Parse C-alpha coordinates from PDB file (backward-compatible)"""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("peptide", pdb_path)
        coords = []
        model = next(structure.get_models())
        chain = next(model.get_chains())
        for residue in chain.get_residues():
            if "CA" in residue and residue.get_resname() in PDB_TO_CANONICAL:
                coords.append(residue["CA"].get_coord())
        return np.array(coords) if coords else None
    except Exception as e:
        return None


def get_contact_map(coords, threshold):
    """Compute contact map from coordinates (backward-compatible)"""
    if coords is None or len(coords) == 0:
        return None
    distance_matrix = squareform(pdist(coords, 'euclidean'))
    return (distance_matrix < threshold).astype(np.float32)


# ============ Dataset ============

class PeptideDataset(Dataset):
    def __init__(self, data_path, pdb_dirs, max_len, is_training=False, config=None):
        self.max_len = max_len
        self.is_training = is_training
        self.config = config
        
        # Parameters
        self.use_augmentation = getattr(config, 'USE_AUGMENTATION', False) if config else False
        self.aug_prob = getattr(config, 'AUG_PROB', 0.5) if config else 0.5
        self.aug_aa_sub_prob = getattr(config, 'AUG_AA_SUB_PROB', 0.1) if config else 0.1
        self.use_noise_for_recon = getattr(config, 'RECON_USE_NOISE', False) if config else False
        self.mask_prob_for_recon = getattr(config, 'RECON_MASK_PROB', 0.15) if config else 0.15
        
        self.augmentation_factor = 1
        if self.is_training and self.use_augmentation:
            self.augmentation_factor = getattr(config, 'AUGMENTATION_FACTOR', 1)
        
        self.contact_threshold = getattr(config, 'CONTACT_MAP_THRESHOLD', 8.0) if config else 8.0
        
        self.knn_k = getattr(config, 'KNN_K', 20) if config else 20
        self.contact_cutoff = getattr(config, 'CONTACT_CUTOFF', 5.0) if config else 5.0
        self.use_ss_motif_graph = getattr(config, 'USE_SS_MOTIF_GRAPH', True) if config else True
        
        # Load data
        self.records = []
        num_pdb_not_found = 0
        num_pdb_parse_failed = 0
        
        print(f"Loading records from {data_path} and linking to PDBs...")
        if not os.path.exists(data_path) or not os.path.isdir(data_path):
            print(f"Error: Data path '{data_path}' does not exist.")
            return
        
        for filename in os.listdir(data_path):
            if filename.endswith(".fas"):
                filepath = os.path.join(data_path, filename)
                try:
                    for record in SeqIO.parse(filepath, "fasta"):
                        full_string = str(record.seq)
                        if len(full_string) % 2 != 0:
                            continue
                        split_point = len(full_string) // 2
                        if len(full_string[:split_point]) != len(full_string[split_point:]):
                            continue
                        
                        seq_id, label_str = record.id.split('|')
                        label = int(label_str)
                        
                        pdb_filename = f"{seq_id}.pdb"
                        pdb_path = os.path.join(pdb_dirs[label], pdb_filename)
                        
                        if not os.path.exists(pdb_path):
                            num_pdb_not_found += 1
                            continue
                        
                        # Validate PDB structure
                        coords = get_full_coords_from_pdb(pdb_path)
                        original_seq = full_string[:split_point]
                        if coords is None or len(coords['CA']) != len(original_seq):
                            num_pdb_parse_failed += 1
                            continue
                        
                        self.records.append((record, pdb_path))
                        
                except Exception as e:
                    print(f"Warning: Could not parse file {filepath}. Error: {e}")
        
        if num_pdb_not_found > 0:
            print(f"Warning: {num_pdb_not_found} records skipped due to missing PDB files.")
        if num_pdb_parse_failed > 0:
            print(f"Warning: {num_pdb_parse_failed} records skipped due to PDB parsing issues.")
        
        print(f"Successfully loaded {len(self.records)} valid records with both FASTA and PDB data.")
        self.actual_num_records = len(self.records)
        
        if self.is_training and self.augmentation_factor > 1:
            print(f"Dataset augmentation enabled. Effective size: {len(self)}")
    
    def __len__(self):
        return self.actual_num_records * self.augmentation_factor
    
    def __getitem__(self, idx):
        actual_idx = idx % self.actual_num_records
        record, pdb_path = self.records[actual_idx]
        
        # Parse FASTA
        seq_id, label_str = record.id.split('|')
        label = int(label_str)
        full_string = str(record.seq)
        split_point = len(full_string) // 2
        original_seq = full_string[:split_point]
        ss = full_string[split_point:]
        
        # Extract full coordinates
        coords = get_full_coords_from_pdb(pdb_path)
        if coords is None or len(coords['CA']) != len(original_seq):
            return None
        
        # Data augmentation
        input_seq_list = list(original_seq)
        if self.is_training and self.use_augmentation and random.random() < self.aug_prob:
            num_to_sub = int(len(input_seq_list) * self.aug_aa_sub_prob)
            if num_to_sub > 0:
                indices_to_sub = random.sample(range(len(input_seq_list)), num_to_sub)
                for i in indices_to_sub:
                    original_aa = input_seq_list[i]
                    possible_replacements = [aa for aa in STANDARD_AMINO_ACIDS if aa != original_aa]
                    if possible_replacements:
                        input_seq_list[i] = random.choice(possible_replacements)
        
        # Add noise for reconstruction task
        if self.is_training and self.use_noise_for_recon and self.mask_prob_for_recon > 0:
            for i in range(len(input_seq_list)):
                if random.random() < self.mask_prob_for_recon:
                    rand_val = random.random()
                    if rand_val < 0.8:
                        input_seq_list[i] = 'X'
                    elif rand_val < 0.9:
                        input_seq_list[i] = random.choice(STANDARD_AMINO_ACIDS)
        
        final_input_seq_str = "".join(input_seq_list)
        ss_target_encoded = [SS_MAP.get(s, SS_MAP['-']) for s in ss]
        
        # Truncation
        original_len = len(final_input_seq_str)
        if original_len > self.max_len:
            final_input_seq_str = final_input_seq_str[:self.max_len]
            ss_target_encoded = ss_target_encoded[:self.max_len]
            ss = ss[:self.max_len]
            for key in coords:
                coords[key] = coords[key][:self.max_len]
            original_len = self.max_len
        
        # ========== Compute Geometric Features ==========
        # 1. Dihedral angles
        phi, psi, omega = calculate_dihedrals(coords)
        
        # 2. Local curvature
        curvature = calculate_local_curvature(coords['CA'])
        
        # 3. SASA proxy
        rsa = calculate_sasa_simple(coords['CA'])
        
        # 4. Physicochemical properties
        aa_props = get_aa_properties(final_input_seq_str)
        
        # 5. Concatenate node geometric features
        node_geometric_features = np.concatenate([
            coords['CA'],           # [L, 3]
            coords['CB'],           # [L, 3]
            phi[:, None],           # [L, 1]
            psi[:, None],           # [L, 1]
            omega[:, None],         # [L, 1]
            curvature[:, None],     # [L, 1]
            rsa[:, None],           # [L, 1]
            aa_props,               # [L, 4]
        ], axis=1)  # [L, 15]
        
        # ========== Build Multi-Graph Edges ==========
        knn_edges = build_knn_edges(coords['CA'], k=self.knn_k)
        seq_edges = build_sequential_edges(original_len, window=2)
        contact_edges = build_contact_edges(coords['CA'], threshold=self.contact_cutoff)
        
        # SS-motif graph
        if self.use_ss_motif_graph:
            ss_edges = build_ss_motif_edges(list(ss))
        else:
            ss_edges = np.zeros((2, 0), dtype=np.int64)
        
        # Merge all edges
        all_edges = np.concatenate([knn_edges, seq_edges, contact_edges, ss_edges], axis=1)
        all_edges = np.unique(all_edges, axis=1)  # Deduplicate
        
        # ========== Compute Enhanced Edge Features ==========
        edge_attr = calculate_edge_features_advanced(
            coords, all_edges, list(ss), phi, psi, omega
        )  # [E, 27]
        
        # Contact map (backward-compatible)
        contact_map = get_contact_map(coords['CA'], self.contact_threshold)
        if contact_map is None:
            return None
        
        # Padding
        padding_len = self.max_len - original_len
        if padding_len > 0:
            ss_target_encoded += [-1] * padding_len
            attention_mask = [1] * original_len + [0] * padding_len
            
            # Pad geometric features
            node_geometric_features = np.pad(
                node_geometric_features,
                ((0, padding_len), (0, 0)),
                'constant'
            )
        else:
            attention_mask = [1] * original_len
        
        return {
            'id': seq_id,
            'sequence_str': final_input_seq_str,
            'label': torch.tensor(label, dtype=torch.float),
            'ss_target': torch.tensor(ss_target_encoded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'contact_map': torch.from_numpy(contact_map),
            
            # Graph features
            'node_geometric_feat': torch.from_numpy(node_geometric_features).float(),  # [max_len, 15]
            'edge_index': torch.from_numpy(all_edges).long(),  # [2, E]
            'edge_attr': torch.from_numpy(edge_attr).float(),  # [E, 27]
            
            # Node coordinates for GVP
            'node_coords': torch.from_numpy(coords['CA'][:original_len]).float(),  # [L, 3]
        }


def custom_collate_fn(batch):
    """Custom collate function for batching graph data"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Standard fields
    ids = [item['id'] for item in batch]
    seq_strs = [item['sequence_str'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    ss_targets = torch.stack([item['ss_target'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
    # Pad contact maps
    max_len_in_batch = max(item['contact_map'].shape[0] for item in batch)
    padded_contact_maps = []
    for item in batch:
        cm = item['contact_map']
        seq_len = cm.shape[0]
        padding_size = max_len_in_batch - seq_len
        padded_cm = torch.nn.functional.pad(cm, (0, padding_size, 0, padding_size), "constant", 0)
        padded_contact_maps.append(padded_cm)
    contact_maps = torch.stack(padded_contact_maps)
    
    # Geometric features
    node_geometric_feats = torch.stack([item['node_geometric_feat'] for item in batch])  # [B, L, 15]
    
    # Pad node coordinates for GVP
    padded_coords = []
    for item in batch:
        coords = item['node_coords']
        seq_len = coords.shape[0]
        padding_size = max_len_in_batch - seq_len
        padded_coord = torch.nn.functional.pad(coords, (0, 0, 0, padding_size), "constant", 0)
        padded_coords.append(padded_coord)
    node_coords = torch.stack(padded_coords)  # [B, L, 3]
    
    # Merge edges
    batch_edge_index = []
    batch_edge_attr = []
    offset = 0
    
    for item in batch:
        edge_index = item['edge_index'] + offset
        batch_edge_index.append(edge_index)
        batch_edge_attr.append(item['edge_attr'])
        offset += max_len_in_batch
    
    batch_edge_index = torch.cat(batch_edge_index, dim=1)  # [2, total_E]
    batch_edge_attr = torch.cat(batch_edge_attr, dim=0)    # [total_E, 27]
    
    return {
        'id': ids,
        'sequence_str': seq_strs,
        'label': labels,
        'ss_target': ss_targets,
        'attention_mask': attention_masks,
        'contact_maps': contact_maps,
        
        # Graph features
        'node_geometric_feat': node_geometric_feats,  # [B, L, 15]
        'edge_index': batch_edge_index,               # [2, E]
        'edge_attr': batch_edge_attr,                 # [E, 27]
        'node_coords': node_coords,                   # [B, L, 3]
    }