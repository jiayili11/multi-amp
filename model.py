"""
MultiAMP Model Architecture
============================
PeptideTriStreamModel: Three-stream model combining:
  - Stream 1: ESM-2 deep protein language model features (multi-scale fusion)
  - Stream 2: LSTM-based shallow sequence features
  - Stream 3: GVP-GNN structural features (geometric vector perceptron)
Fused via Cross-Attention + Transformer Encoder + Gated Fusion.

Supporting modules: PositionalEncoding, GVP, GVPConvLayer, GVP_GNN_Structural_Stream
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import esm
import math

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse

# ========================================
# 0. Positional Encoding
# ========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ========================================
# 1. GVP Core Component (Geometric Vector Perceptron)
# ========================================
class GVP(nn.Module):
    """
    Geometric Vector Perceptron
    Processes scalar features + vector features
    """
    def __init__(self, 
                 in_scalar_dim: int,
                 in_vector_dim: int,
                 out_scalar_dim: int,
                 out_vector_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_scalar_dim = in_scalar_dim
        self.in_vector_dim = in_vector_dim
        self.out_scalar_dim = out_scalar_dim
        self.out_vector_dim = out_vector_dim
        
        # Scalar -> Scalar
        self.Wss = nn.Linear(in_scalar_dim, out_scalar_dim, bias=False)
        # Vector -> Scalar (via vector norm)
        self.Wvs = nn.Linear(in_vector_dim, out_scalar_dim, bias=False)
        # Scalar -> Vector
        if out_vector_dim > 0:
            self.Wsv = nn.Linear(in_scalar_dim, out_vector_dim, bias=False)
            # Vector -> Vector
            self.Wvv = nn.Linear(in_vector_dim, out_vector_dim, bias=False)
        
        self.norm = nn.LayerNorm(out_scalar_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, s: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: Scalar features [N, in_scalar_dim]
            v: Vector features [N, in_vector_dim, 3]
        Returns:
            s_out: [N, out_scalar_dim]
            v_out: [N, out_vector_dim, 3]
        """
        # Vector norm (for v -> s)
        v_norm = torch.norm(v, dim=-1)  # [N, in_vector_dim]
        
        # Scalar output
        s_out = self.Wss(s) + self.Wvs(v_norm)
        s_out = self.norm(s_out)
        s_out = F.gelu(s_out)
        s_out = self.dropout(s_out)
        
        # Vector output (if needed)
        if self.out_vector_dim > 0:
            # v -> v: Preserve direction, adjust magnitude
            v_out = self.Wvv(v.transpose(1, 2)).transpose(1, 2)  # [N, out_vector_dim, 3]
            
            # s -> v: Adjust via gating mechanism
            gate = torch.sigmoid(self.Wsv(s)).unsqueeze(-1)  # [N, out_vector_dim, 1]
            v_out = v_out * gate
        else:
            v_out = None
        
        return s_out, v_out


# ========================================
# 2. GVP-GNN Message Passing Layer
# ========================================
class GVPConvLayer(nn.Module):
    """
    GVP Graph Convolution Layer
    """
    def __init__(self,
                 node_scalar_dim: int,
                 node_vector_dim: int,
                 edge_scalar_dim: int,
                 edge_vector_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # Message function: node features + edge features -> message
        self.message_gvp = GVP(
            in_scalar_dim=node_scalar_dim * 2 + edge_scalar_dim,
            in_vector_dim=node_vector_dim * 2 + edge_vector_dim,
            out_scalar_dim=hidden_dim,
            out_vector_dim=node_vector_dim,
            dropout=dropout
        )
        
        # Update function: aggregated messages -> new node features
        self.update_gvp = GVP(
            in_scalar_dim=node_scalar_dim + hidden_dim,
            in_vector_dim=node_vector_dim * 2,
            out_scalar_dim=node_scalar_dim,
            out_vector_dim=node_vector_dim,
            dropout=dropout
        )
    
    def forward(self, 
                node_s: torch.Tensor,
                node_v: torch.Tensor,
                edge_index: torch.Tensor,
                edge_s: torch.Tensor,
                edge_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_s: [N, node_scalar_dim]
            node_v: [N, node_vector_dim, 3]
            edge_index: [2, E]
            edge_s: [E, edge_scalar_dim]
            edge_v: [E, edge_vector_dim, 3]
        """
        src, dst = edge_index
        
        # Build messages
        msg_s = torch.cat([node_s[src], node_s[dst], edge_s], dim=-1)  # [E, ...]
        msg_v = torch.cat([node_v[src], node_v[dst], edge_v], dim=-2)  # [E, ..., 3]
        
        # Pass through GVP to get messages
        msg_s, msg_v = self.message_gvp(msg_s, msg_v)
        
        # Aggregate messages (sum aggregation)
        num_nodes = node_s.size(0)
        
        # Use msg_s/msg_v dtype and device for aggregation tensors
        aggr_s = torch.zeros(num_nodes, msg_s.size(1), 
                            dtype=msg_s.dtype, device=msg_s.device)
        aggr_v = torch.zeros(num_nodes, msg_v.size(1), 3, 
                            dtype=msg_v.dtype, device=msg_v.device)
        
        aggr_s.index_add_(0, dst, msg_s)
        aggr_v.index_add_(0, dst, msg_v)
        
        # Update nodes
        update_s = torch.cat([node_s, aggr_s], dim=-1)
        update_v = torch.cat([node_v, aggr_v], dim=-2)
        
        new_node_s, new_node_v = self.update_gvp(update_s, update_v)
        
        # Residual connection
        new_node_s = new_node_s + node_s
        new_node_v = new_node_v + node_v
        
        return new_node_s, new_node_v


# ========================================
# 3. GVP-GNN Structural Stream Module
# ========================================
class GVP_GNN_Structural_Stream(nn.Module):
    def __init__(self,
                 esm_dim: int,
                 geometric_dim: int,          # 15
                 edge_scalar_dim: int,        # 27
                 hidden_dim: int,
                 output_dim: int,
                 n_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.esm_dim = esm_dim
        self.geometric_dim = geometric_dim
        self.node_vector_dim = 4  # [CA direction, CB direction, backbone normal, local tangent]
        
        # Input projection
        # Scalars: ESM + geometric scalar features
        total_scalar_dim = esm_dim + (geometric_dim - 6)  # Exclude CA(3)+CB(3) coordinates
        self.node_scalar_project = nn.Linear(total_scalar_dim, hidden_dim)
        
        # Edge feature projection
        # Edge scalars: 27 - 3 (direction vector moved to vector features)
        edge_scalar_input = edge_scalar_dim - 3
        self.edge_scalar_project = nn.Linear(edge_scalar_input, hidden_dim)
        
        # GVP convolution layers
        self.gvp_layers = nn.ModuleList([
            GVPConvLayer(
                node_scalar_dim=hidden_dim,
                node_vector_dim=self.node_vector_dim,
                edge_scalar_dim=hidden_dim,
                edge_vector_dim=1,  # 边方向向量
                hidden_dim=hidden_dim,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Output layers
        self.node_output = nn.Linear(hidden_dim, esm_dim)
        self.graph_output = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def _compute_node_vectors(self, coords: torch.Tensor, cb_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute node vector features
        Args:
            coords: CA coordinates [N, 3]
            cb_coords: CB coordinates [N, 3]
        Returns:
            node_vectors: [N, 4, 3]
        """
        N = coords.size(0)
        vectors = []
        
        # 1. CA -> CB direction
        ca_cb = cb_coords - coords  # [N, 3]
        ca_cb_norm = F.normalize(ca_cb, dim=-1)
        vectors.append(ca_cb_norm)
        
        # 2. CA(i) -> CA(i+1) direction
        ca_next = torch.roll(coords, shifts=-1, dims=0)
        ca_direction = ca_next - coords
        ca_direction[-1] = 0  # Last residue
        ca_direction_norm = F.normalize(ca_direction, dim=-1)
        vectors.append(ca_direction_norm)
        
        # 3. Backbone normal vector (cross product)
        normal = torch.cross(ca_cb_norm, ca_direction_norm, dim=-1)
        normal_norm = F.normalize(normal, dim=-1)
        vectors.append(normal_norm)
        
        # 4. Local tangent vector (CA(i-1) -> CA(i+1))
        ca_prev = torch.roll(coords, shifts=1, dims=0)
        tangent = ca_next - ca_prev
        tangent[0] = 0
        tangent[-1] = 0
        tangent_norm = F.normalize(tangent, dim=-1)
        vectors.append(tangent_norm)
        
        return torch.stack(vectors, dim=1)  # [N, 4, 3]
    
    def forward(self,
                esm_features: torch.Tensor,        # [B*L, esm_dim]
                geometric_features: torch.Tensor,  # [B*L, 15]
                node_coords: torch.Tensor,         # [B*L, 3] CA coordinates
                edge_index: torch.Tensor,          # [2, E]
                edge_attr: torch.Tensor,           # [E, 27]
                batch_index: torch.Tensor):        # [B*L]
        """
        Args:
            geometric_features: [CA(3), CB(3), phi, psi, omega, curvature, SASA, properties(4)]
        """
        # Extract CB coordinates
        cb_coords = geometric_features[:, 3:6]  # [B*L, 3]
        
        # Scalar features: ESM + geometric features (excluding coordinates)
        geo_scalar = torch.cat([
            geometric_features[:, 6:7],   # phi
            geometric_features[:, 7:8],   # psi
            geometric_features[:, 8:9],   # omega
            geometric_features[:, 9:10],  # curvature
            geometric_features[:, 10:11], # rsa
            geometric_features[:, 11:15], # properties
        ], dim=-1)  # [B*L, 9]
        
        node_s = torch.cat([esm_features, geo_scalar], dim=-1)
        node_s = self.node_scalar_project(node_s)
        
        # Vector features
        node_v = self._compute_node_vectors(node_coords, cb_coords)  # [B*L, 4, 3]
        
        # Edge features
        # Scalars: remove direction vector (index 2:5)
        edge_s = torch.cat([
            edge_attr[:, 0:2],    # ca_dist, cb_dist
            edge_attr[:, 5:],     # seq_dist, rbf(16), ss_same, delta_angles(3), hbond
        ], dim=-1)  # [E, 24]
        edge_s = self.edge_scalar_project(edge_s)
        
        # Vector: edge direction vectors
        edge_v = edge_attr[:, 2:5].unsqueeze(1)  # [E, 1, 3]
        
        # GVP message passing
        for layer in self.gvp_layers:
            node_s, node_v = layer(node_s, node_v, edge_index, edge_s, edge_v)
        
        # Output
        updated_node_features = self.node_output(node_s)
        graph_features_for_pooling = self.graph_output(node_s)
        
        graph_embedding = global_mean_pool(graph_features_for_pooling, batch_index)
        
        return graph_embedding, updated_node_features


# ========================================
# 4. Full Three-Stream Model
# ========================================
class PeptideTriStreamModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- Stream 1: ESM-2 Deep Feature Stream ---
        if config.USE_ESM2:
            self.plm, self.alphabet = esm.pretrained.load_model_and_alphabet(config.PLM_NAME)
            self.tokenizer = self.alphabet.get_batch_converter()
            self.embed_dim = self.plm.embed_dim
            self.num_plm_layers = self.plm.num_layers
            self.scale_layers = [6, 12, 18, 24, self.num_plm_layers]
            self.layer_weights = nn.Parameter(torch.ones(len(self.scale_layers)))
        else:
            # 🔥 ESM-2 ablation: Use a simple embedding instead
            # We still need alphabet for tokenization, so load just for that
            _, self.alphabet = esm.pretrained.load_model_and_alphabet(config.PLM_NAME)
            self.tokenizer = self.alphabet.get_batch_converter()
            self.embed_dim = 1280  # ESM-2 default dimension
            # Create a simple embedding layer as replacement
            self.simple_plm_embedding = nn.Embedding(len(self.alphabet.all_toks), self.embed_dim,
                                                     padding_idx=self.alphabet.padding_idx)
        
        # --- Stream 2: Shallow Sequence Feature Stream ---
        raw_embed_dim = config.RAW_EMBED_DIM
        self.raw_aa_embedding = nn.Embedding(len(self.alphabet.all_toks), raw_embed_dim,
                                            padding_idx=self.alphabet.padding_idx)
        
        if config.USE_LSTM:
            self.raw_seq_encoder = nn.LSTM(
                raw_embed_dim, raw_embed_dim, config.RAW_LSTM_LAYERS,
                bidirectional=True, batch_first=True,
                dropout=config.DROPOUT_RATE if config.RAW_LSTM_LAYERS > 1 else 0
            )
            raw_feature_dim = raw_embed_dim * 2
        else:
            # 🔥 LSTM ablation: Use a simple linear layer to keep same output dimension
            self.raw_seq_encoder = None
            raw_feature_dim = raw_embed_dim
        
        # --- Fusion Module: Cross-Attention ---
        self.cross_attention = nn.MultiheadAttention(
            self.embed_dim, config.CROSS_ATTN_NHEAD,
            kdim=raw_feature_dim, vdim=raw_feature_dim,
            batch_first=True, dropout=config.DROPOUT_RATE
        )
        self.cross_attn_norm = nn.LayerNorm(self.embed_dim)
        
        # GNN feature integration
        self.structure_fusion_norm = nn.LayerNorm(self.embed_dim)
        
        # Positional encoding and deep fusion Encoder
        self.pos_encoder = PositionalEncoding(self.embed_dim, config.MAX_SEQ_LEN + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.TRANSFORMER_HEAD_NHEAD,
            dim_feedforward=config.TRANSFORMER_HEAD_DIM_FF,
            dropout=config.DROPOUT_RATE,
            batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.DECODER_LAYERS)
        
        # --- Stream 3: GVP-GNN Structural Stream ---
        self.gnn_stream = GVP_GNN_Structural_Stream(
            esm_dim=self.embed_dim,
            geometric_dim=15,  # Geometric feature dimension
            edge_scalar_dim=27,  # Edge feature dimension
            hidden_dim=config.GNN_HIDDEN_DIM,
            output_dim=config.GNN_OUTPUT_DIM,
            n_layers=config.GNN_LAYERS,
            dropout=config.DROPOUT_RATE
        )
        
        self.final_fusion_gate = nn.Sequential(
            nn.Linear(self.embed_dim + config.GNN_OUTPUT_DIM, 1),
            nn.Sigmoid()
        )
        
        final_fused_dim = self.embed_dim + config.GNN_OUTPUT_DIM
        
        # --- Downstream Task Heads ---
        self.classifier_head = nn.Sequential(
            nn.Linear(final_fused_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim // 2, 1))
        
        self.contrast_project_head = nn.Sequential(
            nn.Linear(final_fused_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, config.CONTRAST_FEATURE_DIM))

        self.ss_feature_extractor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE)
        )

        # 2. Bi-LSTM 捕获上下文依赖
        self.ss_bilstm = nn.LSTM(
            self.embed_dim // 2,
            self.embed_dim // 4,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        # 3. 发射分数层（用于 CRF）
        self.ss_emission_layer = nn.Linear(self.embed_dim // 2, config.NUM_SS)

        # 4. CRF 层（学习结构转移规律）
        from torchcrf import CRF
        self.ss_crf = CRF(config.NUM_SS, batch_first=True)
        
        # recon_hidden_dim = self.embed_dim // 2
        # self.ss_reconstructor_mlp = nn.Sequential(
        #     nn.Linear(self.embed_dim, recon_hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(config.DROPOUT_RATE))
        # self.ss_output_layer = nn.Linear(recon_hidden_dim, config.NUM_SS)
        
        self._init_head_weights()
        self._freeze_plm_layers()
    
    def _init_head_weights(self):
        heads = [
            self.raw_aa_embedding, self.cross_attention,
            self.cross_attn_norm, self.gnn_stream, self.classifier_head,
            self.contrast_project_head, self.ss_feature_extractor,
            self.ss_bilstm, self.ss_emission_layer, self.fusion_encoder, self.structure_fusion_norm,
            self.final_fusion_gate]
        
        # 🔥 Only add LSTM if enabled
        if self.config.USE_LSTM and self.raw_seq_encoder is not None:
            heads.append(self.raw_seq_encoder)
        
        # 🔥 Only add simple PLM embedding if ESM-2 is disabled
        if not self.config.USE_ESM2:
            heads.append(self.simple_plm_embedding)
        
        for head in heads:
            for module in head.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
    
    def _freeze_plm_layers(self):
        # 🔥 Skip if ESM-2 is disabled (ablation)
        if not self.config.USE_ESM2:
            return
        
        if not self.config.finetune_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
            return
        
        for param in self.plm.parameters():
            param.requires_grad = False
        
        if self.config.UNFREEZE_LAST_N > 0:
            for i in range(self.config.UNFREEZE_LAST_N):
                layer_idx = self.num_plm_layers - 1 - i
                if layer_idx >= 0:
                    for param in self.plm.layers[layer_idx].parameters():
                        param.requires_grad = True
    
    def forward(self,
                sequence_strs: List[str],
                attention_mask: torch.Tensor,
                contact_maps: torch.Tensor = None,
                node_geometric_feat: torch.Tensor = None,
                edge_index: torch.Tensor = None,
                edge_attr: torch.Tensor = None,
                node_coords: torch.Tensor = None,
                return_intermediate: bool = False,
                **kwargs) -> Dict[str, torch.Tensor]:
        
        # Get device from any available parameter
        if self.config.USE_ESM2:
            device = next(self.plm.parameters()).device
        else:
            device = next(self.simple_plm_embedding.parameters()).device
        batch_size = len(sequence_strs)

        # ========== 1. ESM-2 Encoding ==========
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequence_strs)]
        _, _, batch_tokens = self.tokenizer(data)
        batch_tokens = batch_tokens.to(device)
        
        if self.config.USE_ESM2:
            # 🔥 Use full ESM-2 model
            with torch.set_grad_enabled(self.training and self.config.finetune_plm):
                results = self.plm(batch_tokens, repr_layers=self.scale_layers, return_contacts=False)
                all_layer_embeddings = torch.stack([results["representations"][layer] for layer in self.scale_layers])
                normalized_weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
                esm_embeddings = (all_layer_embeddings * normalized_weights).sum(dim=0)
                esm_sequence_reps = esm_embeddings[:, 1:-1, :]  # Remove <cls> and <eos>
        else:
            # 🔥 ESM-2 ablation: Use simple embedding
            esm_sequence_reps = self.simple_plm_embedding(batch_tokens[:, 1:-1])

        # ========== 2. Raw Sequence Features ==========
        raw_tokens = batch_tokens[:, 1:-1]
        raw_embedded = self.raw_aa_embedding(raw_tokens)
        
        if self.config.USE_LSTM and self.raw_seq_encoder is not None:
            # 🔥 Use LSTM encoder
            raw_sequence_features, _ = self.raw_seq_encoder(raw_embedded)
        else:
            # 🔥 LSTM ablation: Use raw embeddings directly
            raw_sequence_features = raw_embedded

        # ========== 3. Align Sequence Lengths ==========
        actual_seq_len = min(
            esm_sequence_reps.size(1),
            raw_sequence_features.size(1),
            attention_mask.size(1)
        )
        
        # Also align geometric features if available
        if node_geometric_feat is not None and node_coords is not None:
            actual_seq_len = min(actual_seq_len, node_geometric_feat.size(1), node_coords.size(1))
        
        esm_sequence_reps = esm_sequence_reps[:, :actual_seq_len, :]
        raw_sequence_features = raw_sequence_features[:, :actual_seq_len, :]
        attention_mask_aligned = attention_mask[:, :actual_seq_len]

        # ========== 4. Cross-Attention Fusion ==========
        key_padding_mask = (attention_mask_aligned == 0)
        cross_attended_reps, _ = self.cross_attention(
            query=esm_sequence_reps,
            key=raw_sequence_features,
            value=raw_sequence_features,
            key_padding_mask=key_padding_mask
        )
        fused_sequence_reps = self.cross_attn_norm(cross_attended_reps + esm_sequence_reps)

        # ========== 5. GVP-GNN Structural Stream (Optional) ==========
        global_graph_embedding = None
        updated_node_features = None
        
        # Only run GNN when GVP is enabled and structural inputs are provided
        if (self.config.USE_GVP and 
            node_geometric_feat is not None and 
            edge_index is not None and 
            edge_attr is not None and 
            node_coords is not None):
            
            node_geometric_feat_aligned = node_geometric_feat[:, :actual_seq_len, :]
            node_coords_aligned = node_coords[:, :actual_seq_len, :]
            
            # Flatten to node level
            esm_node_features = fused_sequence_reps.reshape(-1, self.embed_dim)
            geometric_node_features = node_geometric_feat_aligned.reshape(-1, 15)
            node_coords_flat = node_coords_aligned.reshape(-1, 3)
            batch_index = torch.arange(batch_size, device=device).repeat_interleave(actual_seq_len)
            
            # Run GVP-GNN
            global_graph_embedding, updated_node_features = self.gnn_stream(
                esm_features=esm_node_features,
                geometric_features=geometric_node_features,
                node_coords=node_coords_flat,
                edge_index=edge_index.to(device),
                edge_attr=edge_attr.to(device),
                batch_index=batch_index
            )
            
            # Reshape and fuse
            updated_node_features = updated_node_features.reshape(batch_size, actual_seq_len, -1)
            reps_with_structure = self.structure_fusion_norm(fused_sequence_reps + updated_node_features)
        else:
            # Without GVP, use fused sequence features directly
            reps_with_structure = fused_sequence_reps
            # Create zero vector as placeholder for global graph embedding
            global_graph_embedding = torch.zeros(batch_size, self.config.GNN_OUTPUT_DIM, device=device)

        # ========== 6. Transformer Deep Fusion ==========
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat([cls_tokens, reps_with_structure], dim=1)
        encoder_input = self.pos_encoder(encoder_input)
        cls_mask = torch.ones(batch_size, 1, dtype=attention_mask_aligned.dtype, device=device)
        encoder_mask = torch.cat([cls_mask, attention_mask_aligned], dim=1)
        encoder_padding_mask = (encoder_mask == 0)
        encoder_output = self.fusion_encoder(encoder_input, src_key_padding_mask=encoder_padding_mask)
        sequence_summary_embedding = encoder_output[:, 0, :]
        deeply_fused_sequence_reps = encoder_output[:, 1:, :]

        # ========== 7. Final Fusion and Downstream Tasks ==========
        combined_final_embedding = torch.cat([sequence_summary_embedding, global_graph_embedding], dim=1)
        gate = self.final_fusion_gate(combined_final_embedding)
        final_feature = combined_final_embedding * gate

        # Classification
        class_logits = self.classifier_head(final_feature).squeeze(-1)
        
        # Contrastive learning
        contrast_features = self.contrast_project_head(final_feature)

        # Secondary structure reconstruction (optional)
        outputs = {
            'class_logits': class_logits,
            'cls_embedding': contrast_features,
        }
        
        if self.config.USE_SS_RECON:
            ss_features = self.ss_feature_extractor(deeply_fused_sequence_reps)
            ss_context, _ = self.ss_bilstm(ss_features)
            ss_emission_scores = self.ss_emission_layer(ss_context)
            outputs['ss_emission_scores'] = ss_emission_scores
            
            if not self.training:
                mask = attention_mask_aligned.bool()
                ss_predictions = self.ss_crf.decode(ss_emission_scores, mask=mask)
                outputs['ss_predictions'] = ss_predictions

        # Return intermediate features if requested
        if return_intermediate:
            outputs.update({
                'esm_embedding': esm_sequence_reps.mean(dim=1),
                'graph_embedding': global_graph_embedding,
                'transformer_cls': sequence_summary_embedding,
                'final_embedding': final_feature,
            })
            if self.config.USE_SS_RECON and 'ss_emission_scores' in outputs:
                outputs['predicted_ss'] = ss_emission_scores.argmax(dim=-1) if self.training else outputs.get('ss_predictions')

        return outputs
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        # 🔥 Handle ESM-2 ablation
        if self.config.USE_ESM2:
            plm_params = [p for p in self.plm.parameters() if p.requires_grad]
            head_params = [self.layer_weights, self.cls_token]
        else:
            plm_params = []
            head_params = list(self.simple_plm_embedding.parameters()) + [self.cls_token]
        
        # 🔥 Handle LSTM ablation
        head_params += list(self.raw_aa_embedding.parameters())
        if self.config.USE_LSTM and self.raw_seq_encoder is not None:
            head_params += list(self.raw_seq_encoder.parameters())
        
        # Add remaining components
        head_params += (
            list(self.cross_attention.parameters()) +
            list(self.cross_attn_norm.parameters()) +
            list(self.fusion_encoder.parameters()) +
            list(self.gnn_stream.parameters()) +
            list(self.structure_fusion_norm.parameters()) +
            list(self.final_fusion_gate.parameters()) +
            list(self.classifier_head.parameters()) +
            list(self.contrast_project_head.parameters()) +
            list(self.ss_feature_extractor.parameters()) +
            list(self.ss_bilstm.parameters()) +
            list(self.ss_emission_layer.parameters()) +
            list(self.ss_crf.parameters())  # CRF parameters
        )
        
        param_groups = [
            {"params": plm_params, "lr": self.config.PLM_LR},
            {"params": head_params, "lr": self.config.head_lr}
        ]
        return torch.optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)