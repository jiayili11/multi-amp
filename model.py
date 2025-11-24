import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
from typing import Dict, List, Tuple

class BaselinePeptideModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 加载ESM-2模型
        self.plm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.tokenizer = self.alphabet.get_batch_converter()
        self.embed_dim = self.plm.embed_dim  # 1280
        
        # 多尺度层配置
        self.scale_layers = [6, 12, 18, 24, 33]
        
        # 改进的MLP分类头 (3层)
        self.classifier_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 4, 1)
        )
        
        # 初始化MLP权重
        self._init_mlp_weights()
        
        # 冻结PLM参数
        self._freeze_plm()

    def _init_mlp_weights(self):
        """Xavier初始化MLP权重"""
        for layer in self.classifier_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _freeze_plm(self):
        """冻结PLM参数"""
        if not getattr(self.config, 'finetune_plm', False):
            for param in self.plm.parameters():
                param.requires_grad = False

    def forward(self, sequence_strs):
        device = next(self.plm.parameters()).device
        
        # 1. Tokenization
        data = [(f"seq_{i}", seq[:1022]) for i, seq in enumerate(sequence_strs)]
        _, _, batch_tokens = self.tokenizer(data)
        batch_tokens = batch_tokens.to(device)
        
        # 2. 多尺度特征提取
        with torch.set_grad_enabled(self.training and getattr(self.config, 'finetune_plm', False)):
            results = self.plm(
                batch_tokens,
                repr_layers=self.scale_layers,
                return_contacts=False
            )
            
            # 3. 特征融合 (层深加权)
            embeddings = torch.stack([
                results["representations"][layer] 
                for layer in self.scale_layers
            ])
            weights = torch.linspace(0.5, 1.5, len(self.scale_layers), device=device)
            embeddings = (embeddings * F.softmax(weights, dim=0).view(-1, 1, 1, 1)).sum(dim=0)
        
        # 4. CLS Token池化
        pooled_output = embeddings[:, 0, :]  # [batch, embed_dim]
        
        # 5. 通过MLP分类头
        class_logits = self.classifier_head(pooled_output).squeeze(-1)
        
        return {'class_logits': class_logits}

    def get_optimizer(self):
        """优化器配置（适配MLP头）"""
        param_groups = [
            {
                "params": self.classifier_head.parameters(),
                "lr": getattr(self.config, 'head_lr', 1e-4)  # MLP使用更高学习率
            },
        ]
        
        if getattr(self.config, 'finetune_plm', False):
            param_groups.insert(0, {
                "params": self.plm.parameters(),
                "lr": getattr(self.config, 'PLM_LR', 1e-5)
            })
            
        return torch.optim.AdamW(
            param_groups,
            weight_decay=getattr(self.config, 'WEIGHT_DECAY', 1e-4)
        )


class PeptideReconstructorModel_old(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


        self.plm, self.alphabet = esm.pretrained.load_model_and_alphabet(config.PLM_NAME)
        self.tokenizer = self.alphabet.get_batch_converter()
        self.embed_dim = self.plm.embed_dim # 1280 for 650M model

        self.num_plm_layers = self.plm.num_layers # 33 for 650M model
        
        self.classifier_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim // 4, 1)
        )

        self.sequence_reconstructor = nn.Linear(self.embed_dim, self.config.NUM_AA)

        self.ss_reconstructor = nn.Linear(self.embed_dim, self.config.NUM_SS)

        self._init_head_weights()

        self._freeze_plm_layers()


    def _init_head_weights(self):
        for head in [self.classifier_head, self.sequence_reconstructor, self.ss_reconstructor]:
            for layer in head.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _freeze_plm_layers(self):
        if getattr(self.config, 'finetune_plm', False):
            for param in self.plm.parameters():
                param.requires_grad = False
            
            if self.config.UNFREEZE_LAST_N > 0:
                for i in range(self.config.UNFREEZE_LAST_N):
                    layer_idx_to_unfreeze = self.num_plm_layers - 1 - i
                    if layer_idx_to_unfreeze >= 0:
                        for param in self.plm.layers[layer_idx_to_unfreeze].parameters():
                            param.requires_grad = True
        else: 
            for param in self.plm.parameters():
                param.requires_grad = False

    def forward(self, sequence_strs: List[str], attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = next(self.plm.parameters()).device
        
        data = [(f"seq_{i}", seq[:self.config.MAX_SEQ_LEN - 2]) for i, seq in enumerate(sequence_strs)]
        _, _, batch_tokens = self.tokenizer(data)
        batch_tokens = batch_tokens.to(device)

        repr_layers_indices = [6, 12, 18, 24, 33]
        with torch.set_grad_enabled(self.training and getattr(self.config, 'finetune_plm', False)):
            results = self.plm(
                batch_tokens,
                repr_layers=repr_layers_indices, 
                return_contacts=False
            )
        
        token_embeddings = results["representations"][self.num_plm_layers]

        cls_embedding = token_embeddings[:, 0, :] # [batch, embed_dim]
        class_logits = self.classifier_head(cls_embedding).squeeze(-1) # [batch]

        sequence_representations = token_embeddings[:, 1 : -1, :] # [batch, seq_len, embed_dim]

        seq_recon_logits = self.sequence_reconstructor(sequence_representations) # [batch, seq_len, NUM_AA]
        ss_recon_logits = self.ss_reconstructor(sequence_representations)     # [batch, seq_len, NUM_SS]
        
        return {
            'class_logits': class_logits,
            'seq_recon_logits': seq_recon_logits,
            'ss_recon_logits': ss_recon_logits
        }

    def get_optimizer(self) -> torch.optim.Optimizer:
        plm_params = [p for p in self.plm.parameters() if p.requires_grad]
        head_params = list(self.classifier_head.parameters()) + \
                      list(self.sequence_reconstructor.parameters()) + \
                      list(self.ss_reconstructor.parameters())

        param_groups = [
            {"params": plm_params, "lr": self.config.PLM_LR},
            {"params": head_params, "lr": self.config.head_lr}
        ]

        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.WEIGHT_DECAY
        )
    

class PeptideReconstructorModel_old2(nn.Module):
    """
    一个集成了分类、对比学习、重建和高级特征融合策略的终极模型。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. --- 加载ESM-2模型 ---
        self.plm, self.alphabet = esm.pretrained.load_model_and_alphabet(config.PLM_NAME)
        self.tokenizer = self.alphabet.get_batch_converter()
        self.embed_dim = self.plm.embed_dim
        self.num_plm_layers = self.plm.num_layers
        
        # 2. --- 多尺度融合配置 (新增可学习权重) ---
        self.scale_layers = [6, 12, 18, 24, self.num_plm_layers]
        self.layer_weights = nn.Parameter(torch.ones(len(self.scale_layers)))

        # 3. --- 新增: Attention Pooling模块 ---
        self.attention_head = nn.Linear(self.embed_dim, 1)

        # 4. --- 特征融合与任务头定义 ---
        # 融合后的特征维度是 [CLS] (D) + Attention Pooling (D) = 2 * D
        fused_feature_dim = self.embed_dim * 2

        # a. 分类头 (输入维度更新为 2*D)
        self.classifier_head = nn.Sequential(
            nn.Linear(fused_feature_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim // 2, 1)
        )
        
        # b. 对比学习投影头 (输入维度更新为 2*D)
        contrast_dim = getattr(config, 'CONTRAST_FEATURE_DIM', 128)
        self.contrast_project_head = nn.Sequential(
            nn.Linear(fused_feature_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, contrast_dim)
        )

        # c. 共享重建MLP头 (保持不变，输入为D，因为它处理的是序列token)
        recon_hidden_dim = self.embed_dim // 2
        self.shared_reconstructor_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, recon_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE)
        )

        # d. 重建输出层 (保持不变)
        #self.sequence_output_layer = nn.Linear(recon_hidden_dim, config.NUM_AA)
        self.ss_output_layer = nn.Linear(recon_hidden_dim, config.NUM_SS)

        # 5. --- 初始化 & 冻结 ---
        self._init_head_weights()
        self._freeze_plm_layers()

    def _init_head_weights(self):
        """初始化所有自定义头的权重"""
        heads_to_initialize = [
            self.attention_head,
            self.classifier_head, 
            self.contrast_project_head,
            self.shared_reconstructor_mlp,
            #self.sequence_output_layer,
            self.ss_output_layer
        ]
        for head in heads_to_initialize:
            for layer in head.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _freeze_plm_layers(self):
        """根据配置冻结或解冻PLM层"""
        if not getattr(self.config, 'finetune_plm', False):
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

    def forward(self, sequence_strs: List[str], attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        device = next(self.plm.parameters()).device
        
        # 1. --- 分词 ---
        data = [(f"seq_{i}", seq[:self.config.MAX_SEQ_LEN - 2]) for i, seq in enumerate(sequence_strs)]
        _, _, batch_tokens = self.tokenizer(data)
        batch_tokens = batch_tokens.to(device)

        # 2. --- ESM-2多尺度特征提取 ---
        with torch.set_grad_enabled(self.training and getattr(self.config, 'finetune_plm', False)):
            results = self.plm(batch_tokens, repr_layers=self.scale_layers, return_contacts=False)
        
        # 3. --- 可学习的特征融合 ---
        all_layer_embeddings = torch.stack([results["representations"][layer] for layer in self.scale_layers])
        normalized_weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        fused_embeddings = (all_layer_embeddings * normalized_weights).sum(dim=0)
        
        # 4. --- 特征分离与池化 ---
        # [CLS] token 用于全局特征
        cls_embedding_raw = fused_embeddings[:, 0, :]
        
        # 序列 token 用于Attention Pooling和重建
        sequence_reps = fused_embeddings[:, 1:-1, :]
        
        # Attention Pooling
        attention_scores = self.attention_head(sequence_reps)
        seq_len = sequence_reps.size(1)
        mask = attention_mask[:, :seq_len].unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(mask == 0, -10000.0) # 使用安全值
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_pooled_embedding = (sequence_reps * attention_weights).sum(dim=1)
        
        # 5. --- 准备任务输入 ---
        # 用于分类和对比学习的融合特征
        final_feature_for_cls_contrast = torch.cat([cls_embedding_raw, attention_pooled_embedding], dim=1)
        
        # 用于重建的特征 (就是原始的序列token表征)
        features_for_recon = sequence_reps

        # 6. --- 任务分支计算 ---
        # a. 分类
        class_logits = self.classifier_head(final_feature_for_cls_contrast).squeeze(-1)

        # b. 对比学习
        contrast_features = self.contrast_project_head(final_feature_for_cls_contrast)
        
        # c. 重建
        shared_recon_features = self.shared_reconstructor_mlp(features_for_recon)
        #seq_recon_logits = self.sequence_output_layer(shared_recon_features)
        ss_recon_logits = self.ss_output_layer(shared_recon_features)

        return {
            'class_logits': class_logits,
            'cls_embedding': contrast_features,
            #'seq_recon_logits': seq_recon_logits,
            'ss_recon_logits': ss_recon_logits,
        }

    def get_optimizer(self) -> torch.optim.Optimizer:
        """配置优化器，为PLM和所有头部分配不同的学习率"""
        plm_params = [p for p in self.plm.parameters() if p.requires_grad]
        
        head_params = (
            [self.layer_weights] + # 修复了 non-leaf tensor 的问题
            list(self.attention_head.parameters()) +
            list(self.classifier_head.parameters()) +
            list(self.contrast_project_head.parameters()) +
            list(self.shared_reconstructor_mlp.parameters()) +
            list(self.ss_output_layer.parameters())
        )
        #            list(self.sequence_output_layer.parameters()) +

        param_groups = [
            {"params": plm_params, "lr": self.config.PLM_LR},
            {"params": head_params, "lr": self.config.head_lr}
        ]
        
        return torch.optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)
    

class PeptideReconstructorModel(nn.Module):
    """
    终极版模型v2: 
    集成了分类、对比、SS重建、Attention Pooling、可学习融合、
    级联SS特征、以及一致性正则化支持。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. --- ESM-2 & 融合配置 (不变) ---
        self.plm, self.alphabet = esm.pretrained.load_model_and_alphabet(config.PLM_NAME)
        self.tokenizer = self.alphabet.get_batch_converter()
        self.embed_dim = self.plm.embed_dim
        self.num_plm_layers = self.plm.num_layers
        self.scale_layers = [6, 12, 18, 24, self.num_plm_layers]
        self.layer_weights = nn.Parameter(torch.ones(len(self.scale_layers)))

        # 2. --- Attention Pooling模块 (不变) ---
        self.attention_head = nn.Linear(self.embed_dim, 1)

        # 3. --- SS重建模块 (结构调整) ---
        # 为了得到更丰富的特征，我们将SS预测头也做得深一些
        recon_hidden_dim = self.embed_dim // 2
        self.ss_reconstructor_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, recon_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        self.ss_output_layer = nn.Linear(recon_hidden_dim, config.NUM_SS)
        
        self.ss_feature_processor = nn.Sequential(
            nn.Linear(config.NUM_SS, 64), # 将4维的SS概率映射到64维
            nn.GELU(),
        )
        pooled_ss_feature_dim = 64

        # 5. --- 特征融合与最终任务头 ---
        # 新的融合特征维度 = [CLS](D) + AttnPool(D) + SS_feature(64)
        final_fused_dim = self.embed_dim * 2 + pooled_ss_feature_dim

        # a. 分类头 (输入维度再次更新)
        self.classifier_head = nn.Sequential(
            nn.Linear(final_fused_dim, self.embed_dim),
            nn.GELU(), nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(), nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim // 2, 1)
        )
        
        # b. 对比学习投影头 (输入维度再次更新)
        contrast_dim = getattr(config, 'CONTRAST_FEATURE_DIM', 128)
        self.contrast_project_head = nn.Sequential(
            nn.Linear(final_fused_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, contrast_dim)
        )
        
        self._init_head_weights()
        self._freeze_plm_layers()

    def _init_head_weights(self):
        # (更新需要初始化的头列表)
        heads_to_initialize = [
            self.attention_head, self.ss_feature_processor,
            self.classifier_head, self.contrast_project_head,
            self.ss_reconstructor_mlp, self.ss_output_layer
        ]
        for head in heads_to_initialize:
            for layer in head.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None: nn.init.zeros_(layer.bias)

    def _freeze_plm_layers(self):
        """根据配置冻结或解冻PLM层"""
        if not getattr(self.config, 'finetune_plm', False):
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

    def forward(self, sequence_strs: List[str], attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        一次完整的前向传播。
        对于KL散度，我们需要调用这个函数两次（一次正常，一次用dropout后的模型）
        """
        device = next(self.plm.parameters()).device
        
        # 1. & 2. & 3. --- 特征提取和融合 (不变) ---
        data = [(f"seq_{i}", seq[:self.config.MAX_SEQ_LEN - 2]) for i, seq in enumerate(sequence_strs)]
        _, _, batch_tokens = self.tokenizer(data)
        batch_tokens = batch_tokens.to(device)
        with torch.set_grad_enabled(self.training and getattr(self.config, 'finetune_plm', False)):
            results = self.plm(batch_tokens, repr_layers=self.scale_layers, return_contacts=False)
        all_layer_embeddings = torch.stack([results["representations"][layer] for layer in self.scale_layers])
        normalized_weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        fused_embeddings = (all_layer_embeddings * normalized_weights).sum(dim=0)
        
        # 4. --- 特征分离 (不变) ---
        cls_embedding_raw = fused_embeddings[:, 0, :]
        sequence_reps = fused_embeddings[:, 1:-1, :]
        
        # 5. --- 第一次任务分支：计算SS预测 ---
        # 这是级联结构的第一步：先得到SS预测
        ss_recon_hidden = self.ss_reconstructor_mlp(sequence_reps)
        ss_recon_logits = self.ss_output_layer(ss_recon_hidden)

        # 6. --- 新增：处理SS特征以用于分类 ---
        # a. 得到SS预测概率
        ss_probs = F.softmax(ss_recon_logits, dim=-1) # [B, L, num_ss]
        
        # b. 池化SS特征。这里我们用mean pooling，忽略padding
        seq_len = sequence_reps.size(1)
        mask = attention_mask[:, :seq_len].unsqueeze(-1) # [B, L, 1]
        # 只对非padding部分求和，然后除以非padding的长度
        pooled_ss_probs = (ss_probs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8) # [B, num_ss]
        
        # c. 通过一个MLP进一步处理池化后的SS特征
        processed_ss_feature = self.ss_feature_processor(pooled_ss_probs) # [B, 64]

        # 7. --- Attention Pooling (不变) ---
        attention_scores = self.attention_head(sequence_reps)
        attention_scores = attention_scores.masked_fill(mask == 0, -10000.0)
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_pooled_embedding = (sequence_reps * attention_weights).sum(dim=1)
        
        # 8. --- 构建最终的融合特征 ---
        final_fused_feature = torch.cat([
            cls_embedding_raw, 
            attention_pooled_embedding, 
            processed_ss_feature.detach() # 使用.detach()来阻断梯度
        ], dim=1)

        class_logits = self.classifier_head(final_fused_feature).squeeze(-1)
        contrast_features = self.contrast_project_head(final_fused_feature)

        return {
            'class_logits': class_logits,
            'cls_embedding': contrast_features,
            'ss_recon_logits': ss_recon_logits,
        }
    def get_optimizer(self):
        plm_params = [p for p in self.plm.parameters() if p.requires_grad]
        head_params = (
            [self.layer_weights] +
            list(self.attention_head.parameters()) +
            list(self.classifier_head.parameters()) +
            list(self.contrast_project_head.parameters()) +
            list(self.ss_reconstructor_mlp.parameters()) +
            list(self.ss_output_layer.parameters()) +
            list(self.ss_feature_processor.parameters())
        )
        param_groups = [
            {"params": plm_params, "lr": self.config.PLM_LR},
            {"params": head_params, "lr": self.config.head_lr}
        ]
        return torch.optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)
    


import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
from typing import Dict, List

# 假设你的Config类可以通过 from config import EnhancedConfig 导入
# 并且包含了 TRANSFORMER_HEAD_NHEAD, TRANSFORMER_HEAD_DIM_FF, CONTRAST_FEATURE_DIM 等参数

class PeptideTransformerHeadModel(nn.Module):
    """
    终极版模型v3:
    引入Gated Fusion和Transformer Encoder Head来追求极致性能。
    - 使用ESM-2作为强大的序列编码器。
    - 采用可学习的权重来融合ESM-2的多层表征。
    - 使用门控机制(Gated Fusion)智能融合[CLS], Attention Pooling和SS预测三种特征。
    - 使用一个Transformer Encoder层作为分类头，以捕捉特征间的高阶交互。
    - 保留了对比学习和二级结构重建作为辅助任务。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. --- ESM-2 & 融合配置 ---
        self.plm, self.alphabet = esm.pretrained.load_model_and_alphabet(config.PLM_NAME)
        self.tokenizer = self.alphabet.get_batch_converter()
        self.embed_dim = self.plm.embed_dim
        self.num_plm_layers = self.plm.num_layers
        self.scale_layers = [6, 12, 18, 24, self.num_plm_layers]
        self.layer_weights = nn.Parameter(torch.ones(len(self.scale_layers)))

        # 2. --- Attention Pooling模块 ---
        self.attention_head = nn.Linear(self.embed_dim, 1)

        # 3. --- SS重建与特征处理模块 ---
        recon_hidden_dim = self.embed_dim // 2
        self.ss_reconstructor_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, recon_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        self.ss_output_layer = nn.Linear(recon_hidden_dim, config.NUM_SS)
        
        pooled_ss_feature_dim = 64
        self.ss_feature_processor = nn.Sequential(
            nn.Linear(config.NUM_SS, pooled_ss_feature_dim),
            nn.GELU(),
        )
        
        # 4. --- Gated Fusion 模块 ---
        gate_input_dim = self.embed_dim * 2 + pooled_ss_feature_dim
        self.fusion_gate = nn.Linear(gate_input_dim, 3)
        # 填充SS特征以匹配embed_dim，用于加权求和
        self.ss_feature_padding_dim = self.embed_dim - pooled_ss_feature_dim

        # 5. --- Transformer Encoder Head ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=getattr(config, 'TRANSFORMER_HEAD_NHEAD', 8),
            dim_feedforward=getattr(config, 'TRANSFORMER_HEAD_DIM_FF', 2048),
            dropout=config.DROPOUT_RATE,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder_head = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 6. --- 最终的输出层 ---
        self.final_classifier = nn.Linear(self.embed_dim, 1)

        contrast_dim = getattr(config, 'CONTRAST_FEATURE_DIM', 128)
        self.contrast_project_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, contrast_dim)
        )
        
        self._init_head_weights()
        self._freeze_plm_layers()

    def _init_head_weights(self):
        """初始化所有自定义头的权重"""
        heads_to_initialize = [
            self.attention_head, self.ss_feature_processor, self.fusion_gate,
            self.transformer_encoder_head, self.final_classifier, 
            self.contrast_project_head,
            self.ss_reconstructor_mlp, self.ss_output_layer
        ]
        for head in heads_to_initialize:
            for layer in head.modules():
                if isinstance(layer, (nn.Linear, nn.LayerNorm)):
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    elif isinstance(layer, nn.LayerNorm):
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)

    def _freeze_plm_layers(self):
        """根据配置冻结或解冻PLM层"""
        if not getattr(self.config, 'finetune_plm', False):
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

    def _attention_pooling(self, reps, mask):
        """对序列token表征进行注意力池化"""
        attention_scores = self.attention_head(reps)
        seq_len = reps.size(1)
        attn_mask = mask[:, :seq_len].unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(attn_mask == 0, -10000.0)
        attention_weights = F.softmax(attention_scores, dim=1)
        return (reps * attention_weights).sum(dim=1)

    def _process_ss_features(self, logits, mask):
        """对SS预测logits进行池化和处理"""
        ss_probs = F.softmax(logits, dim=-1)
        seq_len = logits.size(1)
        ss_mask = mask[:, :seq_len].unsqueeze(-1)
        pooled_ss_probs = (ss_probs * ss_mask).sum(dim=1) / (ss_mask.sum(dim=1) + 1e-8)
        return self.ss_feature_processor(pooled_ss_probs)

    def forward(self, sequence_strs: List[str], attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        device = next(self.plm.parameters()).device
        
        # 1. 分词
        data = [(f"seq_{i}", seq[:self.config.MAX_SEQ_LEN - 2]) for i, seq in enumerate(sequence_strs)]
        _, _, batch_tokens = self.tokenizer(data)
        batch_tokens = batch_tokens.to(device)

        # 2. ESM-2多尺度特征提取
        with torch.set_grad_enabled(self.training and getattr(self.config, 'finetune_plm', False)):
            results = self.plm(batch_tokens, repr_layers=self.scale_layers, return_contacts=False)
        
        # 3. 可学习的特征融合
        all_layer_embeddings = torch.stack([results["representations"][layer] for layer in self.scale_layers])
        normalized_weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        fused_embeddings = (all_layer_embeddings * normalized_weights).sum(dim=0)
        
        # 4. 特征分离
        cls_embedding_raw = fused_embeddings[:, 0, :]
        sequence_reps = fused_embeddings[:, 1:-1, :]
        
        # 5. SS预测 (级联第一步)
        ss_recon_hidden = self.ss_reconstructor_mlp(sequence_reps)
        ss_recon_logits = self.ss_output_layer(ss_recon_hidden)

        # 6. Gated Fusion
        # a. 准备三个特征源
        attention_pooled_embedding = self._attention_pooling(sequence_reps, attention_mask)
        processed_ss_feature = self._process_ss_features(ss_recon_logits, attention_mask)

        # b. 计算gate值
        gate_input = torch.cat([
            cls_embedding_raw.detach(), 
            attention_pooled_embedding.detach(), 
            processed_ss_feature.detach()
        ], dim=1)
        gate_values = F.softmax(self.fusion_gate(gate_input), dim=-1).unsqueeze(-1)
        
        # c. 特征源加权
        ss_feat_padded = F.pad(processed_ss_feature, (0, self.ss_feature_padding_dim))
        
        stacked_features = torch.stack([
            cls_embedding_raw, 
            attention_pooled_embedding, 
            ss_feat_padded
        ], dim=1)
        
        gated_fused_feature = (stacked_features * gate_values).sum(dim=1)

        # 7. 通过Transformer Encoder Head
        processed_feature = self.transformer_encoder_head(gated_fused_feature.unsqueeze(1)).squeeze(1)

        # 8. 任务输出
        class_logits = self.final_classifier(processed_feature).squeeze(-1)
        contrast_features = self.contrast_project_head(processed_feature)

        return {
            'class_logits': class_logits,
            'cls_embedding': contrast_features,
            'ss_recon_logits': ss_recon_logits,
        }
        
    def get_optimizer(self) -> torch.optim.Optimizer:
        """配置优化器，为PLM和所有头部分配不同的学习率"""
        plm_params = [p for p in self.plm.parameters() if p.requires_grad]
        
        head_params = (
            [self.layer_weights] +
            list(self.attention_head.parameters()) +
            list(self.ss_feature_processor.parameters()) +
            list(self.fusion_gate.parameters()) +
            list(self.transformer_encoder_head.parameters()) +
            list(self.final_classifier.parameters()) +
            list(self.contrast_project_head.parameters()) +
            list(self.ss_reconstructor_mlp.parameters()) +
            list(self.ss_output_layer.parameters())
        )
        
        param_groups = [
            {"params": plm_params, "lr": self.config.PLM_LR},
            {"params": head_params, "lr": self.config.head_lr}
        ]
        
        return torch.optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)


class PeptideDualStreamModel(nn.Module):
    """
    终极版模型v4:
    引入双流架构，使用Cross-Attention融合ESM-2的深度语义特征和原始序列的浅层特征。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- 流1: ESM-2 深度特征流 ---
        self.plm, self.alphabet = esm.pretrained.load_model_and_alphabet(config.PLM_NAME)
        self.tokenizer = self.alphabet.get_batch_converter()
        self.embed_dim = self.plm.embed_dim
        self.num_plm_layers = self.plm.num_layers
        self.scale_layers = [6, 12, 18, 24, self.num_plm_layers]
        self.layer_weights = nn.Parameter(torch.ones(len(self.scale_layers)))
        
        # --- 流2: 原始序列浅层特征流 ---
        raw_embed_dim = getattr(config, 'RAW_EMBED_DIM', 128)
        self.raw_aa_embedding = nn.Embedding(
            num_embeddings=len(self.alphabet.all_toks),
            embedding_dim=raw_embed_dim,
            padding_idx=self.alphabet.padding_idx
        )
        
        self.raw_seq_encoder = nn.LSTM(
            input_size=raw_embed_dim,
            hidden_size=raw_embed_dim,
            num_layers=getattr(config, 'RAW_LSTM_LAYERS', 2),
            bidirectional=True,
            batch_first=True,
            dropout=config.DROPOUT_RATE if getattr(config, 'RAW_LSTM_LAYERS', 2) > 1 else 0
        )
        raw_feature_dim = raw_embed_dim * 2
        
        # --- 融合模块: Cross-Attention ---
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=getattr(config, 'CROSS_ATTN_NHEAD', 8),
            kdim=raw_feature_dim,
            vdim=raw_feature_dim,
            batch_first=True,
            dropout=config.DROPOUT_RATE
        )
        self.cross_attn_norm = nn.LayerNorm(self.embed_dim)
        
        # --- 下游任务头 ---
        # a. Attention Pooling (作用于融合后的序列特征)
        self.attention_head = nn.Linear(self.embed_dim, 1)

        # b. 分类头 (输入是 [CLS] + AttnPool)
        fused_feature_dim = self.embed_dim * 2
        self.classifier_head = nn.Sequential(
            nn.Linear(fused_feature_dim, self.embed_dim),
            nn.GELU(), nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(), nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim // 2, 1)
        )

        # c. 对比学习投影头
        contrast_dim = getattr(config, 'CONTRAST_FEATURE_DIM', 128)
        self.contrast_project_head = nn.Sequential(
            nn.Linear(fused_feature_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, contrast_dim)
        )
        
        # d. SS重建头 (现在接收融合后的序列特征)
        recon_hidden_dim = self.embed_dim // 2
        self.ss_reconstructor_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, recon_hidden_dim),
            nn.GELU(), nn.Dropout(config.DROPOUT_RATE)
        )
        self.ss_output_layer = nn.Linear(recon_hidden_dim, config.NUM_SS)
        
        self._init_head_weights()
        self._freeze_plm_layers()

    def _init_head_weights(self):
        """初始化所有自定义头的权重"""
        heads_to_initialize = [
            self.raw_aa_embedding, self.raw_seq_encoder, self.cross_attention, 
            self.cross_attn_norm, self.attention_head, self.classifier_head,
            self.contrast_project_head, self.ss_reconstructor_mlp, self.ss_output_layer
        ]
        for head in heads_to_initialize:
            for layer in head.modules():
                if isinstance(layer, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        if layer.bias is not None: nn.init.zeros_(layer.bias)
                    elif isinstance(layer, nn.Embedding):
                        nn.init.normal_(layer.weight, mean=0, std=0.02)
                    elif isinstance(layer, nn.LayerNorm):
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)
        # 对LSTM进行特殊的正交初始化，有助于稳定训练
        for name, param in self.raw_seq_encoder.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)


    def _freeze_plm_layers(self):
        """根据配置冻结或解冻PLM层"""
        if not getattr(self.config, 'finetune_plm', False):
            for param in self.plm.parameters(): param.requires_grad = False
            return
        
        for param in self.plm.parameters(): param.requires_grad = False
        
        if self.config.UNFREEZE_LAST_N > 0:
            for i in range(self.config.UNFREEZE_LAST_N):
                layer_idx = self.num_plm_layers - 1 - i
                if layer_idx >= 0:
                    for param in self.plm.layers[layer_idx].parameters():
                        param.requires_grad = True

    def _attention_pooling(self, reps, mask):
        """对序列token表征进行注意力池化"""
        attention_scores = self.attention_head(reps)
        seq_len = reps.size(1)
        attn_mask = mask[:, :seq_len].unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(attn_mask == 0, -10000.0)
        attention_weights = F.softmax(attention_scores, dim=1)
        return (reps * attention_weights).sum(dim=1)

    def forward(self, sequence_strs: List[str], attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        device = next(self.plm.parameters()).device
        
        # 1. --- 分词 ---
        data = [(f"seq_{i}", seq[:self.config.MAX_SEQ_LEN - 2]) for i, seq in enumerate(sequence_strs)]
        _, _, batch_tokens = self.tokenizer(data)
        batch_tokens = batch_tokens.to(device)

        # 2. --- 流1: ESM-2深度特征处理 ---
        with torch.set_grad_enabled(self.training and getattr(self.config, 'finetune_plm', False)):
            results = self.plm(batch_tokens, repr_layers=self.scale_layers, return_contacts=False)
        all_layer_embeddings = torch.stack([results["representations"][layer] for layer in self.scale_layers])
        normalized_weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        esm_embeddings = (all_layer_embeddings * normalized_weights).sum(dim=0)
        
        esm_cls_embedding = esm_embeddings[:, 0, :]
        esm_sequence_reps = esm_embeddings[:, 1:-1, :]
        
        # 3. --- 流2: 原始序列浅层特征处理 ---
        raw_embedded = self.raw_aa_embedding(batch_tokens)
        raw_sequence_embedded = raw_embedded[:, 1:-1, :]
        raw_sequence_features, _ = self.raw_seq_encoder(raw_sequence_embedded)
        
        # 4. --- 融合: Cross-Attention ---
        actual_seq_len = raw_sequence_features.size(1)
        key_padding_mask = (attention_mask[:, :actual_seq_len] == 0)

        cross_attended_reps, _ = self.cross_attention(
            query=esm_sequence_reps,
            key=raw_sequence_features,
            value=raw_sequence_features,
            key_padding_mask=key_padding_mask
        )
        
        fused_sequence_reps = self.cross_attn_norm(cross_attended_reps + esm_sequence_reps)

        # 5. --- 下游任务 ---
        attention_pooled_embedding = self._attention_pooling(fused_sequence_reps, attention_mask)
        
        final_feature_for_cls_contrast = torch.cat([esm_cls_embedding, attention_pooled_embedding], dim=1)
        
        class_logits = self.classifier_head(final_feature_for_cls_contrast).squeeze(-1)
        contrast_features = self.contrast_project_head(final_feature_for_cls_contrast)
        
        ss_recon_hidden = self.ss_reconstructor_mlp(fused_sequence_reps)
        ss_recon_logits = self.ss_output_layer(ss_recon_hidden)

        return {
            'class_logits': class_logits,
            'cls_embedding': contrast_features,
            'ss_recon_logits': ss_recon_logits,
        }
        
    def get_optimizer(self) -> torch.optim.Optimizer:
        """配置优化器，为PLM和所有头部分配不同的学习率"""
        plm_params = [p for p in self.plm.parameters() if p.requires_grad]
        
        head_params = (
            [self.layer_weights] +
            list(self.raw_aa_embedding.parameters()) +
            list(self.raw_seq_encoder.parameters()) +
            list(self.cross_attention.parameters()) +
            list(self.cross_attn_norm.parameters()) +
            list(self.attention_head.parameters()) +
            list(self.classifier_head.parameters()) +
            list(self.contrast_project_head.parameters()) +
            list(self.ss_reconstructor_mlp.parameters()) +
            list(self.ss_output_layer.parameters())
        )
        
        param_groups = [
            {"params": plm_params, "lr": self.config.PLM_LR},
            {"params": head_params, "lr": self.config.head_lr}
        ]
        
        return torch.optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)

import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
from typing import Dict, List

class PeptideDecoderModel(nn.Module):
    """
    终极版模型v5:
    引入标准的Transformer Decoder结构进行双流融合，并优化各任务的信息流。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- 流1: ESM-2 深度特征流 ---
        self.plm, self.alphabet = esm.pretrained.load_model_and_alphabet(config.PLM_NAME)
        self.tokenizer = self.alphabet.get_batch_converter()
        self.embed_dim = self.plm.embed_dim
        self.num_plm_layers = self.plm.num_layers
        self.scale_layers = [6, 12, 18, 24, self.num_plm_layers]
        self.layer_weights = nn.Parameter(torch.ones(len(self.scale_layers)))

        # --- 流2: 原始序列浅层特征流 ---
        raw_embed_dim = getattr(config, 'RAW_EMBED_DIM', 128)
        self.raw_aa_embedding = nn.Embedding(
            num_embeddings=len(self.alphabet.all_toks),
            embedding_dim=raw_embed_dim,
            padding_idx=self.alphabet.padding_idx
        )
        self.raw_seq_encoder = nn.LSTM(
            input_size=raw_embed_dim,
            hidden_size=raw_embed_dim,
            num_layers=getattr(config, 'RAW_LSTM_LAYERS', 2),
            bidirectional=True,
            batch_first=True,
            dropout=config.DROPOUT_RATE if getattr(config, 'RAW_LSTM_LAYERS', 2) > 1 else 0
        )
        raw_feature_dim = raw_embed_dim * 2

        # --- 融合模块: Transformer Decoder ---
        # Decoder的输入memory需要与d_model维度匹配，增加一个线性投影层
        self.memory_projection = nn.Linear(raw_feature_dim, self.embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=getattr(config, 'CROSS_ATTN_NHEAD', 8),
            dim_feedforward=self.embed_dim * 4,
            dropout=config.DROPOUT_RATE,
            activation='gelu',
            batch_first=True
        )
        self.fusion_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=getattr(config, 'DECODER_LAYERS', 1)
        )
        
        # --- 下游任务头 ---
        # a. Attention Pooling
        self.attention_head = nn.Linear(self.embed_dim, 1)

        # b. 强化的分类头
        fused_feature_dim = self.embed_dim * 2
        self.classifier_head = nn.Sequential(
            nn.Linear(fused_feature_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.embed_dim // 2, 1)
        )

        # c. 对比学习投影头
        contrast_dim = getattr(config, 'CONTRAST_FEATURE_DIM', 128)
        self.contrast_project_head = nn.Sequential(
            nn.Linear(fused_feature_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, contrast_dim)
        )
        
        # d. SS重建头
        recon_hidden_dim = self.embed_dim // 2
        self.ss_reconstructor_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, recon_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        self.ss_output_layer = nn.Linear(recon_hidden_dim, config.NUM_SS)
        
        self._init_head_weights()
        self._freeze_plm_layers()

    def _init_head_weights(self):
        """初始化所有自定义头的权重"""
        heads_to_initialize = [
            self.raw_aa_embedding, self.raw_seq_encoder, self.memory_projection,
            self.fusion_decoder, self.attention_head, self.classifier_head,
            self.contrast_project_head, self.ss_reconstructor_mlp, self.ss_output_layer
        ]
        for head in heads_to_initialize:
            for layer in head.modules():
                if isinstance(layer, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        if layer.bias is not None: nn.init.zeros_(layer.bias)
                    elif isinstance(layer, nn.Embedding):
                        nn.init.normal_(layer.weight, mean=0, std=0.02)
                    elif isinstance(layer, nn.LayerNorm):
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)
        # 对LSTM进行特殊的正交初始化
        for name, param in self.raw_seq_encoder.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def _freeze_plm_layers(self):
        """根据配置冻结或解冻PLM层"""
        if not getattr(self.config, 'finetune_plm', False):
            for param in self.plm.parameters(): param.requires_grad = False
            return
        
        for param in self.plm.parameters(): param.requires_grad = False
        
        if self.config.UNFREEZE_LAST_N > 0:
            for i in range(self.config.UNFREEZE_LAST_N):
                layer_idx = self.num_plm_layers - 1 - i
                if layer_idx >= 0:
                    for param in self.plm.layers[layer_idx].parameters():
                        param.requires_grad = True

    def _attention_pooling(self, reps, mask):
        """对序列token表征进行注意力池化"""
        attention_scores = self.attention_head(reps)
        seq_len = reps.size(1)
        attn_mask = mask[:, :seq_len].unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(attn_mask == 0, -10000.0)
        attention_weights = F.softmax(attention_scores, dim=1)
        return (reps * attention_weights).sum(dim=1)

    def forward(self, sequence_strs: List[str], attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        device = next(self.plm.parameters()).device
        
        # 1. --- 分词 ---
        data = [(f"seq_{i}", seq[:self.config.MAX_SEQ_LEN - 2]) for i, seq in enumerate(sequence_strs)]
        _, _, batch_tokens = self.tokenizer(data)
        batch_tokens = batch_tokens.to(device)

        # 2. --- 流1: ESM-2深度特征处理 ---
        with torch.set_grad_enabled(self.training and getattr(self.config, 'finetune_plm', False)):
            results = self.plm(batch_tokens, repr_layers=self.scale_layers, return_contacts=False)
        all_layer_embeddings = torch.stack([results["representations"][layer] for layer in self.scale_layers])
        normalized_weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        esm_embeddings = (all_layer_embeddings * normalized_weights).sum(dim=0)
        
        esm_cls_embedding = esm_embeddings[:, 0, :]
        esm_sequence_reps = esm_embeddings[:, 1:-1, :]
        
        # 3. --- 流2: 原始序列浅层特征处理 ---
        raw_embedded = self.raw_aa_embedding(batch_tokens)
        raw_sequence_embedded = raw_embedded[:, 1:-1, :]
        raw_sequence_features, _ = self.raw_seq_encoder(raw_sequence_embedded)
        
        # 4. --- 融合: Transformer Decoder ---
        actual_seq_len = raw_sequence_features.size(1)
        padding_mask = (attention_mask[:, :actual_seq_len] == 0)

        # 将原始序列特征(memory)投影到与ESM特征(tgt)相同的维度
        memory = self.memory_projection(raw_sequence_features)

        fused_sequence_reps = self.fusion_decoder(
            tgt=esm_sequence_reps,
            memory=memory,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask
        )

        # 5. --- 任务信息流与输出 ---
        # a. 池化, 用于分类/对比
        attention_pooled_embedding = self._attention_pooling(fused_sequence_reps, attention_mask)
        final_feature_for_cls_contrast = torch.cat([esm_cls_embedding, attention_pooled_embedding], dim=1)

        # b. 分类和对比学习 (使用最终融合特征)
        class_logits = self.classifier_head(final_feature_for_cls_contrast).squeeze(-1)
        contrast_features = self.contrast_project_head(final_feature_for_cls_contrast)
        
        # c. SS重建 (使用ESM-2的原始序列特征，任务解耦)
        # 使用.detach()阻断梯度，让SS重建只作为正则，不干扰主干网络的梯度
        ss_recon_hidden = self.ss_reconstructor_mlp(esm_sequence_reps.detach()) 
        ss_recon_logits = self.ss_output_layer(ss_recon_hidden)

        return {
            'class_logits': class_logits,
            'cls_embedding': contrast_features,
            'ss_recon_logits': ss_recon_logits,
        }
        
    def get_optimizer(self) -> torch.optim.Optimizer:
        """配置优化器，为PLM和所有头部分配不同的学习率"""
        plm_params = [p for p in self.plm.parameters() if p.requires_grad]
        
        head_params = (
            [self.layer_weights] +
            list(self.raw_aa_embedding.parameters()) +
            list(self.raw_seq_encoder.parameters()) +
            list(self.memory_projection.parameters()) +
            list(self.fusion_decoder.parameters()) +
            list(self.attention_head.parameters()) +
            list(self.classifier_head.parameters()) +
            list(self.contrast_project_head.parameters()) +
            list(self.ss_reconstructor_mlp.parameters()) +
            list(self.ss_output_layer.parameters())
        )
        
        param_groups = [
            {"params": plm_params, "lr": self.config.PLM_LR},
            {"params": head_params, "lr": self.config.head_lr}
        ]
        
        return torch.optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)
    

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Dict
# import esm
# import math # 新增

# # 确保已安装 torch_geometric
# from torch_geometric.nn import GATv2Conv, global_mean_pool
# from torch_geometric.utils import dense_to_sparse

# # ========================================
# # 0. 位置编码模块 (新增)
# # ========================================
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 5000):
#         super().__init__()
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(1, max_len, d_model)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Tensor, shape [batch_size, seq_len, embedding_dim]
#         """
#         return x + self.pe[:, :x.size(1)]

# # ========================================
# # 1. GNN 结构流模块 (升级版)
# # ========================================
# class GNN_Structural_Stream(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, n_heads=4, dropout=0.2):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         # Input Layer
#         self.layers.append(GATv2Conv(input_dim, hidden_dim, heads=n_heads, dropout=dropout, concat=True))
#         # Hidden Layers
#         for _ in range(n_layers - 2): # pythonic a bit
#             self.layers.append(GATv2Conv(hidden_dim * n_heads, hidden_dim, heads=n_heads, dropout=dropout, concat=True))
#         # Output Layer for Node Features
#         self.node_output_layer = GATv2Conv(hidden_dim * n_heads, input_dim, heads=1, dropout=dropout, concat=False) # 维度改回input_dim
#         # Output layer for Graph Embedding
#         self.graph_output_layer = GATv2Conv(hidden_dim * n_heads, output_dim, heads=1, dropout=dropout, concat=False)

#         self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim * n_heads) for _ in range(n_layers - 1)])
#         self.dropout = nn.Dropout(dropout)
#         self.act = nn.GELU()

#     # 在 GNN_Structural_Stream 的 forward 方法中
#     def forward(self, node_features, edge_index, batch_index):
#         x = node_features
#         # --- 修改开始 ---
#         for i, layer in enumerate(self.layers):
#             x = layer(x, edge_index)
#             # 只对隐藏层应用 norm, act, dropout
#             if i < len(self.layers) -1: # or i < len(self.norms)
#                 x = self.norms[i](x)
#                 x = self.act(x)
#                 x = self.dropout(x)
#         # --- 修改结束 ---
        
#         # 为节点和图分别计算输出 (这里的输入 x 已经是倒数第二层的输出了)
#         updated_node_features = self.node_output_layer(x, edge_index)
#         graph_features_for_pooling = self.graph_output_layer(x, edge_index)

#         # 全局池化得到图嵌入
#         graph_embedding = global_mean_pool(graph_features_for_pooling, batch_index)
        
#         # 返回两个结果
#         return graph_embedding, updated_node_features

# # ========================================
# # 2. 完整的三流模型（修复版）
# # ========================================
# class PeptideTriStreamModel_previous(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        
#         # --- 流1: ESM-2 深度特征流 ---
#         self.plm, self.alphabet = esm.pretrained.load_model_and_alphabet(config.PLM_NAME)
#         self.tokenizer = self.alphabet.get_batch_converter()
#         self.embed_dim = self.plm.embed_dim
#         self.num_plm_layers = self.plm.num_layers
#         self.scale_layers = [6, 12, 18, 24, self.num_plm_layers]
#         self.layer_weights = nn.Parameter(torch.ones(len(self.scale_layers)))
        
#         # --- 流2: 原始序列浅层特征流 ---
#         raw_embed_dim = config.RAW_EMBED_DIM
#         self.raw_aa_embedding = nn.Embedding(len(self.alphabet.all_toks), raw_embed_dim, 
#                                             padding_idx=self.alphabet.padding_idx)
#         self.raw_seq_encoder = nn.LSTM(
#             raw_embed_dim, raw_embed_dim, config.RAW_LSTM_LAYERS,
#             bidirectional=True, batch_first=True,
#             dropout=config.DROPOUT_RATE if config.RAW_LSTM_LAYERS > 1 else 0
#         )
#         raw_feature_dim = raw_embed_dim * 2
        
#         # --- 融合模块: Cross-Attention ---
#         self.cross_attention = nn.MultiheadAttention(
#             self.embed_dim, config.CROSS_ATTN_NHEAD,
#             kdim=raw_feature_dim, vdim=raw_feature_dim,
#             batch_first=True, dropout=config.DROPOUT_RATE
#         )
#         self.cross_attn_norm = nn.LayerNorm(self.embed_dim)

#         # ### 思路一: GNN信息回流 ###
#         # 用于融合GNN回流的节点特征，这里使用简单的残差连接，所以不需要额外层
#         self.structure_fusion_norm = nn.LayerNorm(self.embed_dim)

#         # ### 思路二: 位置编码 和 深度融合Encoder ###
#         self.pos_encoder = PositionalEncoding(self.embed_dim, config.MAX_SEQ_LEN + 1)
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embed_dim,
#             nhead=config.TRANSFORMER_HEAD_NHEAD,
#             dim_feedforward=config.TRANSFORMER_HEAD_DIM_FF,
#             dropout=config.DROPOUT_RATE,
#             batch_first=True)
#         self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.DECODER_LAYERS)

#         # --- 流3: GNN 结构流 (使用升级版) ---
#         self.gnn_stream = GNN_Structural_Stream(
#             input_dim=self.embed_dim,
#             hidden_dim=config.GNN_HIDDEN_DIM,
#             output_dim=config.GNN_OUTPUT_DIM, # For graph pooling
#             n_layers=config.GNN_LAYERS,
#             n_heads=config.GNN_NHEAD,
#             dropout=config.DROPOUT_RATE)
        
#         self.final_fusion_gate = nn.Sequential(
#             nn.Linear(self.embed_dim + config.GNN_OUTPUT_DIM, 1),
#             nn.Sigmoid()
#         )
#         # 保持下游MLP输入维度不变，但融合方式变了
#         final_fused_dim = self.embed_dim + config.GNN_OUTPUT_DIM
        
#         # --- 下游任务头 ---
#         self.classifier_head = nn.Sequential(
#             nn.Linear(final_fused_dim, self.embed_dim),
#             nn.GELU(),
#             nn.Dropout(config.DROPOUT_RATE),
#             nn.Linear(self.embed_dim, self.embed_dim // 2),
#             nn.GELU(),
#             nn.Dropout(config.DROPOUT_RATE),
#             nn.Linear(self.embed_dim // 2, 1))
            
#         self.contrast_project_head = nn.Sequential(
#             nn.Linear(final_fused_dim, self.embed_dim // 2),
#             nn.GELU(),
#             nn.Linear(self.embed_dim // 2, config.CONTRAST_FEATURE_DIM))
            
#         recon_hidden_dim = self.embed_dim // 2
        
#         # ### 思路四: 改变SS重建输入 ###
#         # MLP输入维度不变，但输入源会改变
#         self.ss_reconstructor_mlp = nn.Sequential(
#             nn.Linear(self.embed_dim, recon_hidden_dim),
#             nn.GELU(),
#             nn.Dropout(config.DROPOUT_RATE))
#         self.ss_output_layer = nn.Linear(recon_hidden_dim, config.NUM_SS)
        
#         self._init_head_weights()
#         self._freeze_plm_layers()

#     def _init_head_weights(self):
#         heads = [
#             self.raw_aa_embedding, self.raw_seq_encoder, self.cross_attention,
#             self.cross_attn_norm, self.gnn_stream, self.classifier_head,
#             self.contrast_project_head, self.ss_reconstructor_mlp,
#             self.ss_output_layer, self.fusion_encoder, self.structure_fusion_norm,
#             self.final_fusion_gate
#         ]
#         for head in heads:
#             for module in head.modules():
#                 if isinstance(module, (nn.Linear, nn.Embedding)):
#                     module.weight.data.normal_(mean=0.0, std=0.02)
#                     if isinstance(module, nn.Linear) and module.bias is not None:
#                         module.bias.data.zero_()
#         nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

#     def _freeze_plm_layers(self):
#         if not self.config.finetune_plm:
#             for param in self.plm.parameters():
#                 param.requires_grad = False
#             return
        
#         for param in self.plm.parameters():
#             param.requires_grad = False
        
#         if self.config.UNFREEZE_LAST_N > 0:
#             for i in range(self.config.UNFREEZE_LAST_N):
#                 layer_idx = self.num_plm_layers - 1 - i
#                 if layer_idx >= 0:
#                     for param in self.plm.layers[layer_idx].parameters():
#                         param.requires_grad = True

#     # def _attention_pooling(self, reps, mask):
#     #     """注意力池化，兼容FP16"""
#     #     attention_scores = self.attention_head(reps)
#     #     seq_len = reps.size(1)
#     #     attn_mask = mask[:, :seq_len].unsqueeze(-1)
#     #     # 使用 -1e4 而不是 -1e9，避免 FP16 溢出
#     #     attention_scores = attention_scores.masked_fill(attn_mask == 0, -1e4)
#     #     attention_weights = F.softmax(attention_scores, dim=1)
#     #     return (reps * attention_weights).sum(dim=1)

  
#     def forward(self, sequence_strs: List[str], attention_mask: torch.Tensor, 
#                 contact_maps: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
#         device = next(self.plm.parameters()).device
#         batch_size = len(sequence_strs)
#         data = [(f"seq_{i}", seq) for i, seq in enumerate(sequence_strs)]
#         _, _, batch_tokens = self.tokenizer(data)
#         batch_tokens = batch_tokens.to(device)
#         with torch.set_grad_enabled(self.training and self.config.finetune_plm):
#             results = self.plm(batch_tokens, repr_layers=self.scale_layers, return_contacts=False)
#             all_layer_embeddings = torch.stack([results["representations"][layer] for layer in self.scale_layers])
#             normalized_weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
#             esm_embeddings = (all_layer_embeddings * normalized_weights).sum(dim=0)
#         esm_sequence_reps = esm_embeddings[:, 1:-1, :]
#         raw_tokens = batch_tokens[:, 1:-1]
#         raw_embedded = self.raw_aa_embedding(raw_tokens)
#         raw_sequence_features, _ = self.raw_seq_encoder(raw_embedded)
#         actual_seq_len = min(esm_sequence_reps.size(1), raw_sequence_features.size(1), attention_mask.size(1), contact_maps.size(1))
#         # ... 对齐所有 ...
#         esm_sequence_reps = esm_sequence_reps[:, :actual_seq_len, :]
#         raw_sequence_features = raw_sequence_features[:, :actual_seq_len, :]
#         attention_mask_aligned = attention_mask[:, :actual_seq_len]
#         contact_maps_aligned = contact_maps[:, :actual_seq_len, :actual_seq_len]

#         # 5. 初始融合: Cross-Attention
#         key_padding_mask = (attention_mask_aligned == 0)
#         cross_attended_reps, _ = self.cross_attention(query=esm_sequence_reps, key=raw_sequence_features, value=raw_sequence_features, key_padding_mask=key_padding_mask)
#         fused_sequence_reps = self.cross_attn_norm(cross_attended_reps + esm_sequence_reps)

#         # 6. 流3: GNN (提前计算以获取结构信息)
#         node_features = fused_sequence_reps.reshape(-1, self.embed_dim)
#         batch_index = torch.arange(batch_size, device=device).repeat_interleave(actual_seq_len)
#         # 构建edge_index
#         edge_indices = []
#         for i in range(batch_size):
#             # 对接触图进行阈值化处理，只保留真正的连接
#             contact_matrix = contact_maps_aligned[i]
#             edge_index, _ = dense_to_sparse(contact_matrix)
            
#             # 如果没有边，跳过
#             if edge_index.size(1) == 0:
#                 continue
            
#             # 偏移索引到全局batch中的位置
#             edge_index = edge_index + i * actual_seq_len
#             edge_indices.append(edge_index)
        
#         if edge_indices:
#             edge_index = torch.cat(edge_indices, dim=1).to(device)
#         else:
#             # 如果没有边，创建一个空的edge_index
#             edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
#         global_graph_embedding, updated_node_features = self.gnn_stream(node_features, edge_index, batch_index)
#         updated_node_features = updated_node_features.reshape(batch_size, actual_seq_len, -1)
#         reps_with_structure = self.structure_fusion_norm(fused_sequence_reps + updated_node_features)

#         # 7. 深度融合: Transformer Encoder
#         # ### 思路二: 位置编码 ###
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         encoder_input = torch.cat([cls_tokens, reps_with_structure], dim=1) # 输入是包含结构信息的reps
#         encoder_input = self.pos_encoder(encoder_input) # 添加位置编码

#         cls_mask = torch.ones(batch_size, 1, dtype=attention_mask_aligned.dtype, device=device)
#         encoder_mask = torch.cat([cls_mask, attention_mask_aligned], dim=1)
#         encoder_padding_mask = (encoder_mask == 0)
        
#         encoder_output = self.fusion_encoder(encoder_input, src_key_padding_mask=encoder_padding_mask)
#         sequence_summary_embedding = encoder_output[:, 0, :]
#         deeply_fused_sequence_reps = encoder_output[:, 1:, :] # encoder输出的序列部分

#         # 8. 最终特征融合与下游任务
#         # ### 思路三: 改进最终拼接 (门控融合) ###
#         combined_final_embedding = torch.cat([sequence_summary_embedding, global_graph_embedding], dim=1)
#         gate = self.final_fusion_gate(combined_final_embedding)
#         final_feature = combined_final_embedding * gate # 门控加权
        
#         class_logits = self.classifier_head(final_feature).squeeze(-1)
#         contrast_features = self.contrast_project_head(final_feature)
        
#         # ### 思路四: 改变SS重建输入 ###
#         ss_recon_hidden = self.ss_reconstructor_mlp(deeply_fused_sequence_reps)
#         ss_recon_logits = self.ss_output_layer(ss_recon_hidden)

#         return {
#             'class_logits': class_logits,
#             'cls_embedding': contrast_features,
#             'ss_recon_logits': ss_recon_logits,
#         }

#     def get_optimizer(self) -> torch.optim.Optimizer:
#         plm_params = [p for p in self.plm.parameters() if p.requires_grad]
#         head_params = (
#             [self.layer_weights, self.cls_token] +
#             list(self.raw_aa_embedding.parameters()) +
#             list(self.raw_seq_encoder.parameters()) +
#             list(self.cross_attention.parameters()) +
#             list(self.cross_attn_norm.parameters()) +
#             list(self.fusion_encoder.parameters()) +
#             list(self.gnn_stream.parameters()) +   # list(self.attention_head.parameters()) +
#             list(self.structure_fusion_norm.parameters()) +  
#             list(self.final_fusion_gate.parameters()) +
#             list(self.classifier_head.parameters()) +
#             list(self.contrast_project_head.parameters()) +
#             list(self.ss_reconstructor_mlp.parameters()) +
#             list(self.ss_output_layer.parameters())
#         )
#         param_groups = [
#             {"params": plm_params, "lr": self.config.PLM_LR},
#             {"params": head_params, "lr": self.config.head_lr}
#         ]
#         return torch.optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import esm
import math

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse

# ========================================
# 0. 位置编码模块
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
# 1. GVP 核心组件 (几何向量感知器)
# ========================================
class GVP(nn.Module):
    """
    Geometric Vector Perceptron
    处理标量特征 + 向量特征
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
        
        # 标量 → 标量
        self.Wss = nn.Linear(in_scalar_dim, out_scalar_dim, bias=False)
        # 向量 → 标量 (通过向量范数)
        self.Wvs = nn.Linear(in_vector_dim, out_scalar_dim, bias=False)
        # 标量 → 向量
        if out_vector_dim > 0:
            self.Wsv = nn.Linear(in_scalar_dim, out_vector_dim, bias=False)
            # 向量 → 向量
            self.Wvv = nn.Linear(in_vector_dim, out_vector_dim, bias=False)
        
        self.norm = nn.LayerNorm(out_scalar_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, s: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: 标量特征 [N, in_scalar_dim]
            v: 向量特征 [N, in_vector_dim, 3]
        Returns:
            s_out: [N, out_scalar_dim]
            v_out: [N, out_vector_dim, 3]
        """
        # 向量范数 (用于 v→s)
        v_norm = torch.norm(v, dim=-1)  # [N, in_vector_dim]
        
        # 标量输出
        s_out = self.Wss(s) + self.Wvs(v_norm)
        s_out = self.norm(s_out)
        s_out = F.gelu(s_out)
        s_out = self.dropout(s_out)
        
        # 向量输出 (如果需要)
        if self.out_vector_dim > 0:
            # v → v: 保持方向，调整幅度
            v_out = self.Wvv(v.transpose(1, 2)).transpose(1, 2)  # [N, out_vector_dim, 3]
            
            # s → v: 通过门控机制调整
            gate = torch.sigmoid(self.Wsv(s)).unsqueeze(-1)  # [N, out_vector_dim, 1]
            v_out = v_out * gate
        else:
            v_out = None
        
        return s_out, v_out


# ========================================
# 2. GVP-GNN 消息传递层
# ========================================
class GVPConvLayer(nn.Module):
    """
    GVP 图卷积层
    """
    def __init__(self,
                 node_scalar_dim: int,
                 node_vector_dim: int,
                 edge_scalar_dim: int,
                 edge_vector_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # 消息函数：将节点特征 + 边特征 → 消息
        self.message_gvp = GVP(
            in_scalar_dim=node_scalar_dim * 2 + edge_scalar_dim,
            in_vector_dim=node_vector_dim * 2 + edge_vector_dim,
            out_scalar_dim=hidden_dim,
            out_vector_dim=node_vector_dim,
            dropout=dropout
        )
        
        # 更新函数：聚合消息 → 新节点特征
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
        
        # 构建消息
        msg_s = torch.cat([node_s[src], node_s[dst], edge_s], dim=-1)  # [E, ...]
        msg_v = torch.cat([node_v[src], node_v[dst], edge_v], dim=-2)  # [E, ..., 3]
        
        # 通过 GVP 得到消息
        msg_s, msg_v = self.message_gvp(msg_s, msg_v)
        
        # ========== 修复开始 ⭐ ==========
        # 聚合消息 (sum aggregation)
        num_nodes = node_s.size(0)
        
        # 使用 msg_s/msg_v 的 dtype 和 device 来创建聚合张量
        aggr_s = torch.zeros(num_nodes, msg_s.size(1), 
                            dtype=msg_s.dtype, device=msg_s.device)  # ⭐ 新增 dtype
        aggr_v = torch.zeros(num_nodes, msg_v.size(1), 3, 
                            dtype=msg_v.dtype, device=msg_v.device)  # ⭐ 新增 dtype
        
        aggr_s.index_add_(0, dst, msg_s)
        aggr_v.index_add_(0, dst, msg_v)
        # ========== 修复结束 ==========
        
        # 更新节点
        update_s = torch.cat([node_s, aggr_s], dim=-1)
        update_v = torch.cat([node_v, aggr_v], dim=-2)
        
        new_node_s, new_node_v = self.update_gvp(update_s, update_v)
        
        # 残差连接
        new_node_s = new_node_s + node_s
        new_node_v = new_node_v + node_v
        
        return new_node_s, new_node_v


# ========================================
# 3. GVP-GNN 结构流模块 (升级版)
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
        self.node_vector_dim = 4  # [CA方向, CB方向, 主链法向量, 局部切向量]
        
        # 输入投影
        # 标量：ESM + 几何标量特征
        total_scalar_dim = esm_dim + (geometric_dim - 6)  # 去掉CA(3)+CB(3)坐标
        self.node_scalar_project = nn.Linear(total_scalar_dim, hidden_dim)
        
        # 边特征投影
        # 边标量特征：27 - 3(方向向量，改为向量特征)
        edge_scalar_input = edge_scalar_dim - 3
        self.edge_scalar_project = nn.Linear(edge_scalar_input, hidden_dim)
        
        # GVP 卷积层
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
        
        # 输出层
        self.node_output = nn.Linear(hidden_dim, esm_dim)
        self.graph_output = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def _compute_node_vectors(self, coords: torch.Tensor, cb_coords: torch.Tensor) -> torch.Tensor:
        """
        计算节点的向量特征
        Args:
            coords: CA坐标 [N, 3]
            cb_coords: CB坐标 [N, 3]
        Returns:
            node_vectors: [N, 4, 3]
        """
        N = coords.size(0)
        vectors = []
        
        # 1. CA → CB 方向
        ca_cb = cb_coords - coords  # [N, 3]
        ca_cb_norm = F.normalize(ca_cb, dim=-1)
        vectors.append(ca_cb_norm)
        
        # 2. CA(i) → CA(i+1) 方向
        ca_next = torch.roll(coords, shifts=-1, dims=0)
        ca_direction = ca_next - coords
        ca_direction[-1] = 0  # 最后一个残基
        ca_direction_norm = F.normalize(ca_direction, dim=-1)
        vectors.append(ca_direction_norm)
        
        # 3. 主链法向量 (叉积)
        normal = torch.cross(ca_cb_norm, ca_direction_norm, dim=-1)
        normal_norm = F.normalize(normal, dim=-1)
        vectors.append(normal_norm)
        
        # 4. 局部切向量 (CA(i-1) → CA(i+1))
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
                node_coords: torch.Tensor,         # [B*L, 3] CA坐标
                edge_index: torch.Tensor,          # [2, E]
                edge_attr: torch.Tensor,           # [E, 27]
                batch_index: torch.Tensor):        # [B*L]
        """
        Args:
            geometric_features: [CA(3), CB(3), phi, psi, omega, curvature, rsa, properties(4)]
        """
        # 提取 CB 坐标
        cb_coords = geometric_features[:, 3:6]  # [B*L, 3]
        
        # 标量特征：ESM + 几何特征(去掉坐标)
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
        
        # 向量特征
        node_v = self._compute_node_vectors(node_coords, cb_coords)  # [B*L, 4, 3]
        
        # 边特征
        # 标量：去掉方向向量(index 2:5)
        edge_s = torch.cat([
            edge_attr[:, 0:2],    # ca_dist, cb_dist
            edge_attr[:, 5:],     # seq_dist, rbf(16), ss_same, delta_angles(3), hbond
        ], dim=-1)  # [E, 24]
        edge_s = self.edge_scalar_project(edge_s)
        
        # 向量：边方向向量
        edge_v = edge_attr[:, 2:5].unsqueeze(1)  # [E, 1, 3]
        
        # GVP 消息传递
        for layer in self.gvp_layers:
            node_s, node_v = layer(node_s, node_v, edge_index, edge_s, edge_v)
        
        # 输出
        updated_node_features = self.node_output(node_s)
        graph_features_for_pooling = self.graph_output(node_s)
        
        graph_embedding = global_mean_pool(graph_features_for_pooling, batch_index)
        
        return graph_embedding, updated_node_features


# ========================================
# 4. 完整的三流模型（升级版）
# ========================================
class PeptideTriStreamModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- 流1: ESM-2 深度特征流 ---
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
        
        # --- 流2: 原始序列浅层特征流 ---
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
        
        # --- 融合模块: Cross-Attention ---
        self.cross_attention = nn.MultiheadAttention(
            self.embed_dim, config.CROSS_ATTN_NHEAD,
            kdim=raw_feature_dim, vdim=raw_feature_dim,
            batch_first=True, dropout=config.DROPOUT_RATE
        )
        self.cross_attn_norm = nn.LayerNorm(self.embed_dim)
        
        # GNN信息回流
        self.structure_fusion_norm = nn.LayerNorm(self.embed_dim)
        
        # 位置编码 和 深度融合Encoder
        self.pos_encoder = PositionalEncoding(self.embed_dim, config.MAX_SEQ_LEN + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.TRANSFORMER_HEAD_NHEAD,
            dim_feedforward=config.TRANSFORMER_HEAD_DIM_FF,
            dropout=config.DROPOUT_RATE,
            batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.DECODER_LAYERS)
        
        # --- 流3: GVP-GNN 结构流 (升级!) ⭐ ---
        self.gnn_stream = GVP_GNN_Structural_Stream(
            esm_dim=self.embed_dim,
            geometric_dim=15,  # 升级后的几何特征维度
            edge_scalar_dim=27,  # 升级后的边特征维度
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
        
        # --- 下游任务头 ---
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
                contact_maps: torch.Tensor = None,      # 改为可选参数
                node_geometric_feat: torch.Tensor = None,   # 改为可选参数
                edge_index: torch.Tensor = None,             # 改为可选参数
                edge_attr: torch.Tensor = None,              # 改为可选参数
                node_coords: torch.Tensor = None,            # 改为可选参数
                return_intermediate: bool = False,
                **kwargs) -> Dict[str, torch.Tensor]:
        
        # Get device from any available parameter
        if self.config.USE_ESM2:
            device = next(self.plm.parameters()).device
        else:
            device = next(self.simple_plm_embedding.parameters()).device
        batch_size = len(sequence_strs)

        # ========== 1. ESM-2 编码 ==========
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
                esm_sequence_reps = esm_embeddings[:, 1:-1, :]  # 去掉 <cls> 和 <eos>
        else:
            # 🔥 ESM-2 ablation: Use simple embedding
            esm_sequence_reps = self.simple_plm_embedding(batch_tokens[:, 1:-1])

        # ========== 2. 原始序列特征 ==========
        raw_tokens = batch_tokens[:, 1:-1]
        raw_embedded = self.raw_aa_embedding(raw_tokens)
        
        if self.config.USE_LSTM and self.raw_seq_encoder is not None:
            # 🔥 Use LSTM encoder
            raw_sequence_features, _ = self.raw_seq_encoder(raw_embedded)
        else:
            # 🔥 LSTM ablation: Use raw embeddings directly
            raw_sequence_features = raw_embedded

        # ========== 3. 对齐长度 ==========
        actual_seq_len = min(
            esm_sequence_reps.size(1),
            raw_sequence_features.size(1),
            attention_mask.size(1)
        )
        
        # 如果有几何特征，也要对齐
        if node_geometric_feat is not None and node_coords is not None:
            actual_seq_len = min(actual_seq_len, node_geometric_feat.size(1), node_coords.size(1))
        
        esm_sequence_reps = esm_sequence_reps[:, :actual_seq_len, :]
        raw_sequence_features = raw_sequence_features[:, :actual_seq_len, :]
        attention_mask_aligned = attention_mask[:, :actual_seq_len]

        # ========== 4. Cross-Attention 融合 ==========
        key_padding_mask = (attention_mask_aligned == 0)
        cross_attended_reps, _ = self.cross_attention(
            query=esm_sequence_reps,
            key=raw_sequence_features,
            value=raw_sequence_features,
            key_padding_mask=key_padding_mask
        )
        fused_sequence_reps = self.cross_attn_norm(cross_attended_reps + esm_sequence_reps)

        # ========== 5. GVP-GNN 结构流 (可选) ⭐ ==========
        global_graph_embedding = None
        updated_node_features = None
        
        # 🔥 关键修改：只有当启用GVP且输入不为None时才运行GNN
        if (self.config.USE_GVP and 
            node_geometric_feat is not None and 
            edge_index is not None and 
            edge_attr is not None and 
            node_coords is not None):
            
            node_geometric_feat_aligned = node_geometric_feat[:, :actual_seq_len, :]
            node_coords_aligned = node_coords[:, :actual_seq_len, :]
            
            # 展平为节点级
            esm_node_features = fused_sequence_reps.reshape(-1, self.embed_dim)
            geometric_node_features = node_geometric_feat_aligned.reshape(-1, 15)
            node_coords_flat = node_coords_aligned.reshape(-1, 3)
            batch_index = torch.arange(batch_size, device=device).repeat_interleave(actual_seq_len)
            
            # 调用 GVP-GNN
            global_graph_embedding, updated_node_features = self.gnn_stream(
                esm_features=esm_node_features,
                geometric_features=geometric_node_features,
                node_coords=node_coords_flat,
                edge_index=edge_index.to(device),
                edge_attr=edge_attr.to(device),
                batch_index=batch_index
            )
            
            # 恢复形状并融合
            updated_node_features = updated_node_features.reshape(batch_size, actual_seq_len, -1)
            reps_with_structure = self.structure_fusion_norm(fused_sequence_reps + updated_node_features)
        else:
            # 🔥 没有GVP时，直接使用融合后的序列特征
            reps_with_structure = fused_sequence_reps
            # 创建一个零向量作为全局图嵌入的占位符
            global_graph_embedding = torch.zeros(batch_size, self.config.GNN_OUTPUT_DIM, device=device)

        # ========== 6. Transformer 深度融合 ==========
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat([cls_tokens, reps_with_structure], dim=1)
        encoder_input = self.pos_encoder(encoder_input)
        cls_mask = torch.ones(batch_size, 1, dtype=attention_mask_aligned.dtype, device=device)
        encoder_mask = torch.cat([cls_mask, attention_mask_aligned], dim=1)
        encoder_padding_mask = (encoder_mask == 0)
        encoder_output = self.fusion_encoder(encoder_input, src_key_padding_mask=encoder_padding_mask)
        sequence_summary_embedding = encoder_output[:, 0, :]
        deeply_fused_sequence_reps = encoder_output[:, 1:, :]

        # ========== 7. 最终融合与下游任务 ==========
        combined_final_embedding = torch.cat([sequence_summary_embedding, global_graph_embedding], dim=1)
        gate = self.final_fusion_gate(combined_final_embedding)
        final_feature = combined_final_embedding * gate

        # 分类
        class_logits = self.classifier_head(final_feature).squeeze(-1)
        
        # 对比学习
        contrast_features = self.contrast_project_head(final_feature)

        # 🔥 二级结构重建（可选）
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

        # ⭐ 如果需要中间特征
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
            list(self.ss_crf.parameters())  # ← CRF 参数
        )
        
        param_groups = [
            {"params": plm_params, "lr": self.config.PLM_LR},
            {"params": head_params, "lr": self.config.head_lr}
        ]
        return torch.optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)