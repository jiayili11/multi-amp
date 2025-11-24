"""
MultiAMP Configuration
"""
import torch

class MultiAMPConfig:
    def __init__(self):
        # Device
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model architecture switches
        self.USE_ESM2 = True
        self.USE_GVP = True
        self.USE_SS_RECON = True
        self.USE_LSTM = True
        self.USE_SHALLOW_FEATURES = True
        
        # Paths
        self.TRAIN_DATA_PATH = "./data/train_amp/"
        self.VALID_DATA_PATH = "./data/test_amp/"
        self.SAVE_DIR = "./checkpoints/"
        self.AMP_PDB_DIR = './data/structure/amp_train5985/'
        self.NONAMP_PDB_DIR = './data/structure/nonamp_train5985/'
        self.AMP_VALID_PDB_DIR = './data/structure/amp_testset/'
        self.NONAMP_VALID_PDB_DIR = './data/structure/nonamp_testset/'
        
        # ESM-2 model
        self.PLM_NAME = "esm2_t33_650M_UR50D"
        self.EMBED_DIM = 1280
        self.UNFREEZE_LAST_N = 6
        
        # Training
        self.MAX_SEQ_LEN = 1024
        self.BATCH_SIZE = 16
        self.EPOCHS = 30
        self.PLM_LR = 1e-5
        self.head_lr = 5e-5
        self.WEIGHT_DECAY = 0.01
        self.GRAD_CLIP = 1.0
        self.USE_AMP = True
        self.WARMUP_EPOCHS = 5
        
        # Model architecture
        self.DROPOUT_RATE = 0.3
        self.TEMPERATURE = 0.07
        self.CONTRAST_FEATURE_DIM = 256
        
        # LSTM
        self.RAW_EMBED_DIM = 128
        self.RAW_LSTM_LAYERS = 2
        
        # Transformer
        self.CROSS_ATTN_NHEAD = 8
        self.TRANSFORMER_HEAD_NHEAD = 8
        self.TRANSFORMER_HEAD_DIM_FF = 1024
        self.DECODER_LAYERS = 3
        
        # GVP-GNN
        self.GNN_HIDDEN_DIM = 384
        self.GNN_OUTPUT_DIM = 512
        self.GNN_LAYERS = 3
        self.GNN_NHEAD = 8
        self.KNN_K = 15
        self.CONTACT_CUTOFF = 4.5
        self.USE_SS_MOTIF_GRAPH = True
        self.GEOMETRIC_FEAT_DIM = 15
        self.EDGE_FEAT_DIM = 27
        self.NODE_VECTOR_DIM = 4
        self.EDGE_VECTOR_DIM = 1
        self.HBOND_DISTANCE_CUTOFF = 3.5
        self.HBOND_ANGLE_CUTOFF = 120
        self.CONTACT_MAP_THRESHOLD = 10.0
        
        # Reconstruction
        self.RECON_MASK_PROB = 0.15
        self.RECON_USE_NOISE = True
        self.NUM_AA = 22
        self.NUM_SS = 3
        self.SS_FOCAL_GAMMA = 3.0
        self.SS_CONTINUITY_WEIGHT = 0.1
        self.SS_USE_CRF = True
        self.RECON_IGNORE_INDEX = -1
        
        # Loss weights
        self.W_CLASS = 1.0
        self.W_CONTRAST = 0.1
        self.W_SEQ_RECON = 0
        self.W_SS_RECON = 0.15
        self.W_KL = 0
        
        # Scheduler
        self.SCHEDULER_TYPE = 'plateau'
        self.PLATEAU_PATIENCE = 6
        self.PLATEAU_FACTOR = 0.5
        
        # Runtime
        self.finetune_plm = False
        self.current_epoch = 0
        self.SEED = 42
        
    def update_for_epoch(self, epoch: int):
        """Update config for current epoch"""
        self.current_epoch = epoch
        if epoch >= self.WARMUP_EPOCHS:
            if not self.finetune_plm:
                self.finetune_plm = True
        else:
            self.finetune_plm = False
