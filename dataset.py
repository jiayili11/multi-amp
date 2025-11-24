# # ==============================================================================
# # dataset.py (CORRECTED VERSION)
# # ==============================================================================
# import os
# import torch
# from Bio import SeqIO
# from torch.utils.data import Dataset

# # 定义映射 (保持不变)
# AA_MAP = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX-")}
# SS_MAP = {ss: i for i, ss in enumerate("HCE-")}

# class PeptideDataset(Dataset):
#     def __init__(self, data_path, max_len):
#         self.max_len = max_len
#         self.records = []
        
#         if not os.path.exists(data_path) or not os.path.isdir(data_path):
#             print(f"Error: Data path '{data_path}' does not exist or is not a directory.")
#             return

#         for filename in os.listdir(data_path):
#             if filename.endswith(".fas"):
#                 filepath = os.path.join(data_path, filename)
#                 try:
#                     for record in SeqIO.parse(filepath, "fasta"):
#                         self.records.append(record)
#                 except Exception as e:
#                     print(f"Warning: Could not parse file {filepath}. Error: {e}")
        
#         print(f"Loaded {len(self.records)} records from {data_path}")

#     def __len__(self):
#         return len(self.records)

#     def __getitem__(self, idx):
#         record = self.records[idx]
        
#         seq_id, label_str = record.id.split('|')
#         label = int(label_str)

#         # --- 2. 修正后的序列和二级结构解析逻辑 ---
#         full_string = str(record.seq)
        
#         # 因为氨基酸序列和二级结构序列长度相同，所以对半切分
#         split_point = len(full_string) // 2
#         seq = full_string[:split_point]
#         ss = full_string[split_point:]

#         # 添加一个健全性检查，确保切分正确
#         if len(seq) != len(ss):
#             print(f"Warning: Length mismatch after splitting for record {seq_id}. Seq len: {len(seq)}, SS len: {len(ss)}. Skipping record.")
#             # 返回一个空的或者说是可忽略的样本，或者在加载时就过滤掉
#             return self.__getitem__((idx + 1) % len(self)) # 简单地跳过，加载下一个

#         # --- 3. 编码 (保持不变) ---
#         seq_encoded = [AA_MAP.get(aa, AA_MAP['X']) for aa in seq]
#         ss_encoded = [SS_MAP.get(s, SS_MAP['-']) for s in ss]

#         # --- 4. 填充和截断 (现在可以正常工作了) ---
#         original_len = len(seq_encoded)
        
#         # 填充
#         padding_len = self.max_len - original_len
#         if padding_len > 0:
#             seq_encoded += [AA_MAP['-']] * padding_len
#             ss_encoded += [SS_MAP['-']] * padding_len
        
#         # 截断
#         seq_encoded = seq_encoded[:self.max_len]
#         ss_encoded = ss_encoded[:self.max_len]
        
#         # 创建attention_mask
#         attention_mask = [1] * min(original_len, self.max_len) + [0] * max(0, self.max_len - original_len)

#         return {
#             'id': seq_id,
#             'sequence_str': str(seq), # 只返回原始的氨基酸序列字符串
#             'ss_str': str(ss), # 只返回原始的氨基酸序列字符串
#             'label': torch.tensor(label, dtype=torch.float32),
#             'seq_target': torch.tensor(seq_encoded, dtype=torch.long),
#             'ss_target': torch.tensor(ss_encoded, dtype=torch.long),
#             'attention_mask': torch.tensor(attention_mask, dtype=torch.long) # Transformer mask通常是long或bool
#         }
    
# ==============================================================================
# dataset.py (版本：带随机Masking)
# ==============================================================================
# import os
# import torch
# from Bio import SeqIO
# from torch.utils.data import Dataset
# import random # 需要导入random

# # 定义映射 (保持不变)
# AA_MAP = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX-")}
# SS_MAP = {ss: i for i, ss in enumerate("HCE-")}
# # 创建一个不包含特殊字符的氨基酸列表，用于随机替换
# STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# class PeptideDataset(Dataset):
#     def __init__(self, data_path=None, records=None, max_len=1024, mask_prob=0.15, use_noise=True):
#         self.max_len = max_len
#         self.records = []
#         # *** 新增：从外部传入masking参数 ***
#         self.mask_prob = mask_prob
#         self.use_noise = use_noise

#         if records is not None:
#             # 如果直接传入了records列表，就用它
#             self.records = records
#         elif data_path is not None:
#             # 否则，从路径加载
#             self.records = []
#             if not os.path.exists(data_path) or not os.path.isdir(data_path):
#                 print(f"Error: Data path '{data_path}' does not exist.")
#                 return
#             for filename in os.listdir(data_path):
#                 if filename.endswith(".fas"):
#                     filepath = os.path.join(data_path, filename)
#                     self.records.extend(list(SeqIO.parse(filepath, "fasta")))
#             print(f"Loaded {len(self.records)} records from {data_path}")
#         else:
#             raise ValueError("Either 'data_path' or 'records' must be provided.")

#     def __len__(self):
#         return len(self.records)

#     def __getitem__(self, idx):
#         record = self.records[idx]
#         seq_id, label_str = record.id.split('|')
#         label = int(label_str)

#         # 1. 解析原始序列和二级结构
#         full_string = str(record.seq)
#         split_point = len(full_string) // 2
#         original_seq = full_string[:split_point]
#         ss = full_string[split_point:]

#         if len(original_seq) != len(ss):
#             print(f"Warning: Length mismatch for record {seq_id}. Skipping.")
#             return self.__getitem__((idx + 1) % len(self))
            
#         # 2. *** 核心修改：创建加噪序列和重建目标 ***
        
#         # 目标(target)使用原始、干净的序列
#         seq_target_encoded = [AA_MAP.get(aa, AA_MAP['X']) for aa in original_seq]
#         ss_target_encoded = [SS_MAP.get(s, SS_MAP['-']) for s in ss]

#         # 输入(input)使用加噪后的序列
#         input_seq = list(original_seq)
#         if self.use_noise and self.mask_prob > 0:
#             for i in range(len(input_seq)):
#                 if random.random() < self.mask_prob:
#                     # 遵循BERT策略: 80% mask, 10% random, 10% keep
#                     rand_val = random.random()
#                     if rand_val < 0.8:
#                         input_seq[i] = 'X' # 'X'作为我们的[MASK] token
#                     elif rand_val < 0.9:
#                         input_seq[i] = random.choice(STANDARD_AMINO_ACIDS)
#                     # else: 保持不变

#         input_seq_str = "".join(input_seq)

#         # 3. 填充和截断
#         # 注意：现在填充的是target，因为它们可能比max_len短
#         original_len = len(seq_target_encoded)
        
#         padding_len = self.max_len - original_len
#         if padding_len > 0:
#             # 使用-1 (或config.RECON_IGNORE_INDEX)进行填充
#             seq_target_encoded += [-1] * padding_len
#             ss_target_encoded += [-1] * padding_len
        
#         # 截断
#         seq_target_encoded = seq_target_encoded[:self.max_len]
#         ss_target_encoded = ss_target_encoded[:self.max_len]

#         attention_mask = [1] * min(original_len, self.max_len) + [0] * max(0, self.max_len - original_len)

#         return {
#             'id': seq_id,
#             'sequence_str': input_seq_str,     # <--- 模型输入的是加噪序列
#             'ss_str': str(ss),                 # (ss_str我们不加噪，因为模型不直接用它)
#             'label': torch.tensor(label, dtype=torch.long), # <--- 改回long，在train.py中按需转float
#             'seq_target': torch.tensor(seq_target_encoded, dtype=torch.long),
#             'ss_target': torch.tensor(ss_target_encoded, dtype=torch.long),
#             'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
#             'original_sequence_str': original_seq
#         }

# import os
# import torch
# from Bio import SeqIO
# from torch.utils.data import Dataset
# import random

# # 定义映射 (保持不变)
# AA_MAP = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX-")}
# SS_MAP = {ss: i for i, ss in enumerate("HCE-")}
# # 创建一个不包含特殊字符的氨基-酸列表，用于随机替换
# STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# class PeptideDataset(Dataset):
#     def __init__(self, data_path, max_len, mask_prob=0.15, use_noise=True):
#         self.max_len = max_len
#         self.mask_prob = mask_prob
#         self.use_noise = use_noise
        
#         initial_records = []
#         if not os.path.exists(data_path) or not os.path.isdir(data_path):
#             print(f"Error: Data path '{data_path}' does not exist or is not a directory.")
#             self.records = []
#             return
            
#         print(f"Loading raw records from {data_path}...")
#         for filename in os.listdir(data_path):
#             if filename.endswith(".fas"):
#                 filepath = os.path.join(data_path, filename)
#                 try:
#                     initial_records.extend(list(SeqIO.parse(filepath, "fasta")))
#                 except Exception as e:
#                     print(f"Warning: Could not parse file {filepath}. Error: {e}")
        
#         print(f"Found {len(initial_records)} raw records. Filtering invalid sequences...")
        
#         # --- 核心修正点：在初始化时就进行过滤 ---
#         self.records = []
#         num_skipped = 0
#         for record in initial_records:
#             full_string = str(record.seq)
#             # 检查长度是否为偶数
#             if len(full_string) % 2 != 0:
#                 num_skipped += 1
#                 continue
            
#             # 检查切分后长度是否相等 (更严格的检查)
#             split_point = len(full_string) // 2
#             if len(full_string[:split_point]) != len(full_string[split_point:]):
#                 num_skipped += 1
#                 continue
            
#             self.records.append(record)
        
#         if num_skipped > 0:
#             print(f"Filtered out {num_skipped} records with odd length or split mismatch.")
        
#         print(f"Successfully loaded {len(self.records)} valid records from {data_path}.")

#     def __len__(self):
#         return len(self.records)

#     def __getitem__(self, idx):
#         record = self.records[idx]
#         seq_id, label_str = record.id.split('|')
#         label = int(label_str)

#         full_string = str(record.seq)
#         split_point = len(full_string) // 2
#         original_seq = full_string[:split_point]
#         ss = full_string[split_point:]
        
#         # 目标(target)使用原始、干净的序列
#         ss_target_encoded = [SS_MAP.get(s, SS_MAP['-']) for s in ss]

#         # 输入(input)使用加噪后的序列
#         input_seq = list(original_seq)
#         if self.use_noise and self.mask_prob > 0:
#             for i in range(len(input_seq)):
#                 if random.random() < self.mask_prob:
#                     rand_val = random.random()
#                     if rand_val < 0.8:
#                         input_seq[i] = 'X'
#                     elif rand_val < 0.9:
#                         input_seq[i] = random.choice(STANDARD_AMINO_ACIDS)
#         input_seq_str = "".join(input_seq)

#         # 填充和截断
#         original_len = len(ss_target_encoded)
#         padding_len = self.max_len - original_len
#         if padding_len > 0:
#             ss_target_encoded += [-1] * padding_len
        
#         ss_target_encoded = ss_target_encoded[:self.max_len]
        
#         attention_mask = [1] * min(original_len, self.max_len) + [0] * max(0, self.max_len - original_len)

#         return {
#             'id': seq_id,
#             'sequence_str': input_seq_str,
#             'ss_str': str(ss),
#             'label': torch.tensor(label, dtype=torch.long),
#             # 'seq_target'不再需要，可以移除以节省内存
#             'ss_target': torch.tensor(ss_target_encoded, dtype=torch.long),
#             'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
#         }

# import os
# import torch
# from Bio import SeqIO
# from torch.utils.data import Dataset
# import random

# # 定义映射 (保持不变)
# AA_MAP = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX-")}
# SS_MAP = {ss: i for i, ss in enumerate("HCE-")}
# # 创建一个不包含特殊字符的氨基酸列表，用于随机替换
# STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# class PeptideDataset(Dataset):
#     def __init__(self, data_path, max_len, is_training=False, config=None):
#         """
#         一个灵活的数据集类，支持在线数据增强和为重建任务加噪。

#         Args:
#             data_path (str): 数据文件夹路径。
#             max_len (int): 序列最大长度。
#             is_training (bool): 是否为训练集。增强和加噪只在训练时应用。
#             config (EnhancedConfig): 包含所有控制参数的配置对象。
#         """
#         self.max_len = max_len
#         self.is_training = is_training
        
#         # 从config中获取所有控制参数，如果config未提供，则使用默认值
#         self.use_augmentation = getattr(config, 'USE_AUGMENTATION', False) if config else False
#         self.aug_prob = getattr(config, 'AUG_PROB', 0.5) if config else 0.5
#         self.aug_aa_sub_prob = getattr(config, 'AUG_AA_SUB_PROB', 0.1) if config else 0.1
        
#         self.use_noise_for_recon = getattr(config, 'RECON_USE_NOISE', False) if config else False
#         self.mask_prob_for_recon = getattr(config, 'RECON_MASK_PROB', 0.15) if config else 0.15
        
#         self.augmentation_factor = 1
#         if self.is_training and self.use_augmentation:
#             self.augmentation_factor = getattr(config, 'AUGMENTATION_FACTOR', 1)

#         # --- 数据加载与过滤 (与你之前正确的版本保持一致) ---
#         initial_records = []
#         if not os.path.exists(data_path) or not os.path.isdir(data_path):
#             print(f"Error: Data path '{data_path}' does not exist.")
#             self.records = []
#             return
            
#         print(f"Loading raw records from {data_path}...")
#         for filename in os.listdir(data_path):
#             if filename.endswith(".fas"):
#                 filepath = os.path.join(data_path, filename)
#                 try:
#                     initial_records.extend(list(SeqIO.parse(filepath, "fasta")))
#                 except Exception as e:
#                     print(f"Warning: Could not parse file {filepath}. Error: {e}")
        
#         print(f"Found {len(initial_records)} raw records. Filtering invalid sequences...")
        
#         self.records = []
#         num_skipped = 0
#         for record in initial_records:
#             full_string = str(record.seq)
#             if len(full_string) % 2 != 0:
#                 num_skipped += 1
#                 continue
#             split_point = len(full_string) // 2
#             if len(full_string[:split_point]) != len(full_string[split_point:]):
#                 num_skipped += 1
#                 continue
#             self.records.append(record)
        
#         if num_skipped > 0:
#             print(f"Filtered out {num_skipped} records with invalid length.")
        
#         print(f"Successfully loaded {len(self.records)} valid records from {data_path}.")
#         self.actual_num_records = len(self.records)
#         if self.is_training and self.augmentation_factor > 1:
#             print(f"Dataset augmentation enabled for training set. Factor: {self.augmentation_factor}.")
#             print(f"Original size: {self.actual_num_records}, effective size: {self.__len__()}")

#     def __len__(self):
#         return self.actual_num_records * self.augmentation_factor
    

#     def __getitem__(self, idx):
#         actual_idx = idx % self.actual_num_records
        
#         record = self.records[actual_idx]
#         seq_id, label_str = record.id.split('|')
#         label = int(label_str)

#         full_string = str(record.seq)
#         split_point = len(full_string) // 2
#         original_seq = full_string[:split_point]
#         ss = full_string[split_point:]
        
#         # --- 数据处理流程 (已修复) ---
#         # 1. 初始化输入序列
#         input_seq_list = list(original_seq)

#         # 2. 在线数据增强 (只在训练时进行)
#         if self.is_training and self.use_augmentation and random.random() < self.aug_prob:
#             num_to_sub = int(len(input_seq_list) * self.aug_aa_sub_prob)
#             if num_to_sub > 0:
#                 indices_to_sub = random.sample(range(len(input_seq_list)), num_to_sub)
#                 for i in indices_to_sub:
#                     original_aa = input_seq_list[i]
#                     possible_replacements = [aa for aa in STANDARD_AMINO_ACIDS if aa != original_aa]
#                     if possible_replacements:
#                         input_seq_list[i] = random.choice(possible_replacements)

#         # 3. 为重建任务加噪 (只在训练时进行)
#         if self.is_training and self.use_noise_for_recon and self.mask_prob_for_recon > 0:
#             for i in range(len(input_seq_list)):
#                 if random.random() < self.mask_prob_for_recon:
#                     rand_val = random.random()
#                     if rand_val < 0.8:
#                         input_seq_list[i] = 'X'
#                     elif rand_val < 0.9:
#                         input_seq_list[i] = random.choice(STANDARD_AMINO_ACIDS)
        
#         # --- !! 核心修正点 !! ---
#         # 无论是否进行了增强或加噪，最终都要将列表转换为字符串。
#         # 这一行代码必须在所有if块的外面。
#         final_input_seq_str = "".join(input_seq_list)
        
#         # 目标(target)始终使用最原始、最干净的二级结构序列
#         ss_target_encoded = [SS_MAP.get(s, SS_MAP['-']) for s in ss]

#         # 4. 填充和截断
#         original_len = len(ss_target_encoded)
#         padding_len = self.max_len - original_len
#         if padding_len > 0:
#             ss_target_encoded += [-1] * padding_len
#         ss_target_encoded = ss_target_encoded[:self.max_len]
        
#         attention_mask = [1] * min(original_len, self.max_len) + [0] * max(0, self.max_len - original_len)

#         return {
#             'id': seq_id,
#             'sequence_str': final_input_seq_str,
#             'ss_str': str(ss),
#             'label': torch.tensor(label, dtype=torch.long),
#             'ss_target': torch.tensor(ss_target_encoded, dtype=torch.long),
#             'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
#         }
    

# dataset.py (完整最终版)

# import os
# import torch
# import numpy as np
# import random
# from Bio import SeqIO
# from Bio.PDB import PDBParser
# from Bio.PDB.PDBExceptions import PDBConstructionWarning
# import warnings
# from torch.utils.data import Dataset
# from scipy.spatial.distance import pdist, squareform

# # --- 常量定义 ---
# AA_MAP = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX-")}
# SS_MAP = {ss: i for i, ss in enumerate("HCE-")}
# STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# PDB_TO_CANONICAL = {
#     'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
#     'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
#     'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
#     'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
# }


# def get_coords_from_pdb(pdb_path):
#     """从PDB文件中解析C-alpha坐标"""
#     parser = PDBParser(QUIET=True)
#     try:
#         structure = parser.get_structure("peptide", pdb_path)
#         coords = []
#         model = next(structure.get_models())
#         chain = next(model.get_chains())
#         for residue in chain.get_residues():
#             if "CA" in residue and residue.get_resname() in PDB_TO_CANONICAL:
#                 coords.append(residue["CA"].get_coord())
#         return np.array(coords) if coords else None
#     except Exception as e:
#         return None


# def get_contact_map(coords, threshold):
#     """根据坐标计算接触图"""
#     if coords is None or len(coords) == 0:
#         return None
#     distance_matrix = squareform(pdist(coords, 'euclidean'))
#     return (distance_matrix < threshold).astype(np.float32)


# class PeptideDataset(Dataset):
#     def __init__(self, data_path, pdb_dirs, max_len, is_training=False, config=None):
#         """
#         融合PDB结构信息的肽段数据集
        
#         Args:
#             data_path: FASTA文件目录
#             pdb_dirs: 标签到PDB目录的映射，例如 {1: './amp_pdb', 0: './nonamp_pdb'}
#             max_len: 序列最大长度
#             is_training: 是否为训练集
#             config: 配置对象
#         """
#         self.max_len = max_len
#         self.is_training = is_training
#         self.config = config
        
#         # 数据增强参数
#         self.use_augmentation = getattr(config, 'USE_AUGMENTATION', False) if config else False
#         self.aug_prob = getattr(config, 'AUG_PROB', 0.5) if config else 0.5
#         self.aug_aa_sub_prob = getattr(config, 'AUG_AA_SUB_PROB', 0.1) if config else 0.1
        
#         # 重建任务加噪参数
#         self.use_noise_for_recon = getattr(config, 'RECON_USE_NOISE', False) if config else False
#         self.mask_prob_for_recon = getattr(config, 'RECON_MASK_PROB', 0.15) if config else 0.15
        
#         # 增强因子
#         self.augmentation_factor = 1
#         if self.is_training and self.use_augmentation:
#             self.augmentation_factor = getattr(config, 'AUGMENTATION_FACTOR', 1)
        
#         # 接触图阈值
#         self.contact_threshold = getattr(config, 'CONTACT_MAP_THRESHOLD', 8.0) if config else 8.0
        
#         # 加载数据
#         self.records = []
#         num_pdb_not_found = 0
#         num_pdb_parse_failed = 0
        
#         print(f"Loading records from {data_path} and linking to PDBs...")
        
#         if not os.path.exists(data_path) or not os.path.isdir(data_path):
#             print(f"Error: Data path '{data_path}' does not exist.")
#             return
        
#         for filename in os.listdir(data_path):
#             if filename.endswith(".fas"):
#                 filepath = os.path.join(data_path, filename)
#                 try:
#                     for record in SeqIO.parse(filepath, "fasta"):
#                         # 原始过滤逻辑
#                         full_string = str(record.seq)
#                         if len(full_string) % 2 != 0:
#                             continue
                        
#                         split_point = len(full_string) // 2
#                         if len(full_string[:split_point]) != len(full_string[split_point:]):
#                             continue
                        
#                         # 解析ID和标签
#                         seq_id, label_str = record.id.split('|')
#                         label = int(label_str)
                        
#                         # 查找对应的PDB文件
#                         pdb_filename = f"{seq_id}.pdb"
#                         pdb_path = os.path.join(pdb_dirs[label], pdb_filename)
                        
#                         if not os.path.exists(pdb_path):
#                             num_pdb_not_found += 1
#                             continue
                        
#                         # 验证PDB文件可以正确解析
#                         coords = get_coords_from_pdb(pdb_path)
#                         original_seq = full_string[:split_point]
                        
#                         if coords is None or len(coords) != len(original_seq):
#                             num_pdb_parse_failed += 1
#                             continue
                        
#                         # 只有通过所有检查的才加入
#                         self.records.append((record, pdb_path))
                
#                 except Exception as e:
#                     print(f"Warning: Could not parse file {filepath}. Error: {e}")
        
#         if num_pdb_not_found > 0:
#             print(f"Warning: {num_pdb_not_found} records skipped due to missing PDB files.")
#         if num_pdb_parse_failed > 0:
#             print(f"Warning: {num_pdb_parse_failed} records skipped due to PDB parsing issues.")
        
#         print(f"Successfully loaded {len(self.records)} valid records with both FASTA and PDB data.")
        
#         self.actual_num_records = len(self.records)
#         if self.is_training and self.augmentation_factor > 1:
#             print(f"Dataset augmentation enabled. Effective size: {len(self)}")

#     def __len__(self):
#         return self.actual_num_records * self.augmentation_factor

#     def __getitem__(self, idx):
#         actual_idx = idx % self.actual_num_records
#         record, pdb_path = self.records[actual_idx]
        
#         # 解析FASTA
#         seq_id, label_str = record.id.split('|')
#         label = int(label_str)
#         full_string = str(record.seq)
#         split_point = len(full_string) // 2
#         original_seq = full_string[:split_point]
#         ss = full_string[split_point:]
        
#         # 解析PDB获取坐标
#         coords = get_coords_from_pdb(pdb_path)
#         if coords is None or len(coords) != len(original_seq):
#             return None  # 返回None，让collate_fn过滤
        
#         # 数据处理流程
#         input_seq_list = list(original_seq)
        
#         # 在线数据增强（仅训练时）
#         if self.is_training and self.use_augmentation and random.random() < self.aug_prob:
#             num_to_sub = int(len(input_seq_list) * self.aug_aa_sub_prob)
#             if num_to_sub > 0:
#                 indices_to_sub = random.sample(range(len(input_seq_list)), num_to_sub)
#                 for i in indices_to_sub:
#                     original_aa = input_seq_list[i]
#                     possible_replacements = [aa for aa in STANDARD_AMINO_ACIDS if aa != original_aa]
#                     if possible_replacements:
#                         input_seq_list[i] = random.choice(possible_replacements)
        
#         # 为重建任务加噪（仅训练时）
#         if self.is_training and self.use_noise_for_recon and self.mask_prob_for_recon > 0:
#             for i in range(len(input_seq_list)):
#                 if random.random() < self.mask_prob_for_recon:
#                     rand_val = random.random()
#                     if rand_val < 0.8:
#                         input_seq_list[i] = 'X'
#                     elif rand_val < 0.9:
#                         input_seq_list[i] = random.choice(STANDARD_AMINO_ACIDS)
        
#         final_input_seq_str = "".join(input_seq_list)
#         ss_target_encoded = [SS_MAP.get(s, SS_MAP['-']) for s in ss]
        
#         # 截断和填充（同步处理序列、SS和坐标）
#         original_len = len(final_input_seq_str)
        
#         if original_len > self.max_len:
#             final_input_seq_str = final_input_seq_str[:self.max_len]
#             ss_target_encoded = ss_target_encoded[:self.max_len]
#             coords = coords[:self.max_len]
#             original_len = self.max_len
        
#         # 计算接触图（在截断后）
#         contact_map = get_contact_map(coords, self.contact_threshold)
#         if contact_map is None:
#             return None
        
#         # 填充
#         padding_len = self.max_len - original_len
#         if padding_len > 0:
#             ss_target_encoded += [-1] * padding_len
        
#         attention_mask = [1] * original_len + [0] * padding_len
        
#         return {
#             'id': seq_id,
#             'sequence_str': final_input_seq_str,
#             'label': torch.tensor(label, dtype=torch.float),
#             'ss_target': torch.tensor(ss_target_encoded, dtype=torch.long),
#             'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
#             'contact_map': torch.from_numpy(contact_map)
#         }


# def custom_collate_fn(batch):
#     """
#     自定义collate函数，处理变长接触图的填充
#     """
#     # 过滤掉None样本
#     batch = [item for item in batch if item is not None]
#     if not batch:
#         return None
    
#     # 提取各项数据
#     ids = [item['id'] for item in batch]
#     seq_strs = [item['sequence_str'] for item in batch]
#     labels = torch.stack([item['label'] for item in batch])
#     ss_targets = torch.stack([item['ss_target'] for item in batch])
#     attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
#     # 对contact_map进行填充
#     max_len_in_batch = max(item['contact_map'].shape[0] for item in batch)
#     padded_contact_maps = []
    
#     for item in batch:
#         cm = item['contact_map']
#         seq_len = cm.shape[0]
#         padding_size = max_len_in_batch - seq_len
#         padded_cm = torch.nn.functional.pad(cm, (0, padding_size, 0, padding_size), "constant", 0)
#         padded_contact_maps.append(padded_cm)
    
#     contact_maps = torch.stack(padded_contact_maps)
    
#     return {
#         'id': ids,
#         'sequence_str': seq_strs,
#         'label': labels,
#         'ss_target': ss_targets,
#         'attention_mask': attention_masks,
#         'contact_maps': contact_maps
#     }


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

# --- 常量定义 ---
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

# ============ 几何特征提取 ============

def calculate_dihedrals(coords):
    """计算主链二面角 φ, ψ, ω"""
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
    计算局部曲率 (基于三点角度)
    对每个残基 i，计算 i-1, i, i+1 形成的角度
    """
    L = len(coords_ca)
    curvature = np.zeros(L)
    
    for i in range(1, L-1):
        v1 = coords_ca[i] - coords_ca[i-1]
        v2 = coords_ca[i+1] - coords_ca[i]
        
        # 归一化
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # 计算夹角 (弧度)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # 曲率 = π - angle (直线时为0，弯曲时增大)
        curvature[i] = np.pi - angle
    
    return curvature


def get_full_coords_from_pdb(pdb_path):
    """提取主链原子坐标 (N, CA, C, O) 和 CB"""
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
                
                # CB (甘氨酸用虚拟CB)
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
    """简化版SASA计算"""
    dist_matrix = squareform(pdist(coords_ca, 'euclidean'))
    neighbor_counts = (dist_matrix < neighbor_cutoff).sum(axis=1) - 1
    max_neighbors = neighbor_counts.max() if neighbor_counts.max() > 0 else 1
    rsa = 1.0 - (neighbor_counts / max_neighbors)
    return rsa


# 氨基酸理化性质表
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
    """返回 [L, 4] 的理化性质矩阵"""
    return np.array([AA_PROPERTIES.get(aa, [0,0,0,0]) for aa in sequence])


# ============ 图构建 ============

def build_knn_edges(coords_ca, k=20):
    """构建KNN图"""
    L = len(coords_ca)
    dist_matrix = squareform(pdist(coords_ca, 'euclidean'))
    
    edges = []
    for i in range(L):
        neighbors = np.argsort(dist_matrix[i])[1:k+1]
        for j in neighbors:
            edges.append([i, j])
    
    return np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int64)


def build_sequential_edges(length, window=2):
    """构建序列邻居边"""
    edges = []
    for i in range(length):
        for delta in range(1, window + 1):
            if i + delta < length:
                edges.append([i, i + delta])
                edges.append([i + delta, i])
    return np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int64)


def build_contact_edges(coords_ca, threshold=5.0):
    """
    构建接触图边 (5Å 阈值)
    """
    L = len(coords_ca)
    dist_matrix = squareform(pdist(coords_ca, 'euclidean'))
    
    edges = []
    for i in range(L):
        for j in range(i+1, L):
            if dist_matrix[i, j] < threshold:
                edges.append([i, j])
                edges.append([j, i])  # 双向
    
    return np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int64)


def build_ss_motif_edges(ss_labels):
    """
    构建二级结构motif图
    - 同一helix/sheet内的残基全连接
    - 不同SS元素的简化连接
    """
    L = len(ss_labels)
    edges = []
    
    # 识别连续的SS片段
    segments = []
    current_seg = [0]
    current_type = ss_labels[0]
    
    for i in range(1, L):
        if ss_labels[i] == current_type:
            current_seg.append(i)
        else:
            if current_type in ['H', 'E']:  # 只关心helix和sheet
                segments.append((current_type, current_seg))
            current_seg = [i]
            current_type = ss_labels[i]
    
    if current_type in ['H', 'E']:
        segments.append((current_type, current_seg))
    
    # 同一片段内全连接
    for ss_type, seg in segments:
        for i in seg:
            for j in seg:
                if i != j:
                    edges.append([i, j])
    
    return np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int64)


def detect_hbonds(coords, distance_cutoff=3.5, angle_cutoff=120):
    """
    简化氢键检测
    检测 N-H···O=C 氢键
    条件: N···O距离 < 3.5Å 且角度合适
    """
    L = len(coords['N'])
    hbonds = []
    
    for i in range(L):
        for j in range(L):
            if abs(i - j) < 3:  # 跳过太近的残基
                continue
            
            # 计算 N(i) 到 O(j) 距离
            n_o_dist = np.linalg.norm(coords['N'][i] - coords['O'][j])
            
            if n_o_dist < distance_cutoff:
                # 简化：认为可能存在氢键
                hbonds.append((i, j))
    
    return hbonds


# ============ 边特征计算 ============

def calculate_edge_features_advanced(coords, edge_index, ss_labels, phi, psi, omega):
    """
    计算增强版边特征
    包含：距离、方向、二面角差、氢键特征
    """
    E = edge_index.shape[1]
    edge_features = []
    
    # 预计算氢键
    hbonds_set = set(detect_hbonds(coords))
    
    for e in range(E):
        i, j = edge_index[:, e]
        
        # 1. 距离特征
        ca_dist = np.linalg.norm(coords['CA'][i] - coords['CA'][j])
        cb_dist = np.linalg.norm(coords['CB'][i] - coords['CB'][j])
        
        # 2. 方向向量 (归一化) ⭐ 新增用于GVP
        direction = coords['CA'][j] - coords['CA'][i]
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
        
        # 3. 序列距离
        seq_dist = abs(i - j)
        
        # 4. RBF距离编码
        rbf_min, rbf_max, rbf_bins = 0.0, 20.0, 16
        rbf_centers = np.linspace(rbf_min, rbf_max, rbf_bins)
        rbf_gamma = 1.0
        rbf_encoded = np.exp(-rbf_gamma * (ca_dist - rbf_centers)**2)
        
        # 5. SS配对
        ss_same = 1.0 if ss_labels[i] == ss_labels[j] else 0.0
        
        # ========== 新增特征 ==========
        
        # 6. 二面角差 ⭐
        delta_phi = phi[j] - phi[i]
        delta_psi = psi[j] - psi[i]
        delta_omega = omega[j] - omega[i]
        
        # 7. 氢键特征 ⭐
        is_hbond = 1.0 if (i, j) in hbonds_set or (j, i) in hbonds_set else 0.0
        
        # 拼接
        feat = np.concatenate([
            [ca_dist, cb_dist],          # 2
            direction_norm,               # 3 ⭐ 用于GVP
            [seq_dist],                  # 1
            rbf_encoded,                 # 16
            [ss_same],                   # 1
            [delta_phi, delta_psi, delta_omega],  # 3 ⭐ 新增
            [is_hbond]                   # 1 ⭐ 新增
        ])
        edge_features.append(feat)
    
    return np.array(edge_features)  # [E, 27]


def get_coords_from_pdb(pdb_path):
    """从PDB文件中解析C-alpha坐标 (向后兼容)"""
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
    """根据坐标计算接触图 (向后兼容)"""
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
        
        # 参数
        self.use_augmentation = getattr(config, 'USE_AUGMENTATION', False) if config else False
        self.aug_prob = getattr(config, 'AUG_PROB', 0.5) if config else 0.5
        self.aug_aa_sub_prob = getattr(config, 'AUG_AA_SUB_PROB', 0.1) if config else 0.1
        self.use_noise_for_recon = getattr(config, 'RECON_USE_NOISE', False) if config else False
        self.mask_prob_for_recon = getattr(config, 'RECON_MASK_PROB', 0.15) if config else 0.15
        
        self.augmentation_factor = 1
        if self.is_training and self.use_augmentation:
            self.augmentation_factor = getattr(config, 'AUGMENTATION_FACTOR', 1)
        
        self.contact_threshold = getattr(config, 'CONTACT_MAP_THRESHOLD', 8.0) if config else 8.0
                # ========== 新增参数 ==========
        self.knn_k = getattr(config, 'KNN_K', 20) if config else 20
        self.contact_cutoff = getattr(config, 'CONTACT_CUTOFF', 5.0) if config else 5.0
        self.use_ss_motif_graph = getattr(config, 'USE_SS_MOTIF_GRAPH', True) if config else True
        
        # 加载数据
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
                        
                        # 验证PDB (使用新函数)
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
        
        # 解析FASTA
        seq_id, label_str = record.id.split('|')
        label = int(label_str)
        full_string = str(record.seq)
        split_point = len(full_string) // 2
        original_seq = full_string[:split_point]
        ss = full_string[split_point:]
        
        # ========== 提取完整坐标 ==========
        coords = get_full_coords_from_pdb(pdb_path)
        if coords is None or len(coords['CA']) != len(original_seq):
            return None
        
        # 数据增强
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
        
        # 重建任务加噪
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
        
        # 截断
        original_len = len(final_input_seq_str)
        if original_len > self.max_len:
            final_input_seq_str = final_input_seq_str[:self.max_len]
            ss_target_encoded = ss_target_encoded[:self.max_len]
            ss = ss[:self.max_len]
            for key in coords:
                coords[key] = coords[key][:self.max_len]
            original_len = self.max_len
        
        # ========== 计算几何特征 ==========
        # 1. 二面角
        phi, psi, omega = calculate_dihedrals(coords)
        
        # 2. 局部曲率 ⭐ 新增
        curvature = calculate_local_curvature(coords['CA'])
        
        # 3. SASA代理
        rsa = calculate_sasa_simple(coords['CA'])
        
        # 4. 理化性质
        aa_props = get_aa_properties(final_input_seq_str)
        
        # 5. 拼接节点几何特征
        node_geometric_features = np.concatenate([
            coords['CA'],           # [L, 3]
            coords['CB'],           # [L, 3] ⭐ 新增
            phi[:, None],           # [L, 1]
            psi[:, None],           # [L, 1]
            omega[:, None],         # [L, 1]
            curvature[:, None],     # [L, 1] ⭐ 新增
            rsa[:, None],           # [L, 1]
            aa_props,               # [L, 4]
        ], axis=1)  # [L, 15] (原来14 → 现在15)
        
        # ========== 构建多图边 (升级版) ==========
        knn_edges = build_knn_edges(coords['CA'], k=self.knn_k)
        seq_edges = build_sequential_edges(original_len, window=2)
        contact_edges = build_contact_edges(coords['CA'], threshold=self.contact_cutoff)  # ⭐ 新增
        
        # SS-motif 图 ⭐ 新增
        if self.use_ss_motif_graph:
            ss_edges = build_ss_motif_edges(list(ss))
        else:
            ss_edges = np.zeros((2, 0), dtype=np.int64)
        
        # 合并所有边
        all_edges = np.concatenate([knn_edges, seq_edges, contact_edges, ss_edges], axis=1)
        all_edges = np.unique(all_edges, axis=1)  # 去重
        
        # ========== 计算增强版边特征 ==========
        edge_attr = calculate_edge_features_advanced(
            coords, all_edges, list(ss), phi, psi, omega
        )  # [E, 27] (原来23 → 现在27)
        
        # Contact map (保留兼容)
        contact_map = get_contact_map(coords['CA'], self.contact_threshold)
        if contact_map is None:
            return None
        
        # 填充
        padding_len = self.max_len - original_len
        if padding_len > 0:
            ss_target_encoded += [-1] * padding_len
            attention_mask = [1] * original_len + [0] * padding_len
            
            # 填充几何特征
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
            
            # ========== 升级特征 ==========
            'node_geometric_feat': torch.from_numpy(node_geometric_features).float(),  # [max_len, 15]
            'edge_index': torch.from_numpy(all_edges).long(),  # [2, E]
            'edge_attr': torch.from_numpy(edge_attr).float(),  # [E, 27]
            
            # ========== 新增：用于GVP的向量特征 ==========
            'node_coords': torch.from_numpy(coords['CA'][:original_len]).float(),  # [L, 3] 原始坐标
        }


def custom_collate_fn(batch):
    """升级版collate函数"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # 原有字段
    ids = [item['id'] for item in batch]
    seq_strs = [item['sequence_str'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    ss_targets = torch.stack([item['ss_target'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
    # Contact map填充
    max_len_in_batch = max(item['contact_map'].shape[0] for item in batch)
    padded_contact_maps = []
    for item in batch:
        cm = item['contact_map']
        seq_len = cm.shape[0]
        padding_size = max_len_in_batch - seq_len
        padded_cm = torch.nn.functional.pad(cm, (0, padding_size, 0, padding_size), "constant", 0)
        padded_contact_maps.append(padded_cm)
    contact_maps = torch.stack(padded_contact_maps)
    
    # 几何特征
    node_geometric_feats = torch.stack([item['node_geometric_feat'] for item in batch])  # [B, L, 15]
    
    # ========== 新增：坐标填充 (用于GVP) ==========
    padded_coords = []
    for item in batch:
        coords = item['node_coords']
        seq_len = coords.shape[0]
        padding_size = max_len_in_batch - seq_len
        padded_coord = torch.nn.functional.pad(coords, (0, 0, 0, padding_size), "constant", 0)
        padded_coords.append(padded_coord)
    node_coords = torch.stack(padded_coords)  # [B, L, 3]
    
    # 合并边
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
        
        # 升级特征
        'node_geometric_feat': node_geometric_feats,  # [B, L, 15]
        'edge_index': batch_edge_index,               # [2, E]
        'edge_attr': batch_edge_attr,                 # [E, 27]
        'node_coords': node_coords,                   # [B, L, 3] ⭐ 新增
    }