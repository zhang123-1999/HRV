import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import logging
import pandas as pd
from scipy.interpolate import interp1d

class RadarSpectrogramDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        split_ratio: float = 0.8,
        test_subjects: int = 5,
        window_size: int = 1200, 
        stride: int = 120,
        # === 新增参数: 过采样倍率 ===
        oversample_factor: int = 3  # 困难样本出现频率增加 3 倍
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.oversample_factor = oversample_factor
        
        index_file = self.data_dir / "dataset_index.csv"
        if not index_file.exists():
            raise FileNotFoundError("Index file not found. Run build_nature_dataset.py first.")
            
        df = pd.read_csv(index_file)
        
        # 简单的按比例划分
        split_idx = int(len(df) * split_ratio)
        if mode == "train":
            self.file_paths = df.iloc[:split_idx]['path'].tolist()
        else:
            self.file_paths = df.iloc[split_idx:]['path'].tolist()

        self.samples = []
        self._preload_data() # 在这里做过采样
        
        print(f"[{mode.upper()}] Final Dataset Size: {len(self.samples)} (Original: {len(self.file_paths)})")

    def _preload_data(self):
        """
        预加载 + 困难样本挖掘 (Hard Mining via Oversampling)
        """
        normal_count = 0
        hard_count = 0
        
        for path in self.file_paths:
            try:
                if not Path(path).exists():
                    continue
                    
                with np.load(path) as data:
                    if "spectrogram" not in data or "ihr_curve" not in data: continue
                    spec = data["spectrogram"].astype(np.float32)
                    ihr = data["ihr_curve"].astype(np.float32)
                
                n = spec.shape[1]
                if n < self.window_size: continue
                
                # 质量检查
                if np.mean(ihr) < 30 or np.isnan(np.sum(ihr)): continue
                
                # === 核心逻辑: 困难样本判定 ===
                # 计算这段心率的变异性 (标准差)
                # 一般静息心率标准差 < 2 BPM。如果 > 3 BPM，说明波动剧烈，是难样本。
                ihr_std = np.std(ihr)
                is_hard_sample = (ihr_std > 3.0) 

                # 存入列表 (tuple 是不可变的，比较安全)
                sample_tuple = (spec, ihr)
                
                # 如果是普通样本，加 1 次
                self.samples.append(sample_tuple)
                normal_count += 1
                
                # === 如果是困难样本，且在训练模式，多加几次 ===
                if self.mode == 'train' and is_hard_sample:
                    for _ in range(self.oversample_factor - 1):
                        self.samples.append(sample_tuple)
                    hard_count += 1
                    
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        if self.mode == 'train':
            print(f"   >>> Oversampling Stats: Normal={normal_count}, Hard={hard_count} (x{self.oversample_factor})")

    def __getitem__(self, idx):
        spec_np, ihr_np = self.samples[idx]
        
        # Copy to avoid modifying the cached data if augmentation is applied in place
        spec = spec_np.copy()
        ihr = ihr_np.copy()
        
        # 数据增强 (仅训练)
        if self.mode == 'train':
            # SpecAugment: 随机遮挡频段
            if np.random.rand() < 0.2:
                f_start = np.random.randint(0, 50)
                # 边界检查
                if f_start + 5 < spec.shape[0]:
                    spec[f_start:f_start+5, :] = 0
            
            # Additive Noise
            noise = np.random.normal(0, 0.05, spec.shape)
            spec = spec + noise
        
        # === 格式转换 ===
        x = torch.from_numpy(spec).unsqueeze(0).float() # [1, 64, 1200]
        
        # [可选] 如果使用了超分模型 (输出4800)，需要在这里对 Label 进行插值
        # 默认这里输出原始 1200 长度
        y = torch.from_numpy(ihr).float()
        
        # 如果启用超分，取消下面注释：
        target_len = 4800
        t_old = np.linspace(0, 1, len(ihr))
        t_new = np.linspace(0, 1, target_len)
        f = interp1d(t_old, ihr, kind='linear')
        ihr_high_res = f(t_new).astype(np.float32)
        y = torch.from_numpy(ihr_high_res).float()

        return x, y

    def __len__(self):
        return len(self.samples)