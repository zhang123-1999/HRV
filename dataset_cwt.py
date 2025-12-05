import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import logging

class RadarSpectrogramDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        split_ratio: float = 0.8,
        test_subjects: int = 5,
        window_size: int = 1200, # 10秒 @ 120Hz
        stride: int = 120,       # 滑动步长
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        
        # 1. 加载文件列表
        all_files = sorted(list(self.data_dir.glob("*.npz")))
        subject_map = {}
        for f in all_files:
            sid = f.name.split('_')[0]
            if sid not in subject_map: subject_map[sid] = []
            subject_map[sid].append(f)
        
        all_subjects = sorted(list(subject_map.keys()))
        if len(all_subjects) <= test_subjects: test_subjects = 1
        
        test_subj_ids = all_subjects[-test_subjects:]
        train_subj_ids = all_subjects[:-test_subjects]
        
        self.session_files = []
        if self.mode == 'test':
            for sid in test_subj_ids:
                # Test只选Resting
                self.session_files.extend([f for f in subject_map[sid] if "resting" in f.name.lower()])
        else:
            files = []
            for sid in train_subj_ids: files.extend(subject_map[sid])
            split_idx = int(len(files) * split_ratio)
            if self.mode == 'train': self.session_files = files[:split_idx]
            else: self.session_files = files[split_idx:]

        self.samples = []
        self._preload_data()
        print(f"[{mode.upper()}] Loaded {len(self.samples)} samples from {len(self.session_files)} files.")

    def _preload_data(self):
        """
        预加载数据：直接读取预处理好的 Spectrogram 和 IHR Curve
        """
        for fpath in self.session_files:
            try:
                with np.load(fpath) as data:
                    # === 关键修正 ===
                    # 直接读取 build_nature_dataset 生成的新 Key
                    if "spectrogram" not in data or "ihr_curve" not in data:
                        continue
                        
                    # spec shape: [64, T_total]
                    spec = data["spectrogram"].astype(np.float32)
                    # ihr shape: [T_total]
                    ihr = data["ihr_curve"].astype(np.float32)
                
                # 确保长度一致
                n = spec.shape[1]
                if n < self.window_size: continue
                
                # 切片 (Sliding Window)
                for start in range(0, n - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    
                    # 切片
                    spec_slice = spec[:, start:end] # [64, 1200]
                    ihr_slice = ihr[start:end]      # [1200]
                    
                    # 简单的质量检查: 
                    # 检查 IHR 曲线是否包含太多 0 (无效值) 或者 NaN
                    if np.mean(ihr_slice) < 40 or np.isnan(np.sum(ihr_slice)):
                        continue

                    # 直接存入内存
                    self.samples.append((spec_slice, ihr_slice))
                    
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

    def __getitem__(self, idx):
        # 取出数据
        spec_np, ihr_np = self.samples[idx]
        
        # === 数据增强 (仅训练时) ===
        if self.mode == 'train':
            # 1. 频谱掩码 (SpecAugment): 随机遮挡一段频率或时间，增强抗噪性
            if np.random.rand() < 0.3:
                # 频域遮挡
                f_mask = np.random.randint(0, 10)
                f_start = np.random.randint(0, 64 - f_mask)
                spec_np[f_start:f_start+f_mask, :] = 0
            
            # 2. 随机高斯噪声
            noise = np.random.normal(0, 0.05, spec_np.shape)
            spec_np = spec_np + noise

        # === 格式转换 ===
        # Input: Spectrogram
        # spec_np 是 [64, 1200], ResNet 需要 [Channel, Freq, Time]
        # 所以增加一个维度 -> [1, 64, 1200]
        x = torch.from_numpy(spec_np).unsqueeze(0).float()
        
        # Target: IHR Curve
        # ihr_np 是 [1200]
        y = torch.from_numpy(ihr_np).float()
        
        return x, y

    def __len__(self):
        return len(self.samples)