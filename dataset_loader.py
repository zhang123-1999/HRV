from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from torch.utils.data import Dataset


class RadarECGDataset(Dataset):
    """
    Windowed dataset with Data Augmentation for training.
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 1200,
        stride: int = 120,
        mode: str = "train",
        split_ratio: float = 0.8,
        test_size: int = 20,
        fs: float = 120.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.fs = float(fs)

        valid_modes = {"train", "val", "test"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        self.mode = mode

        all_files: List[Path] = sorted(list(self.data_dir.glob("*.npz")))
        total_files = len(all_files)

        # 划分数据集逻辑 (保持不变)
        if total_files <= test_size:
            test_files = all_files[-1:]
            remaining_files = all_files[:-1]
        else:
            test_files = all_files[-test_size:]
            remaining_files = all_files[:-test_size]

        split_idx = int(len(remaining_files) * split_ratio)
        if split_idx == 0: split_idx = 1
        
        train_files = remaining_files[:split_idx]
        val_files = remaining_files[split_idx:]

        if self.mode == "test":
            self.session_files = test_files
        elif self.mode == "train":
            self.session_files = train_files
        elif self.mode == "val":
            self.session_files = val_files
        
        if not self.session_files:
            raise ValueError(f"No sessions available for mode={self.mode}")

        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        
        print(f"[{self.mode.upper()}] Loading data from {len(self.session_files)} sessions...")
        self._load_samples()
        print(f"[{self.mode.upper()}] Loaded {len(self.samples)} samples.")

    def _check_quality(self, radar: np.ndarray, dist: np.ndarray) -> bool:
        """宽松版质量检测"""
        if np.std(radar) < 1e-6: return False
        # 反转找峰
        peaks, _ = find_peaks(-dist, height=-0.1, distance=10)
        num_peaks = len(peaks)
        # 允许 30 ~ 180 BPM
        if num_peaks < 5 or num_peaks > 30: return False
        
        rr_intervals = np.diff(peaks) / self.fs
        if len(rr_intervals) < 1: return False
        # 允许 0.3s ~ 2.0s
        if np.any(rr_intervals < 0.3) or np.any(rr_intervals > 2.0): return False
        
        return True

    def _load_samples(self) -> None:
        for file_path in self.session_files:
            try:
                with np.load(file_path) as data:
                    radar = np.asarray(data["radar_120"], dtype=np.float32).reshape(-1)
                    dist = np.asarray(data["dist_norm"], dtype=np.float32).reshape(-1)

                n = min(radar.size, dist.size)
                if n < self.window_size: continue

                max_start = n - self.window_size
                current_stride = self.stride 
                
                for start in range(0, max_start + 1, current_stride):
                    end = start + self.window_size
                    radar_slice = radar[start:end]
                    dist_slice = dist[start:end]
                    
                    if radar_slice.shape[0] == self.window_size and dist_slice.shape[0] == self.window_size:
                        # 仅在加载时做质量检查，不做增强
                        if self._check_quality(radar_slice, dist_slice):
                            self.samples.append((radar_slice, dist_slice))
                                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        radar_np, dist_np = self.samples[idx]
        
        # === 关键：仅在训练模式下进行数据增强 ===
        if self.mode == 'train':
            # 1. 随机重采样 (Simulate Heart Rate Variation)
            # 随机拉伸/压缩 0.8x ~ 1.2x
            scale = np.random.uniform(0.8, 1.2)
            
            orig_len = len(radar_np)
            x_old = np.arange(orig_len)
            
            # 创建插值函数
            f_radar = interp1d(x_old, radar_np, kind='linear', fill_value="extrapolate")
            f_dist = interp1d(x_old, dist_np, kind='linear', fill_value="extrapolate")
            
            # 生成新的时间轴 (模拟拉伸或压缩)
            x_new = np.linspace(0, orig_len - 1, int(orig_len * scale))
            
            radar_aug = f_radar(x_new)
            # 注意：如果时间拉伸了，距离图的值(代表样本数)也应该相应放大
            dist_aug = f_dist(x_new) * scale 
            dist_aug = np.clip(dist_aug, 0, 1.0) # 保持归一化范围
            
            # 裁剪或填充回 1200 长度
            if len(radar_aug) > self.window_size:
                # 随机裁剪
                diff = len(radar_aug) - self.window_size
                start = np.random.randint(0, diff + 1)
                radar_aug = radar_aug[start : start + self.window_size]
                dist_aug = dist_aug[start : start + self.window_size]
            else:
                # 边缘填充
                pad_total = self.window_size - len(radar_aug)
                pad_l = pad_total // 2
                pad_r = pad_total - pad_l
                radar_aug = np.pad(radar_aug, (pad_l, pad_r), 'edge')
                dist_aug = np.pad(dist_aug, (pad_l, pad_r), 'edge')
            
            radar_np = radar_aug
            dist_np = dist_aug

            # 2. 随机幅度缩放
            radar_np = radar_np * np.random.uniform(0.8, 1.2)

            # 3. 随机高斯噪声
            noise = np.random.normal(0, 0.02, size=radar_np.shape) # 较小的噪声
            radar_np = radar_np + noise

        # 转为 Tensor
        x = torch.from_numpy(radar_np).unsqueeze(0).float()
        y = torch.from_numpy(dist_np).unsqueeze(0).float()
        return x, y