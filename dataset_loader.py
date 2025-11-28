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
    Returns: (radar_tensor, dist_map_tensor, count_tensor)
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 1200,
        stride: int = 120,
        mode: str = "train",
        split_ratio: float = 0.8,
        test_subjects: int = 5,
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

        # 1. 按受试者分组
        all_files = sorted(list(self.data_dir.glob("*.npz")))
        subject_map = {}
        for f in all_files:
            sid = f.name.split('_')[0] 
            if sid not in subject_map: subject_map[sid] = []
            subject_map[sid].append(f)
        
        all_subjects = sorted(list(subject_map.keys()))
        total_subjs = len(all_subjects)
        
        if total_subjs <= test_subjects:
            test_subjects = 1 # Fallback

        test_subj_ids = all_subjects[-test_subjects:]
        train_subj_ids = all_subjects[:-test_subjects]

        self.session_files = []
        if self.mode == 'test':
            for sid in test_subj_ids:
                # Test只选Resting
                self.session_files.extend([f for f in subject_map[sid] if "resting" in f.name.lower()])
        else:
            train_val_files = []
            for sid in train_subj_ids:
                # Train/Val选所有
                train_val_files.extend(subject_map[sid])
                # Train/Val 也只选 Resting
                #train_val_files.extend([f for f in subject_map[sid] if "resting" in f.name.lower()])
            
            split_point = int(len(train_val_files) * split_ratio)
            if self.mode == 'train':
                self.session_files = train_val_files[:split_point]
            else:
                self.session_files = train_val_files[split_point:]

        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        
        if len(self.session_files) > 0:
            print(f"[{self.mode.upper()}] Loading from {len(self.session_files)} sessions...")
            self._load_samples()
            print(f"[{self.mode.upper()}] Loaded {len(self.samples)} windows.")

    def _check_quality(self, radar: np.ndarray, dist: np.ndarray) -> bool:
        # 1. 信号强度检查
        if np.std(radar) < 1e-6: return False
        
        # 2. 峰值提取 (Ground Truth)
        peaks, _ = find_peaks(-dist, height=-0.1, distance=10)
        
        # [数量检查 - 更严格]
        # 10秒窗口: 
        # < 8 (48 BPM): 可能是心动过缓或漏检
        # > 25 (150 BPM): 可能是运动伪影或误检
        if len(peaks) < 8 or len(peaks) > 25: return False
        
        rr_intervals = np.diff(peaks) / self.fs
        if len(rr_intervals) < 1: return False

        # [生理极限检查]
        # RR < 0.4s (150 BPM) 或 RR > 1.4s (43 BPM)
        if np.any(rr_intervals < 0.4) or np.any(rr_intervals > 1.4):
            return False
        
        # [稳定性检查]
        diff_rr = np.diff(rr_intervals)
        if len(diff_rr) > 0:
            # RMSSD 检查: 降低阈值到 150ms (原 200ms)
            rmssd = np.sqrt(np.mean(diff_rr**2))
            if rmssd > 0.15: return False
            
            # 相对变化率检查: 任何相邻 RR 变化不应超过 20%
            rr_pct = np.abs(diff_rr) / rr_intervals[:-1]
            if np.any(rr_pct > 0.2): return False

        return True

    def _load_samples(self) -> None:
        for file_path in self.session_files:
            try:
                with np.load(file_path) as data:
                    radar = np.asarray(data["radar_120"], dtype=np.float32).reshape(-1)
                    dist = np.asarray(data["dist_norm"], dtype=np.float32).reshape(-1)

                n = min(radar.size, dist.size)
                if n < self.window_size: continue

                for start in range(0, n - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    radar_slice = radar[start:end]
                    dist_slice = dist[start:end]
                    
                    if self._check_quality(radar_slice, dist_slice):
                        self.samples.append((radar_slice, dist_slice))
                                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radar_np, dist_np = self.samples[idx]
        
        # === 数据增强 ===
        if self.mode == 'train':
            scale = np.random.uniform(0.7, 1.3)
            orig_len = len(radar_np)
            x_old = np.arange(orig_len)
            f_radar = interp1d(x_old, radar_np, kind='linear', fill_value="extrapolate")
            f_dist = interp1d(x_old, dist_np, kind='linear', fill_value="extrapolate")
            
            x_new = np.linspace(0, orig_len - 1, int(orig_len * scale))
            radar_aug = f_radar(x_new)
            dist_aug = f_dist(x_new) * scale 
            dist_aug = np.clip(dist_aug, 0, 1.0)
            
            if len(radar_aug) > self.window_size:
                diff = len(radar_aug) - self.window_size
                start = np.random.randint(0, diff + 1)
                radar_aug = radar_aug[start : start + self.window_size]
                dist_aug = dist_aug[start : start + self.window_size]
            else:
                pad_total = self.window_size - len(radar_aug)
                pad_l = pad_total // 2
                pad_r = pad_total - pad_l
                radar_aug = np.pad(radar_aug, (pad_l, pad_r), 'edge')
                dist_aug = np.pad(dist_aug, (pad_l, pad_r), 'edge')
            
            radar_np = radar_aug
            dist_np = dist_aug
            radar_np = radar_np * np.random.uniform(0.8, 1.2) + np.random.normal(0, 0.02, size=radar_np.shape)

        # === 关键：计算最终标签的 R 峰数量 ===
        # 经过增强和裁剪后，峰的数量可能变化，必须重新算
        peaks, _ = find_peaks(-dist_np, height=-0.1, distance=10)
        peak_count = len(peaks)

        x = torch.from_numpy(radar_np).unsqueeze(0).float()
        y = torch.from_numpy(dist_np).unsqueeze(0).float()
        # 返回数量作为第3个输出 (float 类型以便计算 MSE/SmoothL1 Loss)
        c = torch.tensor([peak_count], dtype=torch.float32) 
        
        return x, y, c