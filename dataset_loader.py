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
    Hybrid Strategy Dataset:
    - TRAIN/VAL: Uses ALL data types (Resting, Tilt, Valsalva) to maximize robustness.
    - TEST: Uses ONLY 'Resting' data to evaluate static performance.
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 1200,
        stride: int = 120,
        mode: str = "train",
        split_ratio: float = 0.8,
        test_subjects: int = 5,  # 预留最后 5 个受试者作为测试集
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

        # 1. 按受试者 ID 分组文件
        # 假设文件名格式: "GDN0001_1_Resting.npz" -> ID 是 "GDN0001"
        all_files = sorted(list(self.data_dir.glob("*.npz")))
        subject_map = {}
        for f in all_files:
            # 提取 GDNxxxx
            sid = f.name.split('_')[0] 
            if sid not in subject_map:
                subject_map[sid] = []
            subject_map[sid].append(f)
        
        all_subjects = sorted(list(subject_map.keys()))
        total_subjs = len(all_subjects)
        
        if total_subjs <= test_subjects:
            raise ValueError(f"Total subjects ({total_subjs}) too few for test split ({test_subjects})")

        # 2. 划分受试者 (Subjects Split)
        test_subj_ids = all_subjects[-test_subjects:]
        train_subj_ids = all_subjects[:-test_subjects]

        # 3. 根据模式选择文件策略
        self.session_files = []

        if self.mode == 'test':
            # === 测试集策略：只选 Resting ===
            for sid in test_subj_ids:
                files = subject_map[sid]
                # 过滤只保留 Resting
                rest_files = [f for f in files if "resting" in f.name.lower()]
                self.session_files.extend(rest_files)
            print(f"[TEST] Selected {len(self.session_files)} 'Resting' sessions from {len(test_subj_ids)} subjects.")

        else:
            # === 训练/验证集策略：使用 所有数据 ===
            train_val_files = []
            for sid in train_subj_ids:
                # 不过滤，包含 Tilt/Valsalva 等所有数据
                train_val_files.extend(subject_map[sid])
            
            # 在文件级别按 8:2 划分 Train/Val
            # 为了随机性，这里可以先由文件名排序，或者简单打乱（这里保持排序以复现）
            split_point = int(len(train_val_files) * split_ratio)
            
            if self.mode == 'train':
                self.session_files = train_val_files[:split_point]
                print(f"[TRAIN] Selected {len(self.session_files)} mixed sessions (Rest+Dynamic).")
            else: # val
                self.session_files = train_val_files[split_point:]
                print(f"[VAL] Selected {len(self.session_files)} mixed sessions (Rest+Dynamic).")

        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        
        # 加载数据
        if len(self.session_files) > 0:
            self._load_samples()
            print(f"[{self.mode.upper()}] Loaded {len(self.samples)} windows.")
        else:
            print(f"[{self.mode.upper()}] Warning: No files found.")

    def _check_quality(self, radar: np.ndarray, dist: np.ndarray) -> bool:
        """
        质量检测
        注意：对于训练集中的 Tilt/Valsalva 数据，心率变异性(RMSSD)本身就会很大。
        所以这里的阈值不能设得太死，否则会把宝贵的动态数据过滤掉。
        """
        if np.std(radar) < 1e-6: return False
        
        peaks, _ = find_peaks(-dist, height=-0.1, distance=10)
        num_peaks = len(peaks)
        if num_peaks < 5 or num_peaks > 35: return False # 允许范围略微放宽到 180+
        
        rr_intervals = np.diff(peaks) / self.fs
        if len(rr_intervals) < 1: return False
        
        # 剔除完全离谱的标签
        diff_rr = np.diff(rr_intervals)
        if len(diff_rr) > 0:
            rmssd = np.sqrt(np.mean(diff_rr**2))
            # 这里的阈值设为 200ms。
            # 正常静息 < 100ms，但在 Valsalva 动作中，RR 可能会剧烈变化，
            # 为了让模型学习这种动态，我们允许更高的 RMSSD，只剔除显然是错误的(>200ms)。
            if rmssd * 1000 > 200: 
                return False
        
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
                        if self._check_quality(radar_slice, dist_slice):
                            self.samples.append((radar_slice, dist_slice))
                                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        radar_np, dist_np = self.samples[idx]
        
        # === 数据增强 (仅训练模式) ===
        if self.mode == 'train':
            # 1. 随机重采样
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

            # 2. 随机幅度 & 噪声
            radar_np = radar_np * np.random.uniform(0.8, 1.2)
            noise = np.random.normal(0, 0.02, size=radar_np.shape)
            radar_np = radar_np + noise

        x = torch.from_numpy(radar_np).unsqueeze(0).float()
        y = torch.from_numpy(dist_np).unsqueeze(0).float()
        return x, y