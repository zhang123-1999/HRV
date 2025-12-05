import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class RadarSpectrogramDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_ratio=0.8):
        self.data_dir = Path(data_dir)
        self.mode = mode
        
        # 1. 直接读取 CSV 索引 (比遍历文件快)
        import pandas as pd
        index_file = self.data_dir / "dataset_index.csv"
        if not index_file.exists():
            raise FileNotFoundError("Index file not found. Run build_nature_dataset.py first.")
            
        df = pd.read_csv(index_file)
        
        # === 过滤掉不存在的文件 (例如刚才手动删除的坏文件) ===
        valid_paths = []
        for p in df['path']:
            if Path(p).exists():
                valid_paths.append(p)
            else:
                print(f"Warning: File not found in index, skipping: {p}")
        
        # 简单划分
        split_idx = int(len(valid_paths) * split_ratio)
        if mode == "train":
            self.file_paths = valid_paths[:split_idx]
        else:
            self.file_paths = valid_paths[split_idx:]
            
        print(f"[{mode}] Loaded {len(self.file_paths)} windows.")

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            with np.load(path) as data:
                x = data['spectrogram']
                y = data['ihr_curve']
                
            # 数据增强 (仅训练)
            if self.mode == 'train':
                # SpecAugment: 随机遮挡频段
                if np.random.rand() < 0.2:
                    f_start = np.random.randint(0, 50)
                    x[f_start:f_start+5, :] = 0
            
            return torch.from_numpy(x).unsqueeze(0).float(), torch.from_numpy(y).float()
            
        except Exception:
            # 遇到坏文件返回 0 (Loader 会报错，但在 Dataset 里最好处理掉)
            return torch.zeros(1, 64, 1200), torch.zeros(1200)

    def __len__(self):
        return len(self.file_paths)