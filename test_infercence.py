import torch
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from torch.utils.data import DataLoader
from pathlib import Path

# 导入你的模块
from dataset_loader import RadarECGDataset
from model import IncResUnet
try:
    from core_hrv import compute_hrv_metrics
except ImportError:
    print("Error: core_hrv.py not found. Please ensure it exists.")

# === 配置 ===
DATA_DIR = "results/nature_dataset_120hz"
CHECKPOINT_PATH = "checkpoints/best_rpnet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 120.0  # 采样率

def find_peaks_from_distance_map(dist_map):
    """
    改进版寻峰：物理约束 + 相对高度
    """
    inverted_map = -dist_map 
    
    # 策略: 物理约束 (Refractory Period)
    # distance=30 (250ms) -> 限制最高心率 < 240 BPM
    # prominence=0.05 -> 忽略微小的噪声抖动
    peaks, _ = find_peaks(inverted_map, prominence=0.05, distance=30)
    
    # 兜底策略: 如果找不到峰，尝试降低要求
    if len(peaks) == 0:
        peaks, _ = find_peaks(inverted_map, height=-0.6, distance=30)
        
    return peaks

def run_evaluation():
    # 1. 加载测试集
    # 注意：这里使用 'test_subjects' 参数，因为 dataset_loader 已经更新
    try:
        test_set = RadarECGDataset(
            DATA_DIR, 
            mode='test', 
            test_subjects=5  # 确保这个参数名与 dataset_loader.__init__ 一致
        )
    except ValueError as e:
        print(f"Dataset Error: {e}")
        return
    except TypeError:
        # 兼容旧版参数名 (如果 dataset_loader 没更新)
        try:
            test_set = RadarECGDataset(DATA_DIR, mode='test', test_size=5)
        except Exception as e:
            print(f"Dataset Init Error: {e}")
            return

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print(f"Testing on {len(test_set)} samples...")

    # 2. 加载模型
    model = IncResUnet(in_channels=1, out_channels=1, base_filters=16).to(DEVICE)
    if not Path(CHECKPOINT_PATH).exists():
        print(f"Checkpoint not found at {CHECKPOINT_PATH}")
        return
    
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    results = []
    skipped_count = 0

    # 3. 逐个样本推理
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            # 稳健解包：处理返回 2个 或 3个 元素的情况
            if isinstance(batch_data, list) and len(batch_data) == 3:
                data, target, _ = batch_data # 忽略 count_true
            elif isinstance(batch_data, list) and len(batch_data) == 2:
                data, target = batch_data
            else:
                print(f"Skipping batch {i}: Unexpected data format")
                continue
                
            data = data.to(DEVICE)
            
            # A. 模型预测
            # 处理多输出情况 (dist_map, pred_count)
            output_tuple = model(data)
            if isinstance(output_tuple, tuple):
                output = output_tuple[0] # 取距离图
            else:
                output = output_tuple # 旧模型只返回一个值
            
            # 转为 numpy
            pred_map = output.squeeze().cpu().numpy()
            true_map = target.squeeze().cpu().numpy()

            # B. 后处理：寻峰
            pred_peaks = find_peaks_from_distance_map(pred_map)
            true_peaks = find_peaks_from_distance_map(true_map)

            if len(pred_peaks) < 2 or len(true_peaks) < 2:
                continue

            # C. 计算 RR 间期
            pred_rr_s = np.diff(pred_peaks) / FS
            true_rr_s = np.diff(true_peaks) / FS

            # D. 计算指标
            metrics_pred = compute_hrv_metrics(pred_rr_s)
            metrics_true = compute_hrv_metrics(true_rr_s)

            bpm_pred = metrics_pred.get('mean_hr_bpm', float('nan'))
            bpm_true = metrics_true.get('mean_hr_bpm', float('nan'))
            
            rmssd_pred = metrics_pred.get('rmssd_ms', float('nan'))
            rmssd_true = metrics_true.get('rmssd_ms', float('nan'))

            if np.isnan(bpm_pred) or np.isnan(bpm_true):
                continue

            # === 结果过滤：只统计生理合理的静息心率 ===
            if bpm_pred < 40 or bpm_pred > 130:
                skipped_count += 1
                continue

            results.append({
                'Sample_ID': i,
                'True_BPM': bpm_true,
                'Pred_BPM': bpm_pred,
                'Error_BPM': abs(bpm_pred - bpm_true),
                'True_RMSSD': rmssd_true,
                'Pred_RMSSD': rmssd_pred,
                'Error_RMSSD': abs(rmssd_pred - rmssd_true)
            })

    # 4. 统计结果
    if not results:
        print("No valid results. Check data or model.")
        return

    df = pd.DataFrame(results)
    
    print("\n" + "="*40)
    print("   RESTING STATE - TEST RESULTS")
    print("="*40)
    print(f"Total Valid Samples: {len(df)}")
    print(f"Skipped Outliers: {skipped_count}")
    
    mae_bpm = df['Error_BPM'].mean()
    mae_rmssd = df['Error_RMSSD'].mean()
    
    # 计算中位数误差 (更鲁棒)
    median_bpm = df['Error_BPM'].median()
    median_rmssd = df['Error_RMSSD'].median()
    
    print(f"\nMetric  |  MAE (Mean)  |  Median")
    print(f"--------|--------------|--------")
    print(f"HR (BPM)|  {mae_bpm:.4f}      |  {median_bpm:.4f}")
    print(f"RMSSD   |  {mae_rmssd:.4f} ms |  {median_rmssd:.4f} ms")
    
    # 保存结果
    df.to_csv("results/test_hrv_metrics.csv", index=False)
    print("\nFull results saved to results/test_hrv_metrics.csv")

if __name__ == "__main__":
    run_evaluation()