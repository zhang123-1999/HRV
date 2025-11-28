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

def find_peaks_from_distance_map(dist_map, threshold=0.3):
    """
    从距离图中提取 R 峰位置。
    距离图原理：峰值处为 0，向两侧递增。
    策略：取负号变成波峰，然后用 find_peaks。
    """
    # 1. 反转信号：因为我们要找谷底 (0)，find_peaks 是找山峰
    # 原始 dist_map 范围大概是 [0, 1]
    inverted_map = -dist_map 
    
    # 2. 设置高度阈值：
    # 距离 R 峰越近值越小(接近0)，反转后接近 0 (比如 -0.05)。
    # 设定一个 height 阈值，过滤掉那些预测值很大（说明离 R 峰很远）的噪声
    # 注意：数据归一化过，最大距离是 1.0 (对应30个点)。
    # 阈值设为 -0.5 意味着只看距离真值峰值 15 个点以内的区域
    peaks, _ = find_peaks(inverted_map, height=-threshold, distance=30) # distance=30(0.25s) 防止过近
    return peaks

def run_evaluation():
    # 1. 加载测试集 (最后 20 个 session)
    try:
        test_set = RadarECGDataset(DATA_DIR, mode='test', test_size=20)
    except ValueError as e:
        print(f"Dataset Error: {e}")
        return

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print(f"Testing on {len(test_set)} samples...")

    # 2. 加载模型
    model = IncResUnet(in_channels=1, out_channels=1).to(DEVICE)
    if not Path(CHECKPOINT_PATH).exists():
        print(f"Checkpoint not found at {CHECKPOINT_PATH}")
        return
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    results = []

    # 3. 逐个样本推理
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(DEVICE)
            # data: [1, 1, 1200], target: [1, 1, 1200]
            
            # A. 模型预测
            output = model(data)
            
            # 转为 numpy (去除 Batch 和 Channel 维度)
            pred_map = output.squeeze().cpu().numpy()
            true_map = target.squeeze().cpu().numpy() # Label 也是距离图

            # B. 后处理：从距离图提取峰值索引
            pred_peaks = find_peaks_from_distance_map(pred_map)
            true_peaks = find_peaks_from_distance_map(true_map)

            # C. 计算 RR 间期 (秒)
            if len(pred_peaks) < 2 or len(true_peaks) < 2:
                # 峰值太少，无法计算 HRV
                continue

            pred_rr_s = np.diff(pred_peaks) / FS
            true_rr_s = np.diff(true_peaks) / FS

            # D. 计算 HRV 指标 (使用 core_hrv.py)
            metrics_pred = compute_hrv_metrics(pred_rr_s)
            metrics_true = compute_hrv_metrics(true_rr_s)

            # E. 记录误差
            # 我们主要关注心率 (BPM) 和 RMSSD (毫秒)
            bpm_pred = metrics_pred.get('mean_hr_bpm', float('nan'))
            bpm_true = metrics_true.get('mean_hr_bpm', float('nan'))
            
            rmssd_pred = metrics_pred.get('rmssd_ms', float('nan'))
            rmssd_true = metrics_true.get('rmssd_ms', float('nan'))

            if np.isnan(bpm_pred) or np.isnan(bpm_true):
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
        print("No valid HRV metrics calculated. Check model output or peak detection threshold.")
        return

    df = pd.DataFrame(results)
    
    print("\n" + "="*40)
    print("   EXTENSIVE DATASET - TEST RESULTS")
    print("="*40)
    print(f"Total Samples Evaluated: {len(df)}")
    
    # 计算平均绝对误差 (MAE)
    mae_bpm = df['Error_BPM'].mean()
    mae_rmssd = df['Error_RMSSD'].mean()
    
    # 打印统计
    print(f"\nMetric  |  MAE (Mean Abs Error)")
    print(f"--------|----------------------")
    print(f"HR (BPM)|  {mae_bpm:.4f}")
    print(f"RMSSD   |  {mae_rmssd:.4f} ms")
    
    print("\nDetailed Sample View (First 5):")
    print(df[['Sample_ID', 'True_BPM', 'Pred_BPM', 'True_RMSSD', 'Pred_RMSSD']].head(5).to_string(index=False))

    # 保存结果
    df.to_csv("results/test_hrv_metrics.csv", index=False)
    print("\nFull results saved to results/test_hrv_metrics.csv")

if __name__ == "__main__":
    run_evaluation()