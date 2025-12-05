import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset_cwt import RadarSpectrogramDataset
from model_transformer import SpectroTransNet

# === 配置区域 ===
DATA_DIR = "results/nature_dataset_cleaned" 
CHECKPOINT = "checkpoints_trans/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_rmssd(bpm_curve, fs=120.0):
    """
    从 BPM 曲线计算 RMSSD (单位: ms)
    """
    # 1. 降采样: 120Hz 对 HRV 计算来说太密了，且插值曲线相邻点差异极小。
    #    降采样到 4Hz (每秒4个点) 比较接近真实的 Tachogram 采样密度。
    target_fs = 4.0
    step = int(fs / target_fs)
    if step < 1: step = 1
    
    bpm_down = bpm_curve[::step]
    
    # 2. 转换为 RR 间期 (ms)
    #    RR = 60000 / BPM
    #    加一个极小值防止除零，虽然 BPM 通常 > 30
    rr_ms = 60000.0 / (bpm_down + 1e-6)
    
    # 3. 计算逐差平方均值根 (RMSSD)
    #    diff = RR_{i+1} - RR_{i}
    diffs = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diffs**2))
    
    return rmssd

def evaluate():
    # 加载测试集 (保持与训练一致)
    test_set = RadarSpectrogramDataset(DATA_DIR, mode='test')
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    model = SpectroTransNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
        print(f"Loaded model from {CHECKPOINT}")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {CHECKPOINT}")
        return

    model.eval()
    
    results = []
    print(f"Evaluating {len(test_set)} samples...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(DEVICE)
            
            # y 是真实的 IHR 曲线 [Batch, 1200]
            y_curve = y.numpy().flatten()
            
            # 模型预测
            pred_curve = model(x).cpu().numpy().flatten()
            
            # === 指标计算 ===
            
            # 1. BPM (平均心率)
            bpm_pred = np.mean(pred_curve)
            bpm_true = np.mean(y_curve)
            
            # 2. SDNN (整体变异性 - BPM单位)
            sdnn_pred = np.std(pred_curve)
            sdnn_true = np.std(y_curve)
            
            # 3. RMSSD (高频变异性 - ms单位) <--- 新增
            rmssd_pred = calculate_rmssd(pred_curve, fs=120.0)
            rmssd_true = calculate_rmssd(y_curve, fs=120.0)
            
            # === 异常值过滤 ===
            if bpm_pred < 30 or bpm_pred > 200 or np.isnan(bpm_pred):
                continue

            # 计算单样本的相关性 (反映趋势拟合程度)
            corr = np.corrcoef(pred_curve, y_curve)[0, 1] if np.std(pred_curve) > 1e-6 else 0

            results.append({
                "Sample": i,
                "True_BPM": bpm_true,
                "Pred_BPM": bpm_pred,
                "Error_BPM": abs(bpm_pred - bpm_true),
                
                "True_SDNN": sdnn_true,
                "Pred_SDNN": sdnn_pred,
                "Error_SDNN": abs(sdnn_pred - sdnn_true),
                
                "True_RMSSD": rmssd_true,   # 新增列
                "Pred_RMSSD": rmssd_pred,   # 新增列
                "Error_RMSSD": abs(rmssd_pred - rmssd_true),
                
                "Correlation": corr
            })
            
    # 统计汇总
    df = pd.DataFrame(results)
    
    if not df.empty:
        print("\n" + "="*40)
        print("   TEST RESULTS SUMMARY (With RMSSD)")
        print("="*40)
        print(f"Valid Samples : {len(df)} / {len(test_set)}")
        print(f"MAE BPM       : {df['Error_BPM'].mean():.4f}")
        print(f"MAE RMSSD     : {df['Error_RMSSD'].mean():.4f} ms") # 重点关注
        print(f"MAE SDNN      : {df['Error_SDNN'].mean():.4f} BPM")
        print(f"Avg Correlation: {df['Correlation'].mean():.4f}")
        
        out_file = "results/transformer_test_results_rmssd.csv"
        df.to_csv(out_file, index=False)
        print(f"\nSaved to {out_file}")
    else:
        print("No valid samples.")

if __name__ == "__main__":
    evaluate()