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

def calculate_rmssd_from_curve(bpm_curve, fs=120.0):
    """
    修正版: 通过积分法 (IPFM) 从连续 BPM 曲线还原离散 R 峰，
    从而计算出物理意义正确的 RMSSD。
    """
    # 1. 将 BPM 转换为 瞬时频率 (Hz = beats / sec)
    #    bpm_curve shape: [1200]
    freq_curve = bpm_curve / 60.0
    
    # 2. 积分 (Cumulative Sum) 计算累积相位 (单位: cycles/beats)
    #    除以 fs 是因为 dt = 1/fs
    #    phase[t] 表示从 0时刻到 t时刻 总共跳了多少下
    phase = np.cumsum(freq_curve) / fs
    
    # 3. 找峰 (Find Peaks)
    #    每当 phase 跨越一个整数 (1.0, 2.0, 3.0...)，说明发生了一次心跳
    #    我们可以检测 floor(phase) 的变化
    beats_idx = np.where(np.diff(np.floor(phase)))[0]
    
    # 4. 如果还原出的心跳太少，无法计算 RMSSD
    if len(beats_idx) < 2:
        return np.nan
        
    # 5. 计算 离散 RR 间隔 (单位: ms)
    #    beats_idx 是索引，除以 fs 得到秒
    peak_times_s = beats_idx / fs
    rr_intervals_ms = np.diff(peak_times_s) * 1000.0
    
    # 6. 计算标准 RMSSD
    #    过滤掉非生理性的 RR (比如 < 300ms 或 > 1500ms) 防止噪声干扰
    valid_rr = rr_intervals_ms[(rr_intervals_ms > 300) & (rr_intervals_ms < 1500)]
    
    if len(valid_rr) < 2:
        return np.nan
        
    diffs = np.diff(valid_rr)
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
            rmssd_pred = calculate_rmssd_from_curve(pred_curve, fs=120.0)
            rmssd_true = calculate_rmssd_from_curve(y_curve, fs=120.0)
            
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