import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset_cwt import RadarSpectrogramDataset
from model_transformer import SpectroTransNet

DATA_DIR = "results/nature_dataset_cleaned" 
CHECKPOINT = "checkpoints_trans/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_rmssd_from_curve(bpm_curve, fs=480.0): # 注意默认 fs 改为 480
    freq_curve = bpm_curve / 60.0
    phase = np.cumsum(freq_curve) / fs
    beats_idx = np.where(np.diff(np.floor(phase)))[0]
    
    if len(beats_idx) < 2: return np.nan
    
    peak_times_s = beats_idx / fs
    rr_intervals_ms = np.diff(peak_times_s) * 1000.0
    
    valid_rr = rr_intervals_ms[(rr_intervals_ms > 300) & (rr_intervals_ms < 1500)]
    if len(valid_rr) < 2: return np.nan
        
    diffs = np.diff(valid_rr)
    rmssd = np.sqrt(np.mean(diffs**2))
    return rmssd

def evaluate():
    test_set = RadarSpectrogramDataset(DATA_DIR, mode='test', oversample_factor=1)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    model = SpectroTransNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    
    results = []
    print(f"Evaluating {len(test_set)} samples...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(DEVICE)
            
            # Label 处理: 插值到 4800 以匹配模型输出
            # 这样计算 True Metrics 时精度更高
            if y.shape[1] == 1200:
                y = torch.nn.functional.interpolate(y.unsqueeze(1), size=4800, mode='linear', align_corners=True).squeeze(1)
            y_curve = y.numpy().flatten()
            
            # Model Output: [base, final, logvar]
            _, pred_final, pred_logvar = model(x)
            pred_curve = pred_final.cpu().numpy().flatten()
            
            # Uncertainty
            unc = torch.mean(torch.exp(0.5 * pred_logvar)).item()
            
            # Metrics
            bpm_pred = np.mean(pred_curve)
            bpm_true = np.mean(y_curve)
            
            sdnn_pred = np.std(pred_curve)
            sdnn_true = np.std(y_curve)
            
            rmssd_pred = calculate_rmssd_from_curve(pred_curve, fs=480.0)
            rmssd_true = calculate_rmssd_from_curve(y_curve, fs=480.0)
            
            if bpm_pred < 30 or bpm_pred > 200 or np.isnan(bpm_pred): continue

            results.append({
                "Sample": i,
                "True_BPM": bpm_true, "Pred_BPM": bpm_pred, "Error_BPM": abs(bpm_pred - bpm_true),
                "True_RMSSD": rmssd_true, "Pred_RMSSD": rmssd_pred, "Error_RMSSD": abs(rmssd_pred - rmssd_true),
                "True_SDNN": sdnn_true, "Pred_SDNN": sdnn_pred, "Error_SDNN": abs(sdnn_pred - sdnn_true),
                "Uncertainty": unc
            })
            
    df = pd.DataFrame(results)
    if not df.empty:
        print("\n" + "="*30)
        print(f" FINAL RESULTS (Base + Residual)")
        print("="*30)
        print(f"BPM MAE   : {df['Error_BPM'].mean():.4f}")
        print(f"RMSSD MAE : {df['Error_RMSSD'].mean():.4f}")
        print(f"SDNN MAE  : {df['Error_SDNN'].mean():.4f}")
        df.to_csv("results/final_res_residual.csv", index=False)

if __name__ == "__main__":
    evaluate()