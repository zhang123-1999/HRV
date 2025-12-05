import torch
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from torch.utils.data import DataLoader
from dataset_cwt import RadarSpectrogramDataset
from model_transformer import SpectroTransNet

DATA_DIR = "results/nature_dataset_120hz"
CHECKPOINT = "checkpoints_trans/best_model.pth"
DEVICE = "cuda"

def find_peaks_from_heatmap(heatmap, fs=120.0):
    """基于高斯热图的阈值寻峰。"""
    min_distance = max(1, int(0.3 * fs))
    peaks, _ = find_peaks(heatmap, height=0.4, distance=min_distance)
    return peaks


def heatmap_to_hr_metrics(heatmap, fs):
    peaks = find_peaks_from_heatmap(heatmap, fs)
    if len(peaks) < 2:
        return np.nan, np.nan, peaks
    rr_intervals = np.diff(peaks) / fs
    bpm_series = 60.0 / rr_intervals
    return float(np.mean(bpm_series)), float(np.std(bpm_series)), peaks


def evaluate():
    test_set = RadarSpectrogramDataset(DATA_DIR, mode='test', test_subjects=5)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    model = SpectroTransNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    
    results = []
    
    print(f"Evaluating {len(test_set)} samples...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(DEVICE)
            y_heatmap = y.numpy().flatten()
            
            # Predict
            pred_heatmap = model(x).cpu().numpy().flatten()
            
            bpm_pred, sdnn_pred, pred_peaks = heatmap_to_hr_metrics(pred_heatmap, fs=test_set.target_fs)
            bpm_true, sdnn_true, true_peaks = heatmap_to_hr_metrics(y_heatmap, fs=test_set.target_fs)
            
            if np.isnan(bpm_pred) or np.isnan(bpm_true):
                continue
            
            results.append({
                "Sample": i,
                "True_BPM": bpm_true,
                "Pred_BPM": bpm_pred,
                "Error_BPM": abs(bpm_pred - bpm_true),
                "True_SDNN": sdnn_true,
                "Pred_SDNN": sdnn_pred,
                "Error_SDNN": abs(sdnn_pred - sdnn_true),
                "True_Peaks": len(true_peaks),
                "Pred_Peaks": len(pred_peaks)
            })
            
    df = pd.DataFrame(results)
    if not df.empty:
        print("\nResults Summary:")
        print(f"MAE BPM: {df['Error_BPM'].mean():.4f}")
        print(f"MAE SDNN: {df['Error_SDNN'].mean():.4f}")
        print(f"Mean Pred Peaks: {df['Pred_Peaks'].mean():.2f} | Mean True Peaks: {df['True_Peaks'].mean():.2f}")
        df.to_csv("results/transformer_results.csv", index=False)
    else:
        print("No valid samples for evaluation.")

if __name__ == "__main__":
    evaluate()
