import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp1d
import logging

# 尝试导入您的 HRV 计算模块
try:
    from core_hrv import compute_hrv_metrics
except ImportError:
    # 如果找不到，定义一个简化版作为 fallback
    def compute_hrv_metrics(rr_s):
        rr_ms = rr_s * 1000
        diff_ms = np.diff(rr_ms)
        return {
            "mean_hr_bpm": 60.0 / np.mean(rr_s) if np.mean(rr_s) > 0 else 0,
            "rmssd_ms": np.sqrt(np.mean(diff_ms**2)) if len(diff_ms) > 0 else 0,
            "sdnn_ms": np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else 0
        }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 配置区域 ===
MAT_ROOT_DIR = Path(r"/root/autodl-tmp/data_nature") 
OUT_DIR = Path("results/nature_dataset_120hz")
INDEX_FILE = OUT_DIR / "nature_dataset_index.csv" 

TARGET_FS = 120.0
MAX_DIST_SAMPLES = 30

def _build_dt_labels(peak_indices: np.ndarray, n_samples: int, max_dist_samples: int) -> np.ndarray:
    """生成距离变换标签"""
    dist = np.full(n_samples, float(max_dist_samples), dtype=float)
    if n_samples == 0: return dist.astype(np.float32)
    if peak_indices.size:
        dist[peak_indices] = 0.0
        last = -1_000_000
        for i in range(n_samples):
            if dist[i] == 0.0: last = i
            elif last > -1_000_000: dist[i] = min(dist[i], i - last)
        nxt = 1_000_000
        for i in range(n_samples - 1, -1, -1):
            if dist[i] == 0.0: nxt = i
            elif nxt < 1_000_000: dist[i] = min(dist[i], nxt - i, dist[i])
    dist = np.minimum(dist, float(max_dist_samples))
    return (dist / float(max_dist_samples)).astype(np.float32)

def process_cw_radar_phase(i_sig, q_sig):
    i_centered = i_sig - np.mean(i_sig)
    q_centered = q_sig - np.mean(q_sig)
    return np.unwrap(np.arctan2(q_centered, i_centered))

# === Pan-Tompkins 风格的鲁棒 ECG 峰值检测 ===
def robust_ecg_peak_detection(ecg_raw: np.ndarray, fs: float):
    """
    基于 Pan-Tompkins 思想的鲁棒 R 峰检测
    """
    # 1. 带通滤波 (5-15Hz)
    sos = signal.butter(4, [5, 15], btype='band', fs=fs, output='sos')
    ecg_band = signal.sosfiltfilt(sos, ecg_raw)
    
    # 2. 微分 (Derivative)
    ecg_diff = np.diff(ecg_band, prepend=ecg_band[0])
    
    # 3. 平方 (Squaring)
    ecg_sq = ecg_diff ** 2
    
    # 4. 移动窗口积分 (Window ~150ms)
    window_width = int(0.15 * fs)
    kernel = np.ones(window_width) / window_width
    ecg_integ = np.convolve(ecg_sq, kernel, mode='same')
    
    # 5. 自适应寻峰
    # 修改点：降低阈值系数 0.5 -> 0.3，更容易找到小峰
    min_height = np.mean(ecg_integ) + 0.3 * np.std(ecg_integ) 
    # 修改点：缩短最小间距 300ms -> 250ms，适应高心率
    distance = int(0.25 * fs) 
    
    peaks_idx, _ = signal.find_peaks(ecg_integ, height=min_height, distance=distance)
    
    # [精修] 回溯原始信号找精确最大值点
    refined_peaks = []
    search_win = int(0.1 * fs)
    for p in peaks_idx:
        start = max(0, p - search_win)
        end = min(len(ecg_band), p + search_win)
        if end > start:
            local_peak = start + np.argmax(ecg_band[start:end])
            refined_peaks.append(local_peak)
        
    return np.array(refined_peaks)

def process_mat_file(mat_path):
    try:
        mat = scipy.io.loadmat(mat_path)
        
        if 'radar_i' not in mat or 'tfm_ecg1' not in mat:
            return None

        # 提取数据 (2000Hz)
        raw_i = mat['radar_i'].flatten()
        raw_q = mat['radar_q'].flatten()
        raw_ecg = mat['tfm_ecg1'].flatten()
        fs_raw = float(mat['fs_radar'][0][0])
        n_samples = len(raw_i)
        duration = n_samples / fs_raw

        # 1. 雷达处理
        phase_raw = process_cw_radar_phase(raw_i, raw_q)
        sos = signal.butter(4, [0.8, 3.0], btype='band', fs=fs_raw, output='sos')
        radar_heart_highres = signal.sosfiltfilt(sos, phase_raw)

        # 2. ECG 处理
        peaks_idx = robust_ecg_peak_detection(raw_ecg, fs_raw)
        peaks_time = peaks_idx / fs_raw
        
        # === 修改后的 SQI 质量过滤 ===
        rr_s = np.diff(peaks_time)
        if len(rr_s) < 10: # 至少要有10个间隔才算有效
            logger.warning(f"Skip {mat_path.name}: Too few peaks ({len(rr_s)})")
            return None
            
        # 1. 生理极限检查 (Physiological Limits)
        # RR 间隔应大于 0.3s (心率 < 200 bpm) 且小于 2.0s (心率 > 30 bpm)
        if np.any(rr_s < 0.3) or np.any(rr_s > 2.0):
            logger.warning(f"Skip {mat_path.name}: RR interval out of physiological limits (<0.3s or >2.0s)")
            return None

        # 2. 稳定性检查 (Stability Check)
        # 相邻 RR 间隔变化率应小于 20% (0.2)
        rr_diff_pct = np.abs(np.diff(rr_s)) / rr_s[:-1]
        if np.any(rr_diff_pct > 0.2):
            logger.warning(f"Skip {mat_path.name}: Unstable RR (Change > 20%)")
            return None 

        # === 计算 HRV 真值 ===
        hrv_metrics = compute_hrv_metrics(rr_s)
        true_bpm = hrv_metrics.get('mean_hr_bpm', np.nan)
        true_rmssd = hrv_metrics.get('rmssd_ms', np.nan)
        true_sdnn = hrv_metrics.get('sdnn_ms', np.nan) 
        if np.isnan(true_sdnn): true_sdnn = hrv_metrics.get('sdnn', np.nan)

        # 3. 降采样到 120Hz
        t_raw = np.linspace(0, duration, n_samples)
        t_target = np.arange(0, duration, 1/TARGET_FS)
        
        f_interp = interp1d(t_raw, radar_heart_highres, kind='linear', bounds_error=False, fill_value="extrapolate")
        radar_120 = f_interp(t_target)
        
        # Z-Score
        r_mean = np.mean(radar_120)
        r_std = np.std(radar_120)
        radar_120 = (radar_120 - r_mean) / r_std if r_std > 1e-6 else (radar_120 - r_mean)

        # 4. 生成标签
        peaks_idx_120 = np.round(peaks_time * TARGET_FS).astype(int)
        peaks_idx_120 = peaks_idx_120[(peaks_idx_120 >= 0) & (peaks_idx_120 < len(t_target))]
        dist_norm = _build_dt_labels(peaks_idx_120, len(t_target), MAX_DIST_SAMPLES)

        # 5. 保存 .npz
        session_id = mat_path.stem 
        out_path = OUT_DIR / f"{session_id}.npz"
        np.savez_compressed(
            out_path,
            radar_120=radar_120.astype(np.float32),
            dist_norm=dist_norm.astype(np.float32),
            fs=TARGET_FS
        )
        
        return {
            "session_id": session_id,
            "duration_s": duration,
            "true_bpm": true_bpm,
            "true_rmssd": true_rmssd,
            "true_sdnn": true_sdnn,
            "n_peaks": len(peaks_idx),
            "file_path": str(out_path)
        }

    except Exception as e:
        logger.error(f"Failed to process {mat_path}: {e}")
        return None

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mat_files = list(MAT_ROOT_DIR.rglob("*.mat"))
    print(f"Found {len(mat_files)} MAT files.")
    
    records = []
    count = 0
    for f in mat_files:
        if f.name.startswith("._"): continue
        
        res = process_mat_file(f)
        if res:
            records.append(res)
            count += 1
            if count % 10 == 0: print(f"Processed {count}...")
    
    # 保存索引文件
    if records:
        df = pd.DataFrame(records)
        df.to_csv(INDEX_FILE, index=False)
        print(f"\nSuccess! Index saved to {INDEX_FILE}")
        print(f"Total processed: {count} / {len(mat_files)}")
        print(df[['session_id', 'true_bpm', 'true_rmssd']].head())
    else:
        print("No files processed.")

if __name__ == "__main__":
    main()