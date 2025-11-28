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
MAT_ROOT_DIR = Path(r"/root/autodl-tmp/mmWave/data_nature") 
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

# === 新增：Pan-Tompkins 风格的鲁棒 ECG 峰值检测 ===
def robust_ecg_peak_detection(ecg_raw: np.ndarray, fs: float):
    """
    基于 Pan-Tompkins 思想的鲁棒 R 峰检测
    """
    # 1. 带通滤波 (5-15Hz): 集中 R 峰能量，去除基线和肌电
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
    # 积分后的波形非常平滑，抗噪性强
    min_height = np.mean(ecg_integ) + 0.5 * np.std(ecg_integ)
    distance = int(0.3 * fs) # 不应期 300ms
    
    peaks_idx, _ = signal.find_peaks(ecg_integ, height=min_height, distance=distance)
    
    # [精修] 回溯原始信号找精确最大值点
    # 因为积分会引入相移，需要在积分峰值附近找原始 ecg_band 的最大值
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
        
        # 检查 Key
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

        # 2. ECG 处理 (升级为 Robust Detection)
        # 使用 Pan-Tompkins 算法提取峰值索引
        peaks_idx = robust_ecg_peak_detection(raw_ecg, fs_raw)
        peaks_time = peaks_idx / fs_raw
        
        # === 新增：SQI 质量过滤 ===
        rr_s = np.diff(peaks_time)
        if len(rr_s) < 2:
            logger.warning(f"Skip {mat_path.name}: Too few peaks")
            return None
            
        # 检查 RR 间隔稳定性
        # 如果相邻 RR 变化超过 30%，说明 ECG 信号质量差或有严重心律失常
        rr_diff_pct = np.abs(np.diff(rr_s)) / rr_s[:-1]
        if np.any(rr_diff_pct > 0.3):
            logger.warning(f"Skip {mat_path.name}: Unstable RR (SQI Fail)")
            return None # 直接丢弃该文件，不让坏数据污染训练集

        # === 计算 HRV 真值 ===
        hrv_metrics = compute_hrv_metrics(rr_s)
        
        # 提取关键指标用于索引
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