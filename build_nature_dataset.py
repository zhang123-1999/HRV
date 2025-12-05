import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import logging
import warnings
import pywt # 引入 PyWavelets

# 忽略 Neurokit 的部分警告
warnings.filterwarnings("ignore")

try:
    import neurokit2 as nk
except ImportError as exc:
    raise ImportError("neurokit2 is required. Please run `pip install neurokit2`.") from exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 配置区域 ===
MAT_ROOT_DIR = Path(r"/root/autodl-tmp/data_nature") 
OUT_DIR = Path("results/nature_dataset_cleaned") # 换个新目录，隔离脏数据
INDEX_FILE = OUT_DIR / "dataset_index.csv" 

TARGET_FS = 120.0
WINDOW_SIZE = 1200     # 10秒
STRIDE = 120           # 1秒滑动 (数据增强)
CWT_SCALES = np.arange(1, 65) 

def compute_cwt_spectrogram(sig: np.ndarray) -> np.ndarray:
    """计算 CWT 并归一化"""
    # Ricker 小波
    # cwtmatr = signal.cwt(sig, signal.ricker, CWT_SCALES)
    cwtmatr, _ = pywt.cwt(sig, CWT_SCALES, 'mexh')
    cwtmatr = np.abs(cwtmatr)
    # Z-Score 归一化 (防止某些样本能量过大)
    cwtmatr = (cwtmatr - cwtmatr.mean()) / (cwtmatr.std() + 1e-6)
    return cwtmatr.astype(np.float32)

def check_radar_quality(spectrogram: np.ndarray) -> bool:
    """
    雷达信号质量检查 (SQI)
    简单逻辑：检查 0.8-3.0Hz (对应尺度约 10-50) 频段是否有明显的能量聚集
    而不是全屏噪声。
    """
    # 截取心率频段 (大致范围，取决于 CWT scales)
    # 假设 CWT_SCALES 1-64，心率大概在中间区域
    roi = spectrogram[10:50, :] 
    
    # 1. 检查是否有过强的突发噪声 (体动)
    if np.max(roi) > 10.0: # Z-score 后 > 10 说明异常亮
        return False
        
    # 2. 检查对比度 (心跳应该是一条亮线)
    # 如果标准差太小，说明是一片灰，没有信号
    if np.std(roi) < 0.5:
        return False
        
    return True

def get_ihr_in_window(peak_times_s: np.ndarray, window_start_s: float, duration_s: float, fs: float) -> np.ndarray:
    """
    只为当前 10s 窗口生成 IHR 曲线
    关键：如果窗口内峰太少，或者间隔太大，直接返回 None (丢弃该窗口)
    """
    # 找到落在当前窗口内的峰
    valid_peaks = peak_times_s[
        (peak_times_s >= window_start_s - 0.5) & 
        (peak_times_s <= window_start_s + duration_s + 0.5)
    ]
    
    # === 严格过滤 1: 峰数量 ===
    # 10秒内至少要有 5 个峰 (相当于心率 30bpm)
    if len(valid_peaks) < 5:
        return None
        
    # 计算局部 RR
    rr_intervals = np.diff(valid_peaks)
    
    # === 严格过滤 2: 生理合理性 ===
    # 如果有任何 RR > 1.5s (停搏/脱落) 或 RR < 0.3s (室颤/噪点)
    if np.any(rr_intervals > 1.5) or np.any(rr_intervals < 0.3):
        return None
        
    # === 严格过滤 3: 变异性异常 ===
    # 如果相邻 RR 变化太剧烈 (>0.3s)，说明可能有误检
    if np.any(np.abs(np.diff(rr_intervals)) > 0.3):
        return None

    # 插值生成曲线
    bpm_values = 60.0 / rr_intervals
    t_mid = valid_peaks[:-1] + rr_intervals / 2.0
    
    # 相对时间轴 (0 ~ 10s)
    t_mid_relative = t_mid - window_start_s
    
    # 边界填充
    t_full = np.concatenate(([0.0], t_mid_relative, [duration_s]))
    v_full = np.concatenate(([bpm_values[0]], bpm_values, [bpm_values[-1]]))
    
    t_target = np.arange(int(duration_s * fs)) / fs
    f = interp1d(t_full, v_full, kind='linear', fill_value="extrapolate")
    ihr_curve = f(t_target)
    
    return ihr_curve.astype(np.float32)

def process_mat_file(mat_path):
    try:
        mat = scipy.io.loadmat(mat_path)
        if 'radar_i' not in mat or 'tfm_ecg1' not in mat: return []

        # 1. 基础读取
        raw_i = mat['radar_i'].flatten()
        raw_q = mat['radar_q'].flatten()
        raw_ecg = mat['tfm_ecg1'].flatten()
        fs_raw = float(mat['fs_radar'][0][0])
        
        # 2. 全局预处理 (雷达)
        # 简单相位提取 + 滤波
        phase_raw = np.unwrap(np.arctan2(raw_q - np.mean(raw_q), raw_i - np.mean(raw_i)))
        sos = signal.butter(4, [0.8, 3.0], btype='band', fs=fs_raw, output='sos')
        radar_filtered = signal.sosfiltfilt(sos, phase_raw)
        
        # 降采样
        q_factor = int(fs_raw / TARGET_FS)
        radar_120 = signal.decimate(radar_filtered, q_factor)
        actual_fs = fs_raw / q_factor
        
        # 3. 全局预处理 (ECG) -> 获取所有 R 峰
        # NeuroKit 清洗
        ecg_cleaned = nk.ecg_clean(raw_ecg, sampling_rate=fs_raw, method="neurokit")
        _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs_raw)
        peaks_all_s = info["ECG_R_Peaks"] / fs_raw
        
        # 4. === 核心修改：滑动窗口切片 ===
        valid_windows = []
        n_total = len(radar_120)
        
        for start_idx in range(0, n_total - WINDOW_SIZE + 1, STRIDE):
            end_idx = start_idx + WINDOW_SIZE
            
            # 切片
            radar_slice = radar_120[start_idx:end_idx]
            
            # 计算绝对时间 (秒)
            start_time_s = start_idx / actual_fs
            duration_s = WINDOW_SIZE / actual_fs
            
            # A. 生成 ECG 标签 (如果不合格会返回 None)
            ihr_curve = get_ihr_in_window(peaks_all_s, start_time_s, duration_s, actual_fs)
            if ihr_curve is None:
                continue # ECG 质量差，跳过
            
            # B. 生成雷达时频图
            spec = compute_cwt_spectrogram(radar_slice)
            
            # C. 检查雷达质量
            if not check_radar_quality(spec):
                continue # 雷达全是噪声，跳过
                
            # D. 保存合格的窗口
            window_name = f"{mat_path.stem}_w{start_idx}"
            out_path = OUT_DIR / f"{window_name}.npz"
            
            np.savez_compressed(
                out_path,
                spectrogram=spec, # [64, 1200]
                ihr_curve=ihr_curve # [1200]
            )
            
            valid_windows.append({
                "window_id": window_name,
                "mean_bpm": np.mean(ihr_curve),
                "path": str(out_path)
            })
            
        return valid_windows

    except Exception as e:
        logger.error(f"Error {mat_path.name}: {e}")
        return []

def main():
    if OUT_DIR.exists():
        import shutil
        shutil.rmtree(OUT_DIR) # 清空旧数据，防止混淆
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    mat_files = list(MAT_ROOT_DIR.rglob("*.mat"))
    logger.info(f"Start processing {len(mat_files)} files...")
    
    all_records = []
    for i, f in enumerate(mat_files):
        if f.name.startswith("._"): continue

        # Filter: Only use Resting and Valsalva
        if "Resting" not in f.name and "Valsalva" not in f.name:
            continue

        records = process_mat_file(f)
        all_records.extend(records)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(mat_files)}, Valid Windows: {len(all_records)}")
            
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(INDEX_FILE, index=False)
        print(f"Dataset built! Total valid windows: {len(df)}")
        print(f"Saved to {OUT_DIR}")
    else:
        print("No valid data found. Check your filters.")

if __name__ == "__main__":
    main()