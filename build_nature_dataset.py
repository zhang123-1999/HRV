import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp1d
import logging

# 必须安装 neurokit2: pip install neurokit2
try:
    import neurokit2 as nk
except ImportError as exc:
    raise ImportError("neurokit2 is required. Please run `pip install neurokit2`.") from exc

# 尝试导入您的 HRV 计算模块 (用于统计 metadata，不影响核心训练数据)
try:
    from core_hrv import compute_hrv_metrics
except ImportError:
    # Fallback
    def compute_hrv_metrics(rr_s):
        if len(rr_s) < 1: return {}
        return {"mean_hr_bpm": 60.0 / np.mean(rr_s)}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 配置区域 ===
# 请修改为您的实际路径
MAT_ROOT_DIR = Path(r"/root/autodl-tmp/data_nature") 
OUT_DIR = Path("results/nature_dataset_spectrogram") # 建议换个新文件夹
INDEX_FILE = OUT_DIR / "dataset_index.csv" 

TARGET_FS = 120.0     # 目标采样率
CWT_SCALES = np.arange(1, 65) # CWT 尺度范围 (约对应 0.5Hz - 6Hz @ 120fps)

def compute_cwt_spectrogram(sig: np.ndarray) -> np.ndarray:
    """
    计算连续小波变换 (CWT) 生成时频图
    Input: [T]
    Output: [Freq, T] (e.g., [64, T])
    """
    # 使用 Ricker 小波 (Mexican Hat)，适合捕捉波峰特征
    cwtmatr = signal.cwt(sig, signal.ricker, CWT_SCALES)
    
    # 取模 (Magnitude)
    cwtmatr = np.abs(cwtmatr)
    
    # === 关键：归一化 ===
    # 神经网络对输入范围敏感，这里做 Z-Score 归一化
    # 也可以选 Min-Max，但 Z-Score 对抗异常值更好
    mean_val = cwtmatr.mean()
    std_val = cwtmatr.std()
    cwtmatr = (cwtmatr - mean_val) / (std_val + 1e-6)
    
    return cwtmatr.astype(np.float32)

def compute_ihr_curve(peak_times_s: np.ndarray, n_samples: int, fs: float) -> np.ndarray:
    """
    生成瞬时心率曲线 (Instantaneous Heart Rate Curve)
    Input: ECG R峰时间点 (秒)
    Output: 与雷达信号等长的连续 BPM 曲线 [T]
    """
    if len(peak_times_s) < 2:
        # 如果没有足够的峰，返回全0 (无效数据)
        return np.zeros(n_samples, dtype=np.float32)
    
    # 1. 计算 RR 间隔
    rr_intervals = np.diff(peak_times_s)
    
    # 2. 计算对应的 BPM 值
    # 限制 RR 范围防止除零或异常值 (30-200 BPM)
    rr_intervals = np.clip(rr_intervals, 0.3, 2.0)
    bpm_values = 60.0 / rr_intervals
    
    # 3. 定义 BPM 值的时间点 (在两个 R 峰中间)
    t_mid = peak_times_s[:-1] + rr_intervals / 2.0
    
    # 4. 构建插值所需的完整时间轴和值
    # 为了防止插值越界，我们在 0时刻和结束时刻填充最近的值
    total_duration = n_samples / fs
    
    t_full = np.concatenate(([0.0], t_mid, [total_duration]))
    bpm_full = np.concatenate(([bpm_values[0]], bpm_values, [bpm_values[-1]]))
    
    # 5. 线性插值生成连续曲线
    t_target = np.arange(n_samples) / fs
    f = interp1d(t_full, bpm_full, kind='linear', fill_value="extrapolate")
    ihr_curve = f(t_target)
    
    return ihr_curve.astype(np.float32)

def process_cw_radar_phase(i_sig, q_sig):
    # 简单的相位提取
    i_centered = i_sig - np.mean(i_sig)
    q_centered = q_sig - np.mean(q_sig)
    # 使用 DACM 或 arctan 均可，这里用 arctan + unwrap 是最基础的
    # 如果您有 DACM 代码，建议在这里替换为 DACM
    phase = np.unwrap(np.arctan2(q_centered, i_centered))
    return phase

def process_mat_file(mat_path):
    try:
        mat = scipy.io.loadmat(mat_path)
        
        # 检查必要的键
        if 'radar_i' not in mat or 'tfm_ecg1' not in mat:
            return None

        # 原始数据提取
        raw_i = mat['radar_i'].flatten()
        raw_q = mat['radar_q'].flatten()
        raw_ecg = mat['tfm_ecg1'].flatten()
        fs_raw = float(mat['fs_radar'][0][0]) # 通常是 2000Hz
        
        # === 1. 雷达信号处理 (降采样到 120Hz) ===
        # 先提取相位
        phase_raw = process_cw_radar_phase(raw_i, raw_q)
        
        # 带通滤波 (提取心跳频段 0.8-3.0Hz)
        sos = signal.butter(4, [0.8, 3.0], btype='band', fs=fs_raw, output='sos')
        radar_heart_highres = signal.sosfiltfilt(sos, phase_raw)
        
        # 降采样: 2000Hz -> 120Hz
        # 计算降采样因子
        q = int(fs_raw / TARGET_FS)
        radar_120 = signal.decimate(radar_heart_highres, q)
        
        # 重新计算实际的 fs (因为整数除法可能有微小偏差)
        actual_fs = fs_raw / q
        n_samples_120 = len(radar_120)
        
        # === 2. 生成 Input: 时频图 (Spectrogram) ===
        # [Freq=64, Time=T]
        spectrogram = compute_cwt_spectrogram(radar_120)

        # === 3. ECG Ground Truth 处理 ===
        # 使用 NeuroKit2 提取精准 R 峰
        ecg_cleaned = nk.ecg_clean(raw_ecg, sampling_rate=fs_raw, method="neurokit")
        _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs_raw)
        peaks_idx = info.get("ECG_R_Peaks", [])
        
        if len(peaks_idx) < 5:
            logger.warning(f"Skip {mat_path.name}: Too few peaks.")
            return None
            
        peaks_time_s = peaks_idx / fs_raw
        
        # === 4. 生成 Target: IHR 曲线 ===
        # 生成与 radar_120 等长的 [T] 数组
        ihr_curve = compute_ihr_curve(peaks_time_s, n_samples_120, actual_fs)

        # === 5. 简单的质量过滤 (可选) ===
        # 如果心率曲线中有极端异常值，标记或跳过
        if np.min(ihr_curve) < 30 or np.max(ihr_curve) > 180:
             logger.warning(f"Skip {mat_path.name}: HR out of range ({np.min(ihr_curve):.1f}-{np.max(ihr_curve):.1f})")
             # return None # 可以选择开启这行来严格过滤

        # === 6. 保存数据 ===
        session_id = mat_path.stem 
        out_path = OUT_DIR / f"{session_id}.npz"
        
        np.savez_compressed(
            out_path,
            spectrogram=spectrogram, # [64, T]
            ihr_curve=ihr_curve,     # [T]
            fs=actual_fs
        )
        
        # 计算一些统计量用于索引 CSV
        metrics = compute_hrv_metrics(np.diff(peaks_time_s))
        
        return {
            "session_id": session_id,
            "duration_s": n_samples_120 / actual_fs,
            "true_bpm": metrics.get('mean_hr_bpm', 0),
            "file_path": str(out_path)
        }

    except Exception as e:
        logger.error(f"Failed to process {mat_path}: {e}")
        return None

def main():
    # 创建输出目录
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    mat_files = list(MAT_ROOT_DIR.rglob("*.mat"))
    print(f"Found {len(mat_files)} MAT files. Processing to {OUT_DIR}...")
    
    records = []
    for i, f in enumerate(mat_files):
        if f.name.startswith("._"): continue
        
        res = process_mat_file(f)
        if res:
            records.append(res)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(mat_files)}...")
    
    # 保存索引
    if records:
        df = pd.DataFrame(records)
        df.to_csv(INDEX_FILE, index=False)
        print(f"\nDone! Index saved to {INDEX_FILE}")
        print(f"Total valid samples: {len(df)}")
    else:
        print("No valid data generated.")

if __name__ == "__main__":
    main()