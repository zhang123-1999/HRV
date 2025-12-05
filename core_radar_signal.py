from __future__ import annotations

"""
核心雷达信号处理与 QRS/RR 工具。

合并了原先分散在 QRS.py、method_qrs_basic.py、core_radar_signal.py 的基础方法，
仅保留项目中实际使用的 basic 流程，去除 MRVS/小波/remote sensing 等未使用路径。
"""

from pathlib import Path
from typing import Optional, Tuple
import logging

import numpy as np
from scipy import signal

try:
    import pywt
except ImportError:
    pywt = None

from config_runtime import MAX_RADAR_FRAMES, MAX_RR_COUNT, QUIET_MODE
try:
    from .core_hrv import clean_rr_outliers
except ImportError:
    from core_hrv import clean_rr_outliers

LOGGER = logging.getLogger(__name__)


def _log(msg: str, level: int = logging.INFO) -> None:
    if not QUIET_MODE:
        LOGGER.log(level, msg)


# === 基础工具 ===
def nextpow2(value: int) -> int:
    """Matlab 风格的 nextpow2。"""
    return int(2 ** np.ceil(np.log2(max(value, 1))))


def read_raw_data(
    path: Path,
    num_adc_samples: int,
    num_chirps_per_frame: int,
    num_rx: int,
    fs_adc: float,
    is_real: bool = False,
) -> tuple[np.ndarray, int]:
    """
    读取原始 ADC 数据 -> (frame, chirp, rx, adc)。
    """
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size == 0:
        raise ValueError(f"{path} 文件为空或无法读取。")

    if is_real:
        ints_per_frame = num_adc_samples * num_rx * num_chirps_per_frame
        num_frames = raw.size // ints_per_frame
        if num_frames == 0:
            raise ValueError("数据长度不足，无法组成完整帧 (实数模式)。")
        usable = num_frames * ints_per_frame
        samples = raw[:usable].astype(np.float32)
    else:
        ints_per_frame = 2 * num_adc_samples * num_rx * num_chirps_per_frame
        num_frames = raw.size // ints_per_frame
        if num_frames == 0:
            raise ValueError("数据长度不足，无法组成完整帧 (复数模式)。")
        usable = num_frames * ints_per_frame
        iq_pairs = raw[:usable].reshape(-1, 2).astype(np.float32)
        samples = iq_pairs[:, 0] + 1j * iq_pairs[:, 1]

    radar_cube = samples.reshape(
        num_frames,
        num_chirps_per_frame,
        num_rx,
        num_adc_samples,
    )

    total_frames = radar_cube.shape[0]
    if total_frames > MAX_RADAR_FRAMES:
        radar_cube = radar_cube[:MAX_RADAR_FRAMES]
        num_frames = radar_cube.shape[0]
        _log(f"截断帧数: 原始 {total_frames} -> {num_frames} (上限 {MAX_RADAR_FRAMES})", level=logging.WARNING)
    _log(f"检测到 {num_frames} 帧数据，尺寸: {radar_cube.shape}")
    return radar_cube, num_frames


def apply_tdm_extraction(iq_data: np.ndarray) -> np.ndarray:
    """隔一个 chirp 取一个，保留 TX1。"""
    if iq_data.shape[1] < 2:
        raise ValueError("chirp 数量不足，无法执行 TDM 抽取。")
    usable = (iq_data.shape[1] // 2) * 2
    if usable == 0:
        raise ValueError("无完整 TX1/TX2 组合。")
    trimmed = iq_data[:, :usable, :, :]
    extracted = trimmed[:, ::2, :, :]
    _log(f"TDM 抽取后 chirp 数量: {extracted.shape[1]}")
    return extracted


def beamforming(iq_data: np.ndarray) -> np.ndarray:
    """等权波束形成 -> (frame, chirp, adc)。"""
    num_rx = iq_data.shape[2]
    weights = np.ones(num_rx) / num_rx
    return np.sum(iq_data * weights[np.newaxis, np.newaxis, :, np.newaxis], axis=2)


def preprocess_data(radar_cube: np.ndarray, range_fft_points: int) -> tuple[np.ndarray, np.ndarray]:
    """相位均值对消 + 距离 FFT (返回完整复数谱的一半)。"""
    ref_chirp = np.mean(radar_cube, axis=1, keepdims=True)
    radar_cube_canceled = radar_cube - ref_chirp
    rng_fft = np.fft.fft(radar_cube_canceled, n=range_fft_points, axis=2)
    half = range_fft_points // 2
    return radar_cube_canceled, rng_fft[:, :, :half]


def ca_cfar(signal_line: np.ndarray, n: int, g: int, pfa: float) -> tuple[np.ndarray, np.ndarray]:
    """单元平均 CA-CFAR。"""
    num_cells = len(signal_line)
    detection = np.zeros(num_cells, dtype=int)
    threshold = np.zeros(num_cells)
    total_ref = 2 * n
    alpha = total_ref * (pfa ** (-1 / total_ref) - 1)

    for idx in range(num_cells):
        start = max(0, idx - g - n)
        end = min(num_cells, idx + g + n + 1)
        guard_start = max(0, idx - g)
        guard_end = min(num_cells, idx + g + 1)

        leading = signal_line[start:guard_start]
        trailing = signal_line[guard_end:end]
        ref_cells = np.concatenate([leading, trailing])
        if ref_cells.size == 0:
            continue

        noise_level = np.mean(ref_cells)
        threshold[idx] = alpha * noise_level
        if signal_line[idx] > threshold[idx]:
            detection[idx] = 1

    return detection, threshold


def detect_target(
    range_fft_mag: np.ndarray,
    range_bin_size: float,
    n: int = 16,
    g: int = 2,
    pfa: float = 5e-3,
    target_stability_thresh: float = 0.5,
    ignore_bins: int = 0,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """CFAR + 稳定性判断锁定目标距离单元。"""
    frame_range_energy = np.sum(range_fft_mag, axis=1)  # (frame, range)
    avg_range_energy = np.mean(frame_range_energy, axis=0)
    if ignore_bins:
        avg_range_energy[:ignore_bins] = 0

    detection, threshold = ca_cfar(avg_range_energy, n, g, pfa)
    target_candidates = np.where(detection == 1)[0]
    if target_candidates.size == 0:
        range_energy_var = np.var(frame_range_energy, axis=0)
        valid_mask = avg_range_energy > np.max(avg_range_energy) * 0.1
        if np.any(valid_mask):
            stability = range_energy_var / np.maximum(avg_range_energy, 1e-9)
            stable_candidates = np.where((stability < target_stability_thresh) & valid_mask)[0]
            if stable_candidates.size:
                x_position = stable_candidates[np.argmax(avg_range_energy[stable_candidates])]
            else:
                x_position = int(np.argmax(avg_range_energy))
        else:
            x_position = int(np.argmax(avg_range_energy))
        _log("CFAR 未检测到目标，使用稳定能量最大单元。", level=logging.WARNING)
    else:
        energies = avg_range_energy[target_candidates]
        x_position = int(target_candidates[np.argmax(energies)])

    target_distance = x_position * range_bin_size
    _log(f"锁定的目标距离单元: 索引 {x_position} (距离 {target_distance:.2f} m)")
    return x_position, target_distance, avg_range_energy, threshold


# === basic vital-sign 处理（原 method_qrs_basic） ===
def compute_vital_signs_basic(
    range_fft: np.ndarray,
    x_position: int,
    config: dict,
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 DACM (Differentiate-and-Cross-Multiply) 计算相位微分，
    避免 unwrap 在低信噪比下产生的跳变。返回 (displacement, raw_phase, restored_phase)。
    """
    locked_complex = range_fft[:, :, x_position]
    # 将 chirp 维合并，得到随时间变化的一维复信号
    if locked_complex.ndim > 1:
        complex_sig = np.mean(locked_complex, axis=1)
    else:
        complex_sig = locked_complex

    complex_sig = np.asarray(complex_sig, dtype=np.complex128)
    raw_phase = np.angle(complex_sig)

    # === DACM: d(phi)/dt = (I * dQ - Q * dI) / (I^2 + Q^2) ===
    I = np.real(complex_sig)
    Q = np.imag(complex_sig)
    dI = np.gradient(I)
    dQ = np.gradient(Q)
    denominator = I**2 + Q**2 + 1e-9  # 防止除零
    phase_diff = (I * dQ - Q * dI) / denominator

    # 积分回相位轨迹，避免 unwrap 突变
    restored_phase = np.cumsum(phase_diff)

    nyq = 0.5 * fs
    cutoff = float(config.get("highpass_cutoff", 0.07))
    cutoff = min(max(cutoff, 0.05), nyq * 0.99)
    b_high, a_high = signal.butter(4, cutoff / nyq, btype="high")
    detrended_phase = signal.filtfilt(b_high, a_high, restored_phase)

    displacement = (config["WAVELENGTH_M"] / (4 * np.pi)) * detrended_phase
    vital_signs = displacement
    return vital_signs, raw_phase, restored_phase


def _design_bandpass(low: float, high: float, nyq: float) -> tuple[np.ndarray, np.ndarray]:
    if high <= low:
        high = min(low + 0.1, nyq * 0.95)
    low = max(low, 0.01)
    high = min(high, nyq * 0.95)
    return signal.butter(4, [low / nyq, high / nyq], btype="band")


def filter_vital_signs_basic(
    vital_signs: np.ndarray,
    fs: float,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """固定 Butterworth 带通滤波器提取呼吸/心跳分量。"""
    nyq = 0.5 * fs
    breath_low, breath_high = config["breath_band"]
    b_breath, a_breath = _design_bandpass(breath_low, breath_high, nyq)
    breath_phase = signal.filtfilt(b_breath, a_breath, vital_signs)

    heart_low, heart_high = config["heart_band"]
    heart_low = max(0.5, heart_low)
    heart_high = max(heart_high, heart_low + 0.1)
    b_heart, a_heart = _design_bandpass(heart_low, heart_high, nyq)
    heart_phase = signal.filtfilt(b_heart, a_heart, vital_signs)
    return breath_phase, heart_phase


# === DWT 专用预处理 / 分解 ===
def preprocess_for_dwt(vital_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    供 DWT 使用的预处理：
      - 去均值、去趋势
      - 高通抑制极慢变化
      - 50 Hz 陷波（采样率足够时）
    """
    x = np.asarray(vital_signal, dtype=float)
    if x.size == 0:
        return x

    x = x - np.mean(x)
    x = signal.detrend(x, type="linear")

    nyq = 0.5 * fs
    cutoff = max(0.05, 0.5 / fs)
    cutoff = min(cutoff, nyq * 0.95)
    if cutoff > 0:
        b_high, a_high = signal.butter(4, cutoff / nyq, btype="high")
        x = signal.filtfilt(b_high, a_high, x)

    if fs > 100.0:  # 2 * 50 Hz
        w0 = 50.0 / nyq
        b_notch, a_notch = signal.iirnotch(w0=w0, Q=30.0)
        x = signal.filtfilt(b_notch, a_notch, x)

    return x


def dwt_decompose_vital_signal(
    vital_signal: np.ndarray,
    fs: float,
    wavelet: str = "db5",
    level: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    对预处理后的雷达信号做离散小波分解：
      - Daubechies(db5) 小波，4 层分解
      - 呼吸由 A4 重构，心率由 D3 重构
    返回 (resp_raw, heart_raw, extra_info)。
    """
    if pywt is None:
        raise ImportError("PyWavelets (pywt) is required for DWT-based processing. Please install pywt.")

    x = preprocess_for_dwt(vital_signal, fs=fs)
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="symmetric")
    if len(coeffs) < level + 1:
        raise ValueError(f"DWT decomposition failed: expected {level+1} coefficient sets, got {len(coeffs)}")

    cA4, cD4, cD3, cD2, cD1 = coeffs

    zeros_like = lambda c: np.zeros_like(c)
    coeffs_resp = [cA4, zeros_like(cD4), zeros_like(cD3), zeros_like(cD2), zeros_like(cD1)]
    resp_raw = pywt.waverec(coeffs_resp, wavelet, mode="symmetric")

    coeffs_heart = [zeros_like(cA4), zeros_like(cD4), cD3, zeros_like(cD2), zeros_like(cD1)]
    heart_raw = pywt.waverec(coeffs_heart, wavelet, mode="symmetric")

    n = vital_signal.size
    def _match_length(sig: np.ndarray) -> np.ndarray:
        if sig.size >= n:
            return sig[:n]
        return np.pad(sig, (0, n - sig.size), mode="edge")

    resp_raw = _match_length(resp_raw)
    heart_raw = _match_length(heart_raw)

    extra_info = {
        "fs": float(fs),
        "wavelet": wavelet,
        "level": level,
        "coeff_shapes": [np.asarray(c).shape for c in coeffs],
    }
    nyq = 0.5 * fs
    for lvl in range(1, level + 1):
        f_high = nyq / (2 ** (lvl - 1))
        f_low = nyq / (2**lvl)
        extra_info[f"band_D{lvl}"] = (f_low, f_high)
    extra_info[f"band_A{level}"] = (0.0, nyq / (2**level))
    return resp_raw, heart_raw, extra_info


def akf_smooth_1d(
    signal_in: np.ndarray,
    ref_a4: Optional[np.ndarray] = None,
    ref_d3: Optional[np.ndarray] = None,
    Q: float = 2e-3,
    R_min: float = 5e-3,
    R_max: float = 2e-2,
    P_min: float = 1e-3,
) -> np.ndarray:
    """
    一维自适应 Kalman 滤波器：
      - 状态 = 当前信号值
      - Q: 状态噪声
      - R_k: 测量噪声，随 |A4 - D3| 单调变化
      - P_min: 协方差下界
    如果 ref_a4/ref_d3 为空，退化为常数 R 的 Kalman。
    """
    z = np.asarray(signal_in, dtype=float)
    if z.size == 0:
        return z

    a4 = np.asarray(ref_a4, dtype=float) if ref_a4 is not None else None
    d3 = np.asarray(ref_d3, dtype=float) if ref_d3 is not None else None

    delta = None
    scale = None
    if a4 is not None and d3 is not None and a4.size and d3.size:
        delta = np.abs(np.resize(a4, z.size) - np.resize(d3, z.size))
        scale = float(np.nanpercentile(delta, 95)) if np.any(np.isfinite(delta)) else None
        if scale is None or scale <= 0:
            scale = float(np.nanmax(delta)) if np.any(np.isfinite(delta)) else None

    x_hat = float(z[0])
    P = 1.0
    out = np.zeros_like(z, dtype=float)
    out[0] = x_hat

    for idx in range(1, z.size):
        P = P + Q
        if delta is not None and scale and scale > 0:
            delta_norm = min(1.0, float(delta[idx]) / scale) if np.isfinite(delta[idx]) else 1.0
            R_k = R_min + (R_max - R_min) * delta_norm
        else:
            R_k = R_max

        K = P / (P + R_k)
        x_hat = x_hat + K * (z[idx] - x_hat)
        P = max(P_min, (1.0 - K) * P)
        out[idx] = x_hat

    return out


def extract_breath_heart_dwt(
    vital_signal: np.ndarray,
    fs: float,
    use_akf: bool = True,
    wavelet: str = "db5",
    level: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    综合 DWT + (可选) AKF：
      - dwt_decompose_vital_signal -> resp_raw, heart_raw
      - 可选对 heart_raw 做 AKF 平滑
      - sqrt 压缩减弱大波动
    返回 (resp_processed, heart_processed)
    """
    resp_raw, heart_raw, _info = dwt_decompose_vital_signal(
        vital_signal,
        fs=fs,
        wavelet=wavelet,
        level=level,
    )

    heart_proc = akf_smooth_1d(heart_raw, ref_a4=resp_raw, ref_d3=heart_raw) if use_akf else heart_raw
    resp_proc = resp_raw

    def _sqrt_compress(sig: np.ndarray) -> np.ndarray:
        return np.sign(sig) * np.sqrt(np.abs(sig))

    resp_proc = _sqrt_compress(resp_proc)
    heart_proc = _sqrt_compress(heart_proc)
    return resp_proc, heart_proc


# === QRS / RR 特征（原 QRS.py） ===
def detect_heart_peaks(
    heart_signal: np.ndarray,
    fs: float,
    min_hr: float = 40.0,
    max_hr: float = 180.0,
    prominence_factor: float = 0.3,
) -> np.ndarray:
    """在心跳带通信号上检测 R 峰（雷达版，自适应阈值）。"""
    heart_signal = np.asarray(heart_signal, dtype=float)
    n = heart_signal.size
    if n == 0 or fs <= 0:
        return np.array([], dtype=int)

    min_rr_sec = 60.0 / max_hr
    base_min_distance = max(1, int(fs * min_rr_sec))

    x0 = heart_signal - np.median(heart_signal)
    pos_dyn = np.percentile(x0, 90.0) - np.percentile(x0, 10.0)
    neg_dyn = np.percentile(-x0, 90.0) - np.percentile(-x0, 10.0)
    if neg_dyn > pos_dyn:
        x0 = -x0
    x = np.maximum(x0, 0.0)

    smooth_win = int(0.1 * fs)
    if smooth_win > 1:
        kernel = np.ones(smooth_win, dtype=float) / smooth_win
        x_smooth = np.convolve(x, kernel, mode="same")
    else:
        x_smooth = x

    noise_level = float(np.percentile(x_smooth, 20.0))
    strong_level = float(np.percentile(x_smooth, 90.0))
    if strong_level <= noise_level:
        strong_level = float(x_smooth.max())
    amp = strong_level - noise_level
    if amp <= 0:
        return np.array([], dtype=int)

    def run_with_alpha(alpha: float) -> np.ndarray:
        height_th = noise_level + alpha * amp
        peaks, _ = signal.find_peaks(
            x_smooth,
            height=height_th,
            distance=base_min_distance,
        )
        return peaks

    duration_sec = n / fs
    peaks = run_with_alpha(prominence_factor)
    theo_min_beats = duration_sec * (min_hr / 60.0)
    if peaks.size < 0.6 * theo_min_beats:
        peaks = run_with_alpha(prominence_factor * 0.5)
    return peaks.astype(int)


def reconstruct_qrs_from_peaks(
    length: int,
    peaks: np.ndarray,
    scale_to_std: Optional[float] = None,
) -> np.ndarray:
    """根据峰位置用对称三角波重构 QRS 信号。"""
    qrs = np.zeros(length, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 2:
        for idx in peaks:
            if 0 <= idx < length:
                qrs[idx] = 1.0
        if scale_to_std is not None and qrs.std() > 0:
            qrs *= (scale_to_std / qrs.std())
        return qrs

    for i in range(peaks.size - 1):
        start = peaks[i]
        end = peaks[i + 1]
        if end <= start + 1:
            continue
        center = (start + end) // 2
        left_len = center - start
        right_len = end - center
        if left_len > 0:
            qrs[start:center] = np.linspace(0.0, 1.0, left_len, endpoint=False)
        qrs[center] = 1.0
        if right_len > 1:
            qrs[center + 1:end] = np.linspace(1.0, 0.0, right_len - 1, endpoint=True)

    if scale_to_std is not None:
        std = qrs.std()
        if std > 0:
            qrs *= (scale_to_std / std)
    return qrs


def compute_rr_intervals_from_peaks(
    peaks: np.ndarray,
    fs: float,
    min_rr: float = 0.0,
    max_rr: float = 5.0,
) -> Tuple[np.ndarray, dict]:
    """根据 R 峰索引序列计算 RR 间期（秒）并进行 basic clean。"""
    peaks = np.asarray(peaks, dtype=int)
    rr_count_raw = max(peaks.size - 1, 0)
    info = {
        "rr_count_raw": rr_count_raw,
        "rr_count_clean": 0,
        "dropped": rr_count_raw,
        "min_rr_allowed": float(min_rr),
        "max_rr_allowed": float(max_rr),
    }

    if fs <= 0.0 or peaks.size < 2:
        return np.array([], dtype=float), info

    peaks_sorted = np.sort(peaks.astype(float))
    rr_intervals = np.diff(peaks_sorted) / float(fs)

    valid_mask = (rr_intervals > min_rr) & (rr_intervals <= max_rr)
    rr_clean = rr_intervals[valid_mask]

    info["rr_count_clean"] = int(rr_clean.size)
    info["dropped"] = int(rr_count_raw - rr_clean.size)
    return rr_clean.astype(float), info


def compute_rr_features(rr_intervals: np.ndarray) -> dict:
    """基于清洗后的 RR 间期计算 Sensors 论文中的 5 个统计特征。"""
    rr = np.asarray(rr_intervals, dtype=float)
    rr = rr[np.isfinite(rr)]
    features = {
        "mu_rr": np.nan,
        "norm_max_diff": np.nan,
        "rmssd": np.nan,
        "cv": np.nan,
        "nad": np.nan,
        "rr_count": int(rr.size),
    }

    if rr.size == 0:
        return features

    mu_rr = float(rr.mean())
    if mu_rr <= 0.0:
        return features

    rr_diff = np.diff(rr)
    rmssd = np.sqrt(np.mean(rr_diff ** 2)) if rr_diff.size else 0.0
    norm_max_diff = (rr.max() - rr.min()) / mu_rr
    cv = float(rr.std(ddof=0) / mu_rr)
    nad = float(np.mean(np.abs(rr - mu_rr) / mu_rr))

    features.update(
        {
            "mu_rr": mu_rr,
            "norm_max_diff": float(norm_max_diff),
            "rmssd": float(rmssd),
            "cv": cv,
            "nad": nad,
        }
    )
    return features


# === 3×2 RR 估计方法 ===
def _calc_snr_db(band_mag: np.ndarray) -> float:
    if band_mag.size == 0:
        return float("-inf")
    peak = float(np.max(band_mag))
    if peak <= 0:
        return float("-inf")
    noise_floor = float(np.mean(band_mag)) if band_mag.size else 0.0
    if noise_floor <= 0:
        return float("inf")
    return float(20.0 * np.log10(peak / noise_floor))


def _parabolic_interpolation(freqs: np.ndarray, power: np.ndarray, fallback: float) -> float:
    """简单抛物线插值，使用峰值前后 3 点细化频率。"""
    if freqs.size != 3 or power.size != 3:
        return fallback
    denom = float(power[0] - 2 * power[1] + power[2])
    if denom == 0:
        return fallback
    delta = 0.5 * float(power[0] - power[2]) / denom
    if not np.isfinite(delta) or abs(delta) > 1.0:
        return fallback
    step = float(freqs[1] - freqs[0])
    return float(freqs[1] + delta * step)


def _narrowband_signal_by_peak(
    heart_signal: np.ndarray,
    fs: float,
    f_peak: float,
    hr_band: tuple[float, float],
    delta_f: float = 0.4,
) -> np.ndarray:
    low = max(hr_band[0], f_peak - delta_f)
    high = min(hr_band[1], f_peak + delta_f)
    nyq = 0.5 * fs
    if high <= low:
        spread = max(0.2, delta_f * 0.5)
        low = max(hr_band[0], f_peak - spread)
        high = min(hr_band[1], f_peak + spread)
    b, a = _design_bandpass(low, high, nyq)
    detrended = heart_signal - np.mean(heart_signal)
    return signal.filtfilt(b, a, detrended)


def estimate_rr_fft(
    heart_signal: np.ndarray,
    fs: float,
    method: str = "direct",
    hr_band: tuple[float, float] = (0.8, 3.0),
    win_sec: float = 10.0,
    step_sec: float = 2.0,
    min_snr_db: float = 3.0,
) -> np.ndarray:
    """
    使用 FFT 在心跳信号上估计 RR 间期序列（秒）。
    """
    method_lower = method.lower()
    heart_signal = np.asarray(heart_signal, dtype=float)
    n = heart_signal.size
    if n == 0 or fs <= 0:
        return np.array([], dtype=float)

    if method_lower == "qrs":
        n_fft = nextpow2(n)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        spectrum = np.fft.rfft(heart_signal - np.mean(heart_signal), n=n_fft)
        band_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
        if not np.any(band_mask):
            return np.array([], dtype=float)
        band_mag = np.abs(spectrum)[band_mask]
        if band_mag.size == 0:
            return np.array([], dtype=float)
        f_peak = float(freqs[band_mask][np.argmax(band_mag)])
        if f_peak <= 0:
            return np.array([], dtype=float)

        filtered = _narrowband_signal_by_peak(heart_signal, fs, f_peak, hr_band=hr_band)
        min_rr = 1.0 / max(hr_band[1], 1e-6)
        max_rr = 1.0 / max(hr_band[0], 1e-6)
        peaks = detect_heart_peaks(
            filtered,
            fs=fs,
            min_hr=hr_band[0] * 60.0,
            max_hr=hr_band[1] * 60.0,
        )
        rr_clean, _ = compute_rr_intervals_from_peaks(peaks, fs, min_rr=min_rr, max_rr=max_rr)
        return rr_clean.astype(float)

    if method_lower != "direct":
        raise ValueError(f"Unsupported method '{method}'; expected 'direct' or 'qrs'.")

    win_len = int(win_sec * fs)
    step_len = max(1, int(step_sec * fs))
    if win_len < 4 or win_len > n:
        return np.array([], dtype=float)

    rr_list: list[float] = []
    window = np.hanning(win_len)
    freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)
    band_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
    if not np.any(band_mask):
        return np.array([], dtype=float)

    for start in range(0, n - win_len + 1, step_len):
        seg = heart_signal[start : start + win_len]
        if not np.any(np.isfinite(seg)):
            continue
        seg = seg - np.mean(seg)
        fft_vals = np.fft.rfft(seg * window)
        band_mag = np.abs(fft_vals)[band_mask]
        if band_mag.size == 0:
            continue

        snr_db = _calc_snr_db(band_mag)
        if snr_db < min_snr_db:
            continue
        band_freqs = freqs[band_mask]
        f0 = float(band_freqs[np.argmax(band_mag)])
        if f0 <= 0:
            continue
        rr_list.append(1.0 / f0)

    return np.asarray(rr_list, dtype=float)


def estimate_rr_dwt(
    heart_signal: np.ndarray,
    fs: float,
    method: str = "direct",
    wavelet: str = "db5",
    max_level: int = 4,
    hr_band: tuple[float, float] = (0.8, 3.0),
) -> np.ndarray:
    """
    使用离散小波变换 (DWT) 提取心跳相关分量，再估计 RR 间期。
    内部使用 db5 / 4 层分解，心率基于 D3 子带。
    """
    if pywt is None:
        raise ImportError("PyWavelets (pywt) is required for DWT-based RR estimation. Please install pywt.")

    vital_signal = np.asarray(heart_signal, dtype=float)
    if vital_signal.size == 0 or fs <= 0:
        return np.array([], dtype=float)

    level = min(max_level, 4) if max_level > 0 else 4
    resp_proc, heart_proc = extract_breath_heart_dwt(
        vital_signal,
        fs=fs,
        use_akf=True,
        wavelet=wavelet,
        level=level,
    )
    nyq = 0.5 * fs
    b_heart, a_heart = _design_bandpass(hr_band[0], hr_band[1], nyq)
    if heart_proc.size > max(len(a_heart), len(b_heart)) * 3:
        heart_proc = signal.filtfilt(b_heart, a_heart, heart_proc)

    method_lower = method.lower()
    if method_lower == "direct":
        return estimate_rr_fft(
            heart_proc,
            fs=fs,
            method="direct",
            hr_band=hr_band,
        )
    if method_lower == "qrs":
        peaks = detect_heart_peaks(
            heart_proc,
            fs=fs,
            min_hr=hr_band[0] * 60.0,
            max_hr=hr_band[1] * 60.0,
        )
        rr_clean, _ = compute_rr_intervals_from_peaks(
            peaks,
            fs,
            min_rr=1.0 / max(hr_band[1], 1e-6),
            max_rr=1.0 / max(hr_band[0], 1e-6),
        )
        return rr_clean.astype(float)

    raise ValueError(f"Unsupported method '{method}'; expected 'direct' or 'qrs'.")


def estimate_hr_from_dwt(
    vital_signal: np.ndarray,
    fs: float,
    mode: str = "direct",
    heart_band: tuple[float, float] = (0.8, 3.0),
    wavelet: str = "db5",
    level: int = 4,
) -> tuple[float, np.ndarray]:
    """
    使用 DWT + (可选 AKF) 估计心率：
      - mode='direct': 在心率信号 PSD 上找主峰
      - mode='qrs': 在心率信号上做峰检测 -> RR -> 平均心率
    返回 (hr_bpm, heart_signal_for_debug)
    """
    if pywt is None:
        raise ImportError("PyWavelets (pywt) is required for DWT-based HR estimation. Please install pywt.")

    vital_signal = np.asarray(vital_signal, dtype=float)
    if vital_signal.size == 0 or fs <= 0:
        return np.nan, vital_signal

    resp_proc, heart_proc = extract_breath_heart_dwt(
        vital_signal,
        fs=fs,
        use_akf=True,
        wavelet=wavelet,
        level=level,
    )

    nyq = 0.5 * fs
    b_heart, a_heart = _design_bandpass(heart_band[0], heart_band[1], nyq)
    if heart_proc.size > max(len(a_heart), len(b_heart)) * 3:
        heart_proc = signal.filtfilt(b_heart, a_heart, heart_proc)

    heart_energy = float(np.mean(np.square(heart_proc))) if heart_proc.size else 0.0
    resp_energy = float(np.mean(np.square(resp_proc))) if resp_proc.size else 0.0
    if LOGGER.isEnabledFor(logging.DEBUG) and resp_energy > 0:
        LOGGER.debug("DWT energy ratio (D3/A4): %.3f", heart_energy / resp_energy)

    mode_lower = mode.lower()
    if mode_lower == "direct":
        nperseg = min(256, heart_proc.size)
        if nperseg < 16:
            return np.nan, heart_proc
        freqs, psd = signal.welch(heart_proc, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        mask = (freqs >= heart_band[0]) & (freqs <= heart_band[1])
        if not np.any(mask):
            return np.nan, heart_proc
        band_freqs = freqs[mask]
        band_psd = psd[mask]
        peak_idx = int(np.argmax(band_psd))
        f_peak = float(band_freqs[peak_idx])
        if 0 < peak_idx < band_psd.size - 1:
            f_peak = _parabolic_interpolation(band_freqs[peak_idx - 1 : peak_idx + 2], band_psd[peak_idx - 1 : peak_idx + 2], f_peak)
        hr_bpm = float(f_peak * 60.0) if f_peak > 0 else np.nan
        return hr_bpm, heart_proc

    if mode_lower == "qrs":
        peaks = detect_heart_peaks(
            heart_proc,
            fs=fs,
            min_hr=heart_band[0] * 60.0,
            max_hr=heart_band[1] * 60.0,
        )
        rr_clean, _ = compute_rr_intervals_from_peaks(
            peaks,
            fs,
            min_rr=1.0 / max(heart_band[1], 1e-6),
            max_rr=1.0 / max(heart_band[0], 1e-6),
        )
        hr_bpm = float(60.0 / np.mean(rr_clean)) if rr_clean.size else np.nan
        return hr_bpm, heart_proc

    raise ValueError(f"Unsupported mode '{mode}'; expected 'direct' or 'qrs'.")


def estimate_rr_time_interp(
    heart_signal: np.ndarray,
    fs: float,
    method: str = "direct",
    min_hr: float = 40.0,
    max_hr: float = 180.0,
) -> np.ndarray:
    """
    时域 + 插值前的 RR 估计方法。
    """
    heart_signal = np.asarray(heart_signal, dtype=float)
    n = heart_signal.size
    if n == 0 or fs <= 0:
        return np.array([], dtype=float)

    method_lower = method.lower()
    min_rr = 60.0 / max(max_hr, 1e-6)
    max_rr = 60.0 / max(min_hr, 1e-6)

    if method_lower == "direct":
        distance = max(1, int(fs * min_rr))
        peaks, _ = signal.find_peaks(heart_signal - np.mean(heart_signal), distance=distance)
    elif method_lower == "qrs":
        peaks = detect_heart_peaks(
            heart_signal,
            fs=fs,
            min_hr=min_hr,
            max_hr=max_hr,
        )
    else:
        raise ValueError(f"Unsupported method '{method}'; expected 'direct' or 'qrs'.")

    rr_clean, _ = compute_rr_intervals_from_peaks(peaks, fs, min_rr=min_rr, max_rr=max_rr)
    return rr_clean.astype(float)


# === 高层封装 ===
def extract_rr_from_bin(
    bin_path: Path,
    method: str = "basic",
    num_adc_samples: int = 200,
    num_chirps_per_frame: int = 4,
    num_rx: int = 4,
    fs_adc: float = 4e6,
    slope: float = 80e12,
    frame_periodicity_ms: float = 50.0,
    wavelength_m: float = 0.0039,
    ignore_bins: int = 0,
    tdm_extraction: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    对单个 bin 文件执行完整雷达流程，返回：
        rr_clean_s : RR 序列（秒）
        fs_slow    : 慢时间采样率（Hz）

    仅保留 basic 路径，其它 method 将抛出异常。
    """
    if method != "basic":
        raise ValueError("仅保留 basic 处理路径，其他 method 已移除。")

    radar_cube, num_frames = read_raw_data(
        bin_path,
        num_adc_samples=num_adc_samples,
        num_chirps_per_frame=num_chirps_per_frame,
        num_rx=num_rx,
        fs_adc=fs_adc,
        is_real=False,
    )
    if num_frames == 0:
        raise ValueError(f"{bin_path} 未读取到任何帧。")

    if tdm_extraction:
        radar_cube = apply_tdm_extraction(radar_cube)

    radar_cube = beamforming(radar_cube)
    n_fft = nextpow2(num_adc_samples)
    _radar_cube_canceled, range_fft = preprocess_data(radar_cube, n_fft)
    range_fft_mag = np.abs(range_fft)

    range_bin_size = 3e8 * fs_adc / (2 * slope * n_fft)
    frame_period_s = frame_periodicity_ms / 1000.0
    fs_slow = 1.0 / frame_period_s

    x_position, _target_distance, _avg_energy, _cfar_th = detect_target(
        range_fft_mag,
        range_bin_size=range_bin_size,
        n=16,
        g=2,
        pfa=5e-3,
        target_stability_thresh=0.5,
        ignore_bins=ignore_bins,
    )

    config = {
        "WAVELENGTH_M": wavelength_m,
        "highpass_cutoff": 0.07,
        "breath_band": (0.1, 0.5),
        "heart_band": (0.8, 3.0),
    }

    vital_signs, _locked_phase, _unwrapped_phase = compute_vital_signs_basic(
        range_fft,
        x_position,
        config,
        fs_slow,
    )
    _breath_phase, heart_phase = filter_vital_signs_basic(vital_signs, fs_slow, config)

    peaks = detect_heart_peaks(heart_phase, fs=fs_slow)
    rr_clean, _info = compute_rr_intervals_from_peaks(peaks, fs_slow)
    rr_clean, _ = clean_rr_outliers(rr_clean)

    if rr_clean.size > MAX_RR_COUNT:
        rr_clean = rr_clean[:MAX_RR_COUNT]
    return rr_clean, fs_slow


__all__ = [
    "nextpow2",
    "read_raw_data",
    "apply_tdm_extraction",
    "beamforming",
    "preprocess_data",
    "detect_target",
    "compute_vital_signs_basic",
    "filter_vital_signs_basic",
    "detect_heart_peaks",
    "reconstruct_qrs_from_peaks",
    "compute_rr_intervals_from_peaks",
    "compute_rr_features",
    "estimate_rr_fft",
    "estimate_rr_dwt",
    "estimate_hr_from_dwt",
    "estimate_rr_time_interp",
    "extract_rr_from_bin",
    "preprocess_for_dwt",
    "dwt_decompose_vital_signal",
    "akf_smooth_1d",
    "extract_breath_heart_dwt",
]
