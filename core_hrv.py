from __future__ import annotations

"""
核心 HRV 计算函数，供 mmWave / MIT-BIH / 本地 bin 数据共享。

包含：
    - clean_rr_outliers: RR 异常检测与线性插值修复
    - compute_hrv_metrics: 时间域 + Lomb-Scargle 频域 + SampEn
    - build_rr_tachogram: RR(t) 等间隔插值（tachogram）
    - compute_hrv_metrics_interp: 基于插值 RR(t) 的 HRV 计算（当前沿用 Lomb/时间域）
"""

from typing import Dict, Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

HRV_FEATURE_COLS = [
    "mean_rr_ms",
    "mean_hr_bpm",
    "cv",
    "nad",
    "rmssd_ms",
    "sdsd_ms",
    "pnn100",
    "sd1_ms",
    "sd2_ms",
    "sd1_sd2_ratio",
    "hf_power_rel_ls",
    "sampen",
]

RR_MIN_DEFAULT = 0.3
RR_MAX_DEFAULT = 2.0
RR_REL_THRESH_DEFAULT = 0.3


def clean_rr_outliers(
    rr_s: np.ndarray,
    rr_min: float = RR_MIN_DEFAULT,
    rr_max: float = RR_MAX_DEFAULT,
    rel_thresh: float = RR_REL_THRESH_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RR 级别 outlier 清理：
      - 生理范围 [rr_min, rr_max]
      - 与全局/局部中位数偏差 > rel_thresh * 中位数 的视为异常
      - 异常点用索引轴线性插值
    返回 (rr_clean, is_outlier_mask)
    """
    rr_s = np.asarray(rr_s, dtype=float)
    n = rr_s.size
    if n < 3:
        return rr_s, np.zeros_like(rr_s, dtype=bool)

    median_rr = np.median(rr_s)
    is_outlier = (rr_s < rr_min) | (rr_s > rr_max) | (np.abs(rr_s - median_rr) > rel_thresh * median_rr)

    for i in range(1, n - 1):
        local_med = np.median(rr_s[i - 1 : i + 2])
        if np.abs(rr_s[i] - local_med) > rel_thresh * local_med:
            is_outlier[i] = True

    rr_clean = rr_s.copy()
    if np.any(is_outlier):
        good_idx = np.where(~is_outlier)[0]
        bad_idx = np.where(is_outlier)[0]
        if good_idx.size >= 2:
            rr_clean[bad_idx] = np.interp(bad_idx, good_idx, rr_s[good_idx])
        else:
            rr_clean = rr_s.copy()
            is_outlier[:] = False

    return rr_clean, is_outlier


def _sample_entropy(rr_ms: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    N = rr_ms.size
    if N <= m + 1:
        return np.nan
    sd = rr_ms.std(ddof=1)
    if sd == 0:
        return np.nan
    r = r_factor * sd

    def _phi(m_val: int) -> float:
        count = 0
        total = 0
        for i in range(N - m_val):
            template = rr_ms[i : i + m_val]
            dist = np.max(np.abs(rr_ms[i + 1 : N - m_val + 1] - template[:, None]), axis=0)
            count += np.sum(dist <= r)
            total += N - m_val - i - 1
        if total == 0 or count == 0:
            return 0.0
        return count / total

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m == 0 or phi_m1 == 0:
        return np.nan
    return float(-np.log(phi_m1 / phi_m))


def _freq_domain_features_ls(rr_s_in: np.ndarray) -> Tuple[float, float, float]:
    """Lomb–Scargle 频域：返回 (TP, HF, HF_rel)。"""
    if rr_s_in.size < 4:
        return np.nan, np.nan, np.nan

    t = np.cumsum(rr_s_in)
    t = t - t[0]
    rr_detrended = rr_s_in - rr_s_in.mean()

    f = np.linspace(0.04, 0.40, 512)  # Hz
    pxx = signal.lombscargle(t, rr_detrended, 2 * np.pi * f)

    total_power = float(np.trapz(pxx, f))
    hf_mask = (f >= 0.15) & (f <= 0.40)
    if hf_mask.any():
        hf_power = float(np.trapz(pxx[hf_mask], f[hf_mask]))
    else:
        hf_power = np.nan

    if total_power > 0 and not np.isnan(hf_power):
        hf_power_rel = float(hf_power / total_power)
    else:
        hf_power_rel = np.nan

    return total_power, hf_power, hf_power_rel


def _freq_domain_features_ls_uniform(rr_interp_s: np.ndarray, fs_interp: float) -> Tuple[float, float, float]:
    """
    针对已等间隔采样的 RR(t) 序列，用均匀时间轴执行 Lomb–Scargle：
        - 输入: rr_interp_s (秒), fs_interp (Hz)
        - 输出: (total_power, hf_power, hf_power_rel)
    """
    rr = np.asarray(rr_interp_s, dtype=float)
    if rr.size < 4 or fs_interp <= 0:
        return np.nan, np.nan, np.nan

    t = np.arange(rr.size, dtype=float) / float(fs_interp)
    rr_detrended = rr - rr.mean()

    f = np.linspace(0.04, 0.40, 512)  # Hz，与非插值路径保持一致
    pxx = signal.lombscargle(t, rr_detrended, 2 * np.pi * f)

    total_power = float(np.trapz(pxx, f))
    hf_mask = (f >= 0.15) & (f <= 0.40)
    if hf_mask.any():
        hf_power = float(np.trapz(pxx[hf_mask], f[hf_mask]))
    else:
        hf_power = np.nan

    if total_power > 0 and not np.isnan(hf_power):
        hf_power_rel = float(hf_power / total_power)
    else:
        hf_power_rel = np.nan

    return total_power, hf_power, hf_power_rel


def compute_hrv_metrics(rr_s: np.ndarray) -> Dict[str, float]:
    """
    输入 RR 序列（秒），计算时间域 + Lomb-Scargle 频域 + SampEn：
        mean_rr_ms, mean_hr_bpm, sdnn_ms, cv, nad,
        rmssd_ms, sdsd_ms, pnn100,
        sd1_ms, sd2_ms, sd1_sd2_ratio,
        hf_power_rel_ls, sampen
    返回一个 dict。
    """
    rr_s = np.asarray(rr_s, dtype=float) if rr_s is not None else np.array([], dtype=float)
    rr_ms = rr_s * 1000.0
    N = rr_ms.size

    mean_rr_ms = float(rr_ms.mean()) if N > 0 else np.nan
    mean_hr_bpm = 60000.0 / mean_rr_ms if mean_rr_ms > 0 else np.nan

    diff_ms = np.diff(rr_ms) if N >= 2 else np.array([], dtype=float)

    if N > 1:
        sdnn_ms = float(rr_ms.std(ddof=1))
    else:
        sdnn_ms = np.nan

    if mean_rr_ms > 0 and not np.isnan(sdnn_ms):
        cv = float(sdnn_ms / mean_rr_ms)
    else:
        cv = np.nan

    if mean_rr_ms > 0 and N > 0:
        nad = float(np.mean(np.abs(rr_ms - mean_rr_ms)) / mean_rr_ms)
    else:
        nad = np.nan

    if diff_ms.size > 1 and not np.isnan(sdnn_ms):
        var_diff = float(diff_ms.var(ddof=1))
        sd1_ms = np.sqrt(0.5 * var_diff)
        sd2_ms = np.sqrt(max(2.0 * sdnn_ms**2 - 0.5 * var_diff, 0.0))
    else:
        sd1_ms = np.nan
        sd2_ms = np.nan

    if diff_ms.size > 0:
        rmssd_ms = float(np.sqrt(np.mean(diff_ms**2)))
    else:
        rmssd_ms = np.nan

    if diff_ms.size > 1:
        sdsd_ms = float(diff_ms.std(ddof=1))
    else:
        sdsd_ms = np.nan

    if diff_ms.size > 0:
        pnn100 = float(100.0 * np.mean(np.abs(diff_ms) > 100.0))
    else:
        pnn100 = np.nan

    if not np.isnan(sd1_ms) and not np.isnan(sd2_ms) and sd2_ms > 0:
        sd1_sd2_ratio = float(sd1_ms / sd2_ms)
    else:
        sd1_sd2_ratio = np.nan

    _total_power_ls, _hf_power_ls, hf_power_rel_ls = _freq_domain_features_ls(rr_s)
    sampen = _sample_entropy(rr_ms, m=2, r_factor=0.2)

    metrics: Dict[str, float] = {
        "mean_rr_ms": mean_rr_ms,
        "mean_hr_bpm": mean_hr_bpm,
        "cv": cv,
        "nad": nad,
        "rmssd_ms": rmssd_ms,
        "sdsd_ms": sdsd_ms,
        "pnn100": pnn100,
        "sd1_ms": sd1_ms,
        "sd1_sd2_ratio": sd1_sd2_ratio,
        "hf_power_rel_ls": hf_power_rel_ls,
        "sampen": sampen,
    }
    metrics["sd2_ms"] = sd2_ms  # 保留 sd2 方便调试/扩展
    return metrics


def build_rr_tachogram(peak_times_s: np.ndarray, fs_interp: float = 4.0, kind: str = "cubic") -> tuple[np.ndarray, np.ndarray]:
    """
    给定 R 峰时间戳（秒），构造等间隔 RR(t) 心率变异序列：
        - 在 [t0, tN] 上以 fs_interp 插值
        - 返回 (t_interp, rr_interp_s)
    """
    peak_times_s = np.asarray(peak_times_s, dtype=float)
    if peak_times_s.size < 3:
        return np.array([]), np.array([])

    rr_s = np.diff(peak_times_s)
    t_mid = peak_times_s[:-1] + rr_s / 2.0

    dt = 1.0 / fs_interp
    t_uniform = np.arange(t_mid[0], t_mid[-1], dt)

    interp_kind = kind
    if np.unique(t_mid).size < 4 and kind == "cubic":
        interp_kind = "linear"

    f = interp1d(t_mid, rr_s, kind=interp_kind, fill_value="extrapolate")
    rr_interp_s = f(t_uniform)

    return t_uniform, rr_interp_s


def compute_hrv_metrics_interp(rr_interp_s: np.ndarray, fs_interp: float) -> Dict[str, float]:
    """
    基于插值后的 RR(t)（等间隔采样）计算 HRV：
        - 时间域 + SampEn 复用 compute_hrv_metrics
        - HF 相对功率基于均匀时间轴的 Lomb-Scargle 重新计算
    """
    rr_interp_s = np.asarray(rr_interp_s, dtype=float)
    if rr_interp_s.size == 0 or fs_interp <= 0:
        return compute_hrv_metrics(rr_interp_s)

    metrics = compute_hrv_metrics(rr_interp_s)
    _tp, _hf, hf_rel_uniform = _freq_domain_features_ls_uniform(rr_interp_s, fs_interp)
    metrics["hf_power_rel_ls"] = hf_rel_uniform
    return metrics
