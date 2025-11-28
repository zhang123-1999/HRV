from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

from config_runtime import MAX_ECG_DURATION_SEC, MAX_RADAR_FRAMES, MAX_RR_COUNT, QUIET_MODE
from helper_fns import get_bandpass_filter, loadTestDataFromDataset
try:
    from core_hrv import (
        HRV_FEATURE_COLS,
        build_rr_tachogram,
        clean_rr_outliers,
        compute_hrv_metrics,
        compute_hrv_metrics_interp,
    )
    from core_radar_signal import (
        compute_vital_signs_basic,
        detect_heart_peaks,
        detect_target,
        estimate_hr_from_dwt,
        estimate_rr_fft,
        estimate_rr_time_interp,
        filter_vital_signs_basic,
    )
except ImportError:
    from core_hrv import (
        HRV_FEATURE_COLS,
        build_rr_tachogram,
        clean_rr_outliers,
        compute_hrv_metrics,
        compute_hrv_metrics_interp,
    )
    from core_radar_signal import (
        compute_vital_signs_basic,
        detect_heart_peaks,
        detect_target,
        estimate_hr_from_dwt,
        estimate_rr_fft,
        estimate_rr_time_interp,
        filter_vital_signs_basic,
    )

LOGGER = logging.getLogger(__name__)

ECG_SAMPLING_RATE = 256.0  # Hz
RADAR_HR_BAND = (0.8, 3.0)  # Hz
RADAR_MAX_HR_FREQ = 3.0  # Hz
FS_INTERP_DEFAULT = 4.0  # Hz for RR tachogram interpolation
RADAR_WAVEFORM_UP_FACTOR = 4  # 雷达心跳波形插值倍数 (fs_interp = up_factor * fs_radar)

RR_MIN_S = 0.3
RR_MAX_S = 2.0
RR_REL_THRESH = 0.3

RADAR_CONFIG_BASE = {
    "FRAME_PERIOD_S": None,
    "RANGE_RESOLUTION_M": 0.1,
    "breath_band": (0.1, 0.5),
    "heart_band": (0.8, 3.0),
    "highpass_cutoff": 0.07,
    "WAVELENGTH_M": 0.0039,
}


def _abbr_posture(name: str) -> str:
    name_lower = name.lower()
    if name_lower.startswith("ly"):
        return "ly"
    if name_lower.startswith("sit"):
        return "sit"
    return name_lower.replace("-", "_")


def _abbr_state(name: str) -> str:
    name_lower = name.lower()
    if name_lower.startswith("rest"):
        return "rest"
    if name_lower.startswith("post"):
        return "post"
    return name_lower.replace("-", "_")


def _estimate_fs_from_timestamps(times: pd.Series | np.ndarray) -> float:
    ts = pd.to_datetime(times)
    dt = ts.diff().dropna().dt.total_seconds()
    med = dt.median()
    if med <= 0:
        raise ValueError("Non-positive median delta encountered.")
    return 1.0 / med


def _bandpass_filter(signal_in: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    b, a = get_bandpass_filter(fs, low, high, order)
    return signal.filtfilt(b, a, signal_in)


def _infer_range_bin_size(radar_data: Dict) -> float:
    if radar_data is None:
        return RADAR_CONFIG_BASE["RANGE_RESOLUTION_M"]
    if "rangeBinSize" in radar_data:
        try:
            val = float(radar_data["rangeBinSize"])
            if val > 0:
                return val
        except Exception:
            pass

    r_bins = radar_data.get("rBins")
    if r_bins is not None:
        try:
            arr = np.asarray(r_bins, dtype=float).reshape(-1)
            if arr.size > 1:
                diffs = np.diff(arr)
                pos = diffs[np.isfinite(diffs) & (diffs > 0)]
                if pos.size:
                    return float(np.median(pos))
        except Exception:
            pass

    return RADAR_CONFIG_BASE["RANGE_RESOLUTION_M"]


def _load_demographics(info_path: Path) -> Dict[str, Dict[str, float]]:
    if not info_path.exists():
        return {}
    df = pd.read_excel(info_path)
    demo_map: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        pid = str(row.get("PARTICIPANT ID", "")).strip()
        if not pid:
            continue
        gender_raw = str(row.get("GENDER", "")).lower()
        if gender_raw.startswith("m"):
            gender = 1.0
        elif gender_raw.startswith("f"):
            gender = 0.0
        else:
            try:
                gender = float(row.get("GENDER"))
            except Exception:
                gender = np.nan
        age = float(row.get("AGE", np.nan)) if pd.notna(row.get("AGE", np.nan)) else np.nan
        weight_kg = float(row.get("WEIGHT (kg)", np.nan)) if pd.notna(row.get("WEIGHT (kg)", np.nan)) else np.nan
        height_cm = float(row.get("HEIGHT (cm)", np.nan)) if pd.notna(row.get("HEIGHT (cm)", np.nan)) else np.nan
        healthy_raw = str(row.get("HEALTHY", "")).lower()
        if healthy_raw in {"yes", "healthy", "true", "1"}:
            label = 0
        elif healthy_raw in {"no", "false", "0"}:
            label = 1
        else:
            label = np.nan
        demo_map[pid] = {
            "gender": gender,
            "age": age,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "label": label,
        }
    return demo_map


def _get_demo(participant: str, demo_map: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    base = {
        "gender": np.nan,
        "age": np.nan,
        "weight_kg": np.nan,
        "height_cm": np.nan,
        "label": 0,  # 默认标签为 0，非 MIT 数据集不做异常标注
    }
    meta = demo_map.get(participant, {})
    base.update({k: v for k, v in meta.items() if pd.notna(v) and k != "label"})
    return base


def _iter_sessions(data_root: Path) -> Iterable[tuple[Path, str, str, str]]:
    participants = sorted([p for p in data_root.glob("P*") if p.is_dir()])
    for p_dir in participants:
        participant = p_dir.name
        for posture_dir in sorted([d for d in p_dir.iterdir() if d.is_dir()]):
            posture = posture_dir.name
            for state_dir in sorted([d for d in posture_dir.iterdir() if d.is_dir()]):
                state = state_dir.name
                yield state_dir, participant, posture, state


def _hrv_from_rr(rr_clean_s: np.ndarray, rr_mode: str, fs_interp: float = FS_INTERP_DEFAULT) -> tuple[Dict[str, float] | None, Dict[str, float] | None]:
    rr_clean_s = np.asarray(rr_clean_s, dtype=float)
    hrv_raw = hrv_interp = None
    if rr_mode in ("raw", "both") and rr_clean_s.size >= 2:
        hrv_raw = compute_hrv_metrics(rr_clean_s)
    if rr_mode in ("interp", "both") and rr_clean_s.size >= 2:
        peak_times = np.cumsum(rr_clean_s)
        peak_times = peak_times - peak_times[0]
        _, rr_interp_s = build_rr_tachogram(peak_times, fs_interp=fs_interp)
        hrv_interp = compute_hrv_metrics_interp(rr_interp_s, fs_interp=fs_interp)
    return hrv_raw, hrv_interp


def _detect_rr_waveform_interp(
    heart_signal: np.ndarray,
    fs: float,
    up_factor: int = RADAR_WAVEFORM_UP_FACTOR,
    min_hr: float = 40.0,
    max_hr: float = 180.0,
) -> np.ndarray:
    """
    对已经带通出的雷达心跳波形进行「时间轴插值 + QRS 检测」，返回 RR 间期 (秒)。
    """
    heart_signal = np.asarray(heart_signal, dtype=float)
    n = heart_signal.size
    if n < 4 or fs <= 0.0:
        return np.array([], dtype=float)

    # 原始时间轴
    t = np.arange(n, dtype=float) / float(fs)

    fs_interp = float(fs) * float(up_factor)
    if fs_interp <= fs:
        fs_interp = fs
    dt_interp = 1.0 / fs_interp
    t_interp = np.arange(t[0], t[-1] + 1e-9, dt_interp)
    if t_interp.size < 4:
        return np.array([], dtype=float)

    heart_interp = np.interp(t_interp, t, heart_signal)

    peaks = detect_heart_peaks(
        heart_interp,
        fs=fs_interp,
        min_hr=min_hr,
        max_hr=max_hr,
    )
    if peaks.size < 2:
        return np.array([], dtype=float)

    peaks = peaks.astype(float)
    rr_s = np.diff(peaks) / fs_interp
    return rr_s


def _process_radar(radar_data: Dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Extensive 雷达预处理：CFAR 锁定目标 -> 相位解缠/高通 -> 位移 -> 呼吸/心跳带通 -> RR。
    返回 (vital_signs, heart_sig, rr_clean_s, fs_radar)。
    """
    if radar_data is None:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), 0.0

    rffts = radar_data.get("rFFTs", [])
    if len(rffts) == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), 0.0
    if len(rffts) > MAX_RADAR_FRAMES:
        radar_data = dict(radar_data)
        radar_data["rFFTs"] = radar_data["rFFTs"][:MAX_RADAR_FRAMES]
        rffts = radar_data["rFFTs"]

    periodicity_ms = float(radar_data.get("chirpConfig", {}).get("PERIODICITY", 0.0))
    if periodicity_ms <= 0 and radar_data.get("frame_period_s"):
        periodicity_ms = float(radar_data["frame_period_s"]) * 1e3
    if periodicity_ms <= 0:
        raise ValueError("Invalid radar PERIODICITY for Extensive dataset.")
    frame_period_s = periodicity_ms * 1e-3
    fs_radar = 1.0 / frame_period_s

    range_bin_size = _infer_range_bin_size(radar_data)
    cfg = dict(RADAR_CONFIG_BASE)
    cfg.update(
        {
            "FRAME_PERIOD_S": frame_period_s,
            "RANGE_RESOLUTION_M": range_bin_size,
        }
    )

    range_fft = np.asarray(rffts)
    if range_fft.ndim == 2:  # 单帧
        range_fft = range_fft[None, ...]
    if range_fft.ndim != 3:
        raise ValueError(f"Unexpected rFFTs shape: {range_fft.shape}")

    range_fft_mag = np.abs(range_fft)
    x_position, _target_distance, _avg_energy, _cfar_th = detect_target(
        range_fft_mag,
        range_bin_size=cfg["RANGE_RESOLUTION_M"],
        n=16,
        g=2,
        pfa=5e-3,
        target_stability_thresh=0.5,
        ignore_bins=0,
    )

    vital_signs, _locked_phase, _unwrapped_phase = compute_vital_signs_basic(
        range_fft,
        x_position,
        cfg,
        fs_radar,
    )
    breath_sig, heart_sig = filter_vital_signs_basic(vital_signs, fs_radar, cfg)

    rr_s = _detect_rr_waveform_interp(
        heart_signal=heart_sig,
        fs=fs_radar,
        up_factor=RADAR_WAVEFORM_UP_FACTOR,
        min_hr=40.0,
        max_hr=180.0,
    )
    rr_clean_s, _ = clean_rr_outliers(rr_s, rr_min=RR_MIN_S, rr_max=RR_MAX_S, rel_thresh=RR_REL_THRESH)
    if rr_clean_s.size > MAX_RR_COUNT:
        rr_clean_s = rr_clean_s[:MAX_RR_COUNT]

    return vital_signs, heart_sig, rr_clean_s, fs_radar


def extract_radar_heart_signal_with_ts(radar_data: Dict) -> tuple[pd.Series, np.ndarray, float]:
    """
    提取 Extensive 雷达心跳波形及时间戳（不做 RR/HRV 计算）。
    返回 (timestamps_mmwave, heart_signal, fs_radar)。
    """
    if not radar_data or "rFFTs" not in radar_data or len(radar_data["rFFTs"]) == 0:
        return pd.Series([], dtype="datetime64[ns]"), np.array([], dtype=float), 0.0

    rffts = radar_data["rFFTs"]
    if len(rffts) > MAX_RADAR_FRAMES:
        rffts = rffts[:MAX_RADAR_FRAMES]

    ts_mmwave = pd.Series(pd.to_datetime(radar_data.get("timestamps_mmwave", [])))
    if ts_mmwave.empty:
        return pd.Series([], dtype="datetime64[ns]"), np.array([], dtype=float), 0.0
    if ts_mmwave.size > len(rffts):
        ts_mmwave = ts_mmwave.iloc[: len(rffts)]
    elif ts_mmwave.size < len(rffts):
        rffts = rffts[: ts_mmwave.size]

    periodicity_ms = float(radar_data.get("chirpConfig", {}).get("PERIODICITY", 0.0))
    if periodicity_ms <= 0 and radar_data.get("frame_period_s"):
        periodicity_ms = float(radar_data["frame_period_s"]) * 1e3
    if periodicity_ms <= 0:
        return pd.Series([], dtype="datetime64[ns]"), np.array([], dtype=float), 0.0
    frame_period_s = periodicity_ms * 1e-3
    fs_radar = 1.0 / frame_period_s

    cfg = dict(RADAR_CONFIG_BASE)
    cfg.update(
        {
            "FRAME_PERIOD_S": frame_period_s,
            "RANGE_RESOLUTION_M": _infer_range_bin_size(radar_data),
        }
    )

    range_fft = np.asarray(rffts)
    if range_fft.ndim == 2:  # 单帧
        range_fft = range_fft[None, ...]
    if range_fft.ndim != 3:
        return pd.Series([], dtype="datetime64[ns]"), np.array([], dtype=float), 0.0

    range_fft_mag = np.abs(range_fft)
    x_position, _target_distance, _avg_energy, _cfar_th = detect_target(
        range_fft_mag,
        range_bin_size=cfg["RANGE_RESOLUTION_M"],
        n=16,
        g=2,
        pfa=5e-3,
        target_stability_thresh=0.5,
        ignore_bins=0,
    )

    vital_signs, _locked_phase, _unwrapped_phase = compute_vital_signs_basic(
        range_fft,
        x_position,
        cfg,
        fs_radar,
    )
    _breath_sig, heart_sig = filter_vital_signs_basic(vital_signs, fs_radar, cfg)

    return ts_mmwave.reset_index(drop=True), heart_sig.astype(float), float(fs_radar)


def process_extensive_radar_legacy(
    radar_data: Dict,
    rr_mode: str = "both",
    fs_interp: float = FS_INTERP_DEFAULT,
) -> tuple[np.ndarray, Dict[str, float] | None, Dict[str, float] | None, float]:
    """
    旧版 Extensive 雷达 RR/HRV 流程（raw HRV + RR 插值 HRV）。

    参数
    ----
    radar_data : Dict
        helper_fns.loadTestDataFromDataset() 返回的 radar_data 字典。
    rr_mode : {"raw", "interp", "both"}
        控制返回的 HRV 指标种类。
    fs_interp : float
        RR 序列插值的采样率，默认 4 Hz。

    返回
    ----
    rr_clean_s : np.ndarray
        清洗后的 RR 间期序列（秒）。
    hrv_raw : dict 或 None
        非插值 HRV 特征（若 rr_mode 不包含 "raw" 或 RR 太短，则为 None）。
    hrv_interp : dict 或 None
        插值 HRV 特征（若 rr_mode 不包含 "interp" 或 RR 太短，则为 None）。
    fs_radar : float
        雷达帧率（Hz）。
    """
    if radar_data and len(radar_data.get("rFFTs", [])) > MAX_RADAR_FRAMES:
        radar_data = dict(radar_data)
        radar_data["rFFTs"] = radar_data["rFFTs"][:MAX_RADAR_FRAMES]

    fs_radar = 1.0 / (radar_data["chirpConfig"]["PERIODICITY"] * 1e-3)
    rFFTs_mean = np.abs(np.mean(radar_data["rFFTs"], axis=(0, 1)))
    r_bin = int(np.argmax(rFFTs_mean))

    raw_phase = []
    for rFFT in radar_data["rFFTs"]:
        rFFT_t = np.mean(rFFT, axis=0)
        comp = rFFT_t[r_bin]
        phase = np.arctan2(comp.imag, comp.real)
        raw_phase.append(phase)

    unwrapped = np.unwrap(raw_phase)
    vital_signs_signal = unwrapped - np.mean(unwrapped)

    filtered = _bandpass_filter(vital_signs_signal, fs_radar, *RADAR_HR_BAND, order=4)

    distance = int(fs_radar * 0.5)  # 至少 0.5 s 间隔，对应 HR <= 120 bpm
    peaks, _ = signal.find_peaks(filtered, height=0.0, distance=distance)
    rr_s = np.diff(peaks) / fs_radar if len(peaks) > 1 else np.array([])

    rr_clean_s, _ = clean_rr_outliers(
        rr_s,
        rr_min=RR_MIN_S,
        rr_max=RR_MAX_S,
        rel_thresh=RR_REL_THRESH,
    )
    if rr_clean_s.size > MAX_RR_COUNT:
        rr_clean_s = rr_clean_s[:MAX_RR_COUNT]

    rr_clean_s = np.asarray(rr_clean_s, dtype=float)
    hrv_raw = hrv_interp = None

    if rr_mode in ("raw", "both") and rr_clean_s.size >= 2:
        hrv_raw = compute_hrv_metrics(rr_clean_s)

    if rr_mode in ("interp", "both") and rr_clean_s.size >= 2:
        peak_times = np.cumsum(rr_clean_s)
        peak_times = peak_times - peak_times[0]
        _, rr_interp_s = build_rr_tachogram(peak_times, fs_interp=fs_interp)
        hrv_interp = compute_hrv_metrics_interp(rr_interp_s, fs_interp=fs_interp)

    return rr_clean_s, hrv_raw, hrv_interp, fs_radar


def _process_ecg(movesense_data: Dict) -> np.ndarray:
    df_ecg = movesense_data["df_ecg"]
    ecg_signal = df_ecg["mV"].values
    timestamps = df_ecg["Timestamp"]
    try:
        fs_est = _estimate_fs_from_timestamps(timestamps)
    except Exception:
        fs_est = ECG_SAMPLING_RATE

    fs_ecg = fs_est if fs_est > 0 else ECG_SAMPLING_RATE
    filtered = _bandpass_filter(ecg_signal, fs_ecg, 5.0, 15.0, order=4)
    distance = int(fs_ecg * 0.4)
    height = max(0.3, np.percentile(filtered, 80) * 0.5)
    peaks, _ = signal.find_peaks(filtered, distance=distance, height=height, prominence=0.2)
    rr_s = np.diff(peaks) / fs_ecg if len(peaks) > 1 else np.array([])
    rr_clean_s, _ = clean_rr_outliers(rr_s, rr_min=RR_MIN_S, rr_max=RR_MAX_S, rel_thresh=RR_REL_THRESH)
    if rr_clean_s.size > MAX_RR_COUNT:
        rr_clean_s = rr_clean_s[:MAX_RR_COUNT]
    return rr_clean_s


def detect_ecg_r_peaks(movesense_data: Dict) -> tuple[pd.Series, float]:
    """
    仅提取 Movesense ECG 的 R 峰时间戳与估计采样率，用于后续标签对齐。
    """
    df_ecg = movesense_data.get("df_ecg", pd.DataFrame())
    if df_ecg.empty:
        return pd.Series([], dtype="datetime64[ns]"), ECG_SAMPLING_RATE

    timestamps = pd.Series(pd.to_datetime(df_ecg["Timestamp"]))
    signal_mv = df_ecg["mV"].to_numpy(dtype=float)

    try:
        fs_ecg = _estimate_fs_from_timestamps(timestamps)
    except Exception:
        fs_ecg = ECG_SAMPLING_RATE
    fs_ecg = fs_ecg if fs_ecg > 0 else ECG_SAMPLING_RATE

    filtered = _bandpass_filter(signal_mv, fs_ecg, 5.0, 15.0, order=4)

    # 约束峰间距，避免过高心率误检；阈值使用信号分位数自适应。
    min_spacing = max(1, int(fs_ecg * 0.35))
    amp_thresh = max(0.25, np.percentile(filtered, 85) * 0.4)
    peaks, _ = signal.find_peaks(
        filtered,
        distance=min_spacing,
        height=amp_thresh,
        prominence=0.2,
    )

    if peaks.size == 0:
        return pd.Series([], dtype="datetime64[ns]"), fs_ecg

    peak_ts = timestamps.iloc[peaks].reset_index(drop=True)
    return peak_ts, float(fs_ecg)


def _build_row(base_id: str, dataset_name: str, hrv: Dict[str, float] | None, demo: Dict[str, float]) -> Dict[str, float]:
    label_val = demo.get("label", 0)
    if pd.isna(label_val):
        label_val = 0
    row = {
        "id": base_id,
        "dataset": dataset_name,
        "label": float(label_val),
        "gender": demo.get("gender", np.nan),
        "age": demo.get("age", np.nan),
        "weight_kg": demo.get("weight_kg", np.nan),
        "height_cm": demo.get("height_cm", np.nan),
    }
    for k in HRV_FEATURE_COLS:
        row[k] = hrv.get(k, np.nan) if hrv else np.nan
    return row


def _build_compare_row(
    base_id: str,
    participant: str,
    posture: str,
    state: str,
    source: str,
    radar_method: str,
    radar_mode: str,
    rr_kind: str,
    hrv: Dict[str, float] | None,
) -> Dict[str, float]:
    row = {
        "id": base_id,
        "participant": participant,
        "posture": posture,
        "state": state,
        "source": source,
        "radar_method": radar_method,
        "radar_mode": radar_mode,
        "rr_kind": rr_kind,
    }
    for k in HRV_FEATURE_COLS:
        row[k] = hrv.get(k, np.nan) if hrv else np.nan
    return row


def compute_hrv_for_extensive_ecg(
    data_root: Path,
    participants_info_xlsx: Path,
    out_csv: Path | None = None,
    rr_mode: str = "raw",
    verbose: bool = False,
    summary: bool = False,
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    遍历 Extensive 数据集所有片段，从 movesense_ecg 提取 RR/HRV + 人口学信息。
    仅计算原始 RR HRV，不再输出插值版（rr_mode 保留兼容性，统一按 raw 处理）。
    """
    def log_info(msg: str, level: int = logging.INFO) -> None:
        if verbose and not QUIET_MODE:
            LOGGER.log(level, msg)

    _ = rr_mode  # 兼容旧参数，ECG 仅输出 raw

    demo_map = _load_demographics(participants_info_xlsx)
    raw_rows: List[Dict[str, float]] = []
    total_sessions = 0
    n_ok = 0
    n_fail = 0

    for session_path, participant, posture, state in _iter_sessions(data_root):
        total_sessions += 1
        try:
            _radar_data, movesense_data, _meta = loadTestDataFromDataset(
                str(session_path),
                plot=False,
                log_summary=False,
                verbose=verbose,
                max_duration_s=MAX_ECG_DURATION_SEC,
            )
        except Exception as exc:
            n_fail += 1
            log_info(f"[skip] {session_path}: load error {exc}", level=logging.WARNING)
            continue
        if movesense_data is None:
            n_fail += 1
            log_info(f"[skip] {session_path}: missing ECG data", level=logging.WARNING)
            continue

        rr_clean_s = _process_ecg(movesense_data)
        if rr_clean_s.size < 2:
            n_fail += 1
            log_info(f"[skip] {session_path}: RR 间期不足", level=logging.WARNING)
            continue

        hrv_raw = compute_hrv_metrics(rr_clean_s)

        demo = _get_demo(participant, demo_map)
        base_id = f"{participant}_{_abbr_posture(posture)}_{_abbr_state(state)}"
        raw_rows.append(_build_row(base_id, "extensive_ecg", hrv_raw, demo))
        n_ok += 1

    cols = ["id", "dataset", "label", "gender", "age", "weight_kg", "height_cm"] + HRV_FEATURE_COLS
    df_raw = pd.DataFrame(raw_rows).reindex(columns=cols) if raw_rows else pd.DataFrame(columns=cols)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(out_csv, index=False)

    print(f"[extensive_ecg] 数据集处理完成: 总计 {total_sessions} 条, 成功 {n_ok} 条, 失败 {n_fail} 条")

    return df_raw, None


def compute_hrv_for_extensive_radar(
    data_root: Path,
    participants_info_xlsx: Path,
    out_csv: Path | None = None,
    rr_mode: str = "both",
    verbose: bool = False,
    summary: bool = False,
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    遍历 Extensive 数据集所有片段，从 radar_rFFTs/phase 信号提取 RR/HRV + 人口学信息。
    返回 (df_raw, df_interp)；rr_mode 控制 raw/interp/both。
    """
    def log_info(msg: str, level: int = logging.INFO) -> None:
        if verbose and not QUIET_MODE:
            LOGGER.log(level, msg)

    demo_map = _load_demographics(participants_info_xlsx)
    raw_rows: List[Dict[str, float]] = []
    interp_rows: List[Dict[str, float]] = []
    total_sessions = 0
    n_ok = 0
    n_fail = 0

    for session_path, participant, posture, state in _iter_sessions(data_root):
        total_sessions += 1
        try:
            radar_data, _movesense_data, _meta = loadTestDataFromDataset(
                str(session_path),
                plot=False,
                log_summary=False,
                verbose=verbose,
                max_duration_s=MAX_ECG_DURATION_SEC,
            )
        except Exception as exc:
            n_fail += 1
            log_info(f"[skip] {session_path}: load error {exc}", level=logging.WARNING)
            continue
        if radar_data is None:
            n_fail += 1
            log_info(f"[skip] {session_path}: missing radar data", level=logging.WARNING)
            continue

        _vital_signs, _heart_sig, rr_clean_s, fs_radar = _process_radar(radar_data)
        if fs_radar <= 0:
            n_fail += 1
            log_info(f"[skip] {session_path}: invalid radar fs", level=logging.WARNING)
            continue
        if rr_clean_s.size < 2:
            n_fail += 1
            log_info(f"[skip] {session_path}: RR 间期不足", level=logging.WARNING)
            continue

        hrv_raw, hrv_interp = _hrv_from_rr(rr_clean_s, rr_mode=rr_mode, fs_interp=FS_INTERP_DEFAULT)

        demo = _get_demo(participant, demo_map)
        base_id = f"{participant}_{_abbr_posture(posture)}_{_abbr_state(state)}"
        if rr_mode in ("raw", "both"):
            raw_rows.append(_build_row(base_id, "extensive_radar", hrv_raw, demo))
        if rr_mode in ("interp", "both"):
            interp_rows.append(_build_row(base_id, "extensive_radar", hrv_interp, demo))
        n_ok += 1

    cols = ["id", "dataset", "label", "gender", "age", "weight_kg", "height_cm"] + HRV_FEATURE_COLS
    df_raw = pd.DataFrame(raw_rows).reindex(columns=cols) if raw_rows else pd.DataFrame(columns=cols)
    df_interp = pd.DataFrame(interp_rows).reindex(columns=cols) if interp_rows else pd.DataFrame(columns=cols)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if rr_mode == "both":
            df_raw.to_csv(out_csv, index=False)
            interp_path = out_csv.with_name(f"{out_csv.stem}_interp{out_csv.suffix}")
            df_interp.to_csv(interp_path, index=False)
        elif rr_mode == "raw":
            df_raw.to_csv(out_csv, index=False)
        else:
            df_interp.to_csv(out_csv, index=False)

    print(f"[extensive_radar] 数据集处理完成: 总计 {total_sessions} 条, 成功 {n_ok} 条, 失败 {n_fail} 条")

    if rr_mode == "raw":
        return df_raw, None
    if rr_mode == "interp":
        return None, df_interp
    return df_raw, df_interp


def compare_extensive_hrv_methods(
    data_root: Path,
    participants_info_xlsx: Path,
    out_csv: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    遍历 Extensive 数据集，对比 ECG 参考与 3 条雷达 HRV 方法线（fft/dwt/interp，均为 direct）。
    """
    def log_info(msg: str, level: int = logging.INFO) -> None:
        if verbose and not QUIET_MODE:
            LOGGER.log(level, msg)

    rows: List[Dict[str, float]] = []
    total_sessions = 0
    n_ok = 0
    n_skip = 0
    method_success: Dict[tuple[str, str], int] = {}
    method_fail: Dict[tuple[str, str], int] = {}

    for session_path, participant, posture, state in _iter_sessions(data_root):
        total_sessions += 1
        try:
            radar_data, movesense_data, _meta = loadTestDataFromDataset(
                str(session_path),
                plot=False,
                log_summary=False,
                verbose=verbose,
                max_duration_s=MAX_ECG_DURATION_SEC,
            )
        except Exception as exc:
            n_skip += 1
            log_info(f"[skip] {session_path}: load error {exc}", level=logging.WARNING)
            continue

        if radar_data is None or movesense_data is None:
            n_skip += 1
            log_info(f"[skip] {session_path}: missing radar or ECG", level=logging.WARNING)
            continue

        try:
            rr_ecg_clean_s = _process_ecg(movesense_data)
        except Exception as exc:
            n_skip += 1
            log_info(f"[skip] {session_path}: ECG processing error {exc}", level=logging.WARNING)
            continue

        if rr_ecg_clean_s.size < 2:
            n_skip += 1
            log_info(f"[skip] {session_path}: ECG RR too short", level=logging.WARNING)
            continue

        posture_abbr = _abbr_posture(posture)
        state_abbr = _abbr_state(state)
        base_id = f"{participant}_{posture_abbr}_{state_abbr}"

        hrv_ecg_raw = compute_hrv_metrics(rr_ecg_clean_s)

        rows.append(
            _build_compare_row(
                base_id=base_id,
                participant=participant,
                posture=posture_abbr,
                state=state_abbr,
                source="ecg",
                radar_method="ecg",
                radar_mode="reference",
                rr_kind="raw",
                hrv=hrv_ecg_raw,
            )
        )

        try:
            vital_signs, heart_signal, rr_base, fs_radar = _process_radar(radar_data)
        except Exception as exc:
            n_skip += 1
            log_info(f"[skip] {session_path}: radar processing error {exc}", level=logging.WARNING)
            continue

        if heart_signal.size == 0 or fs_radar <= 0:
            n_skip += 1
            log_info(f"[skip] {session_path}: radar heart signal empty", level=logging.WARNING)
            continue

        rr_legacy = np.array([], dtype=float)
        hrv_legacy_raw: Dict[str, float] | None = None
        hrv_legacy_interp: Dict[str, float] | None = None
        try:
            rr_legacy, hrv_legacy_raw, hrv_legacy_interp, _fs_legacy = process_extensive_radar_legacy(
                radar_data,
                rr_mode="both",
                fs_interp=FS_INTERP_DEFAULT,
            )
        except Exception as exc:
            log_info(f"[warn] {session_path}: legacy radar HRV failed ({exc})", level=logging.WARNING)

        def _append_radar_rows(rr_raw: np.ndarray, method_name: str, mode: str, allow_nan: bool = True) -> None:
            rr_raw = np.asarray(rr_raw, dtype=float)
            rr_raw = rr_raw[np.isfinite(rr_raw)]
            success = True
            if rr_raw.size == 0:
                success = False
                method_fail[(method_name, mode)] = method_fail.get((method_name, mode), 0) + 1
                if not allow_nan:
                    return

            hrv_radar_raw = None
            hrv_radar_interp = None
            if success:
                rr_clean, _ = clean_rr_outliers(rr_raw, rr_min=RR_MIN_S, rr_max=RR_MAX_S, rel_thresh=RR_REL_THRESH)
                if rr_clean.size > MAX_RR_COUNT:
                    rr_clean = rr_clean[:MAX_RR_COUNT]
                if rr_clean.size < 2:
                    success = False
                    method_fail[(method_name, mode)] = method_fail.get((method_name, mode), 0) + 1
                    log_info(f"[skip] {session_path}: {method_name}_{mode} RR after clean too short", level=logging.WARNING)
                else:
                    hrv_radar_raw = compute_hrv_metrics(rr_clean)
                    peak_times_radar = np.cumsum(rr_clean)
                    peak_times_radar = peak_times_radar - peak_times_radar[0]
                    _, rr_interp_s = build_rr_tachogram(peak_times_radar, fs_interp=FS_INTERP_DEFAULT)
                    hrv_radar_interp = compute_hrv_metrics_interp(rr_interp_s, fs_interp=FS_INTERP_DEFAULT)

            rows.append(
                _build_compare_row(
                    base_id=base_id,
                    participant=participant,
                    posture=posture_abbr,
                    state=state_abbr,
                    source="radar",
                    radar_method=method_name,
                    radar_mode=mode,
                    rr_kind="raw",
                    hrv=hrv_radar_raw,
                )
            )
            rows.append(
                _build_compare_row(
                    base_id=base_id,
                    participant=participant,
                    posture=posture_abbr,
                    state=state_abbr,
                    source="radar",
                    radar_method=method_name,
                    radar_mode=mode,
                    rr_kind="interp",
                    hrv=hrv_radar_interp,
                )
            )
            if success:
                method_success[(method_name, mode)] = method_success.get((method_name, mode), 0) + 1

        def _append_radar_legacy(hrv_val: Dict[str, float] | None, method_name: str, mode: str, rr_kind: str) -> None:
            rows.append(
                _build_compare_row(
                    base_id=base_id,
                    participant=participant,
                    posture=posture_abbr,
                    state=state_abbr,
                    source="radar",
                    radar_method=method_name,
                    radar_mode=mode,
                    rr_kind=rr_kind,
                    hrv=hrv_val,
                )
            )
            if hrv_val is not None:
                method_success[(method_name, mode)] = method_success.get((method_name, mode), 0) + 1
            else:
                method_fail[(method_name, mode)] = method_fail.get((method_name, mode), 0) + 1

        # radar_interp_direct：使用 _process_radar 输出的波形插值 + QRS RR
        _append_radar_rows(rr_base, "interp", "direct", allow_nan=False)

        # radar_fft 仅保留 direct（旧版 raw HRV）
        _append_radar_legacy(hrv_legacy_raw, "fft", "direct", "raw")

        # DWT direct
        try:
            hr_dwt_direct, heart_dwt_direct = estimate_hr_from_dwt(
                vital_signs,
                fs_radar,
                mode="direct",
                heart_band=RADAR_HR_BAND,
            )
            rr_dwt_direct = estimate_rr_fft(
                heart_dwt_direct,
                fs_radar,
                method="direct",
                hr_band=RADAR_HR_BAND,
            )
            if rr_dwt_direct.size < 2 and np.isfinite(hr_dwt_direct) and hr_dwt_direct > 0:
                rr_val = 60.0 / hr_dwt_direct
                duration = max(1.0, heart_dwt_direct.size / fs_radar)
                n_rr = max(2, int(duration / max(rr_val, 1e-6)))
                rr_dwt_direct = np.full(n_rr, rr_val, dtype=float)
            _append_radar_rows(rr_dwt_direct, "dwt", "direct", allow_nan=False)
        except ImportError as exc:
            log_info(f"[skip] {session_path}: dwt unavailable ({exc})", level=logging.WARNING)
            method_fail[("dwt", "direct")] = method_fail.get(("dwt", "direct"), 0) + 1
            _append_radar_rows(np.array([]), "dwt", "direct", allow_nan=True)
        except Exception as exc:
            log_info(f"[skip] {session_path}: dwt_direct failed ({exc})", level=logging.WARNING)
            method_fail[("dwt", "direct")] = method_fail.get(("dwt", "direct"), 0) + 1
            _append_radar_rows(np.array([]), "dwt", "direct", allow_nan=True)

        n_ok += 1

    cols = [
        "id",
        "participant",
        "posture",
        "state",
        "source",
        "radar_method",
        "radar_mode",
        "rr_kind",
    ] + HRV_FEATURE_COLS

    df = pd.DataFrame(rows).reindex(columns=cols) if rows else pd.DataFrame(columns=cols)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        # 额外拆分输出，便于分别查看各方法
        for (src, mth, mode), sub_df in df.groupby(["source", "radar_method", "radar_mode"]):
            safe_src = str(src).replace("/", "_")
            safe_method = str(mth).replace("/", "_")
            safe_mode = str(mode).replace("/", "_")
            split_path = out_csv.with_name(f"{out_csv.stem}_{safe_src}_{safe_method}_{safe_mode}{out_csv.suffix}")
            sub_df.to_csv(split_path, index=False)

    log_info(
        f"[compare_extensive_hrv_methods] total={total_sessions}, processed={n_ok}, skipped={n_skip}, "
        f"methods_ok={ {k: v for k, v in method_success.items()} }, "
        f"methods_fail={ {k: v for k, v in method_fail.items()} }",
        level=logging.INFO,
    )
    print(f"[compare_extensive_hrv_methods] 数据集处理完成: 总计 {total_sessions} 条, 成功 {n_ok} 条, 跳过 {n_skip} 条")
    return df


def _build_dt_labels(peak_indices: np.ndarray, n_samples: int, max_dist_samples: int) -> np.ndarray:
    dist = np.full(n_samples, float(max_dist_samples), dtype=float)
    if n_samples == 0:
        return dist.astype(np.float32)

    if peak_indices.size:
        dist[peak_indices] = 0.0
        last = -1_000_000
        for i in range(n_samples):
            if dist[i] == 0.0:
                last = i
            elif last > -1_000_000:
                dist[i] = min(dist[i], i - last)

        nxt = 1_000_000
        for i in range(n_samples - 1, -1, -1):
            if dist[i] == 0.0:
                nxt = i
            elif nxt < 1_000_000:
                dist[i] = min(dist[i], nxt - i, dist[i])

    dist = np.minimum(dist, float(max_dist_samples))
    return (dist / float(max_dist_samples)).astype(np.float32)


def _session_id_from_path(session_path: Path) -> str:
    parts = session_path.parts
    pid = next((p for p in parts if p.upper().startswith("P")), session_path.name)
    posture = parts[-2] if len(parts) >= 2 else "posture"
    state = parts[-1] if parts else "state"
    return f"{pid}_{_abbr_posture(posture)}_{_abbr_state(state)}"


def build_radar_ecg_distance_transform_for_session(
    session_path: Path,
    fs_target: float = 120.0,
    max_dist_samples: int = 30,
    max_duration_s: float | None = MAX_ECG_DURATION_SEC,
    verbose: bool = False,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    将雷达心跳波形对齐到 ECG 时间轴并生成距离变换标签。
    返回 (session_id, t_grid, radar_resampled, dist_norm)。
    """
    session_path = Path(session_path)
    if fs_target <= 0:
        raise ValueError("fs_target must be positive.")

    log = LOGGER.info if verbose and not QUIET_MODE else lambda *_args, **_kwargs: None

    radar_data, movesense_data, _meta = loadTestDataFromDataset(
        str(session_path),
        plot=False,
        log_summary=False,
        verbose=verbose,
        max_duration_s=max_duration_s,
    )
    if radar_data is None or movesense_data is None:
        raise RuntimeError(f"{session_path} 缺少雷达或 ECG 数据")

    ts_radar, heart_signal, fs_radar = extract_radar_heart_signal_with_ts(radar_data)
    if ts_radar.empty or heart_signal.size < 2 or fs_radar <= 0:
        raise RuntimeError(f"{session_path} 雷达心跳波形为空或帧率无效")

    peak_ts, fs_ecg = detect_ecg_r_peaks(movesense_data)
    if peak_ts.empty:
        raise RuntimeError(f"{session_path} 未检测到 ECG R 峰")
    log(f"[dt] fs_radar={fs_radar:.2f}Hz, fs_ecg~{fs_ecg:.2f}Hz, radar_frames={len(ts_radar)}, peaks={len(peak_ts)}")

    ts_ecg_full = pd.to_datetime(movesense_data["df_ecg"]["Timestamp"])
    t_start = max(ts_radar.iloc[0], ts_ecg_full.iloc[0])
    t_end = min(ts_radar.iloc[-1], ts_ecg_full.iloc[-1])
    if t_end <= t_start:
        raise RuntimeError(f"{session_path} 雷达与 ECG 无交集")

    radar_mask = (ts_radar >= t_start) & (ts_radar <= t_end)
    ts_radar_win = ts_radar[radar_mask].reset_index(drop=True)
    heart_win = heart_signal[radar_mask.to_numpy()]
    if ts_radar_win.empty or heart_win.size < 2:
        raise RuntimeError(f"{session_path} 重叠时间段内雷达帧不足")

    peak_mask = (peak_ts >= t_start) & (peak_ts <= t_end)
    peak_ts_win = peak_ts[peak_mask].reset_index(drop=True)
    if peak_ts_win.empty:
        raise RuntimeError(f"{session_path} 重叠时间段内无 ECG R 峰")

    t0 = t_start
    t_radar_sec = (ts_radar_win - t0).dt.total_seconds().to_numpy()
    t_peaks_sec = (peak_ts_win - t0).dt.total_seconds().to_numpy()

    # 去重 + 保持时间递增，避免插值异常。
    t_radar_series = pd.Series(t_radar_sec)
    keep_mask = ~t_radar_series.duplicated()
    t_radar_sec = t_radar_series[keep_mask].to_numpy()
    heart_win = heart_win[keep_mask.to_numpy()]

    duration_overlap = float((t_end - t_start).total_seconds())
    if duration_overlap <= 1.0:
        raise RuntimeError(f"{session_path} 重叠时间过短 ({duration_overlap:.2f}s)")

    # 统一到雷达覆盖区间内，避免外推。
    t_min = float(t_radar_sec[0])
    t_max = min(float(t_radar_sec[-1]), duration_overlap)
    step = 1.0 / fs_target
    if t_max <= t_min + step:
        raise RuntimeError(f"{session_path} 可用雷达时间段过短")
    t_grid = np.arange(t_min, t_max, step, dtype=float)
    if t_grid.size < 3:
        raise RuntimeError(f"{session_path} t_grid 采样点过少")

    # 若原始帧率已接近目标采样率，则直接使用线性插值对齐到统一网格。
    interp_kind = "linear" if t_radar_sec.size < 8 else "cubic"
    interp_fn = interp1d(
        t_radar_sec,
        heart_win,
        kind=interp_kind,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    radar_resampled = interp_fn(t_grid).astype(float)

    mean_val = float(np.mean(radar_resampled))
    std_val = float(np.std(radar_resampled))
    if std_val < 1e-6:
        radar_resampled = radar_resampled - mean_val
    else:
        radar_resampled = (radar_resampled - mean_val) / std_val

    peaks_rel = t_peaks_sec[(t_peaks_sec >= t_grid[0]) & (t_peaks_sec <= t_grid[-1] + 1e-9)]
    if peaks_rel.size == 0:
        raise RuntimeError(f"{session_path} 公共时间段内无可用 R 峰")

    peak_indices = np.rint((peaks_rel - t_grid[0]) * fs_target).astype(int)
    peak_indices = np.unique(peak_indices[(peak_indices >= 0) & (peak_indices < t_grid.size)])
    dist_norm = _build_dt_labels(peak_indices, t_grid.size, max_dist_samples)

    session_id = _session_id_from_path(session_path)
    return session_id, t_grid, radar_resampled, dist_norm
