"""
AUTHOR: Felipe Parralejo

METHODS TO READ DATA IN THE FORMAT INDICATED IN THE ARTICLE

Additionally, an IIR filter generator is included

"""

import json
import pickle
import zlib
from typing import Optional, Tuple

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from config_runtime import MAX_ECG_DURATION_SEC, MAX_RADAR_FRAMES, QUIET_MODE

LOGGER = logging.getLogger(__name__)

# 默认截取时长由全局配置控制，避免 Notebook 卡死。
DEFAULT_MAX_DURATION_S = float(MAX_ECG_DURATION_SEC)


def _log(msg: str, verbose: bool, level: int = logging.INFO) -> None:
    if verbose and not QUIET_MODE:
        LOGGER.log(level, msg)


def _trim_df_by_duration(df: pd.DataFrame, time_col: str, max_duration_s: Optional[float]) -> Tuple[pd.DataFrame, bool]:
    """
    截断 DataFrame 到指定时长，返回 (截断后的 df, 是否截断过)。
    """
    if max_duration_s is None or df.empty:
        return df, False
    ts = pd.to_datetime(df[time_col])
    start = ts.iloc[0]
    end = start + pd.Timedelta(seconds=max_duration_s)
    mask = ts <= end
    truncated = mask.sum() < len(df)
    return df.loc[mask].copy(), truncated


def load_nbr(folder: str, verbose: bool = False):
    non_breathing_ts = pd.read_csv(folder + "/non_breathing_ts.csv", header=None, index_col=0, parse_dates=[1]).squeeze().to_dict()
    _log(" -> [helper_fns.load_nbr] Finished loading non-breathing timestamps", verbose)
    return non_breathing_ts


def load_movesense(folder: str, max_duration_s: Optional[float] = DEFAULT_MAX_DURATION_S, verbose: bool = False):
    df_acc = pd.read_csv(folder + "/movesense_acc.csv", parse_dates=['Timestamp'])
    df_ecg = pd.read_csv(folder + "/movesense_ecg.csv", parse_dates=['Timestamp'])

    acc_truncated = ecg_truncated = False
    if max_duration_s is not None:
        df_acc, acc_truncated = _trim_df_by_duration(df_acc, "Timestamp", max_duration_s)
        df_ecg, ecg_truncated = _trim_df_by_duration(df_ecg, "Timestamp", max_duration_s)

    _log(' -> [helper_fns.load_movesense] Finished loading Movesense data', verbose)

    return {'df_acc': df_acc, 'df_ecg': df_ecg, 'truncated': acc_truncated or ecg_truncated}


def loadAndDecompress(fname):
    with open(fname,'rb') as f:
        data=f.read()
    data=zlib.decompress(data)
    data = pickle.loads(data)
    return data


def load_radar(folder: str, max_duration_s: Optional[float] = DEFAULT_MAX_DURATION_S, verbose: bool = False):
    timestamps_mmwave = pd.read_csv(folder + "/radar_timestamps.csv", parse_dates=[0], header=None).squeeze().to_list()
    chirpConfig = json.load(open(folder + "/radar_chirpConfig.json", "r"))
    rFFTs, rBins = loadAndDecompress(folder + "/radar_rFFTs.zlib")

    truncated = False
    frame_period_ms = float(chirpConfig.get("PERIODICITY", 0.0))
    frame_period_s = frame_period_ms * 1e-3 if frame_period_ms else None
    if max_duration_s is not None and frame_period_s and frame_period_s > 0:
        max_frames = max(1, int(max_duration_s / frame_period_s))
        total_frames = len(rFFTs)
        if total_frames > max_frames:
            rFFTs = rFFTs[:max_frames]
            timestamps_mmwave = timestamps_mmwave[:max_frames]
            truncated = True

    if len(rFFTs) > MAX_RADAR_FRAMES:
        rFFTs = rFFTs[:MAX_RADAR_FRAMES]
        timestamps_mmwave = timestamps_mmwave[:MAX_RADAR_FRAMES]
        truncated = True

    _log(' -> [helper_fns.load_radar] Finished loading radar data', verbose)

    return {
        'rFFTs': rFFTs,
        'rBins': rBins,
        'timestamps_mmwave': timestamps_mmwave,
        'chirpConfig': chirpConfig,
        'frame_period_s': frame_period_s,
        'truncated': truncated,
    }


def loadTestDataFromDataset(
    folder: str,
    plot: bool = False,
    max_duration_s: Optional[float] = DEFAULT_MAX_DURATION_S,
    verbose: bool = False,
    log_summary: bool = True,
):
    """
    加载雷达 / Movesense / 非呼吸时间段，并可选截取前 max_duration_s 秒。
    仅在末尾输出一次统计信息，避免中间大量调试输出。
    """
    radar_data = None
    movesense_data = None
    non_breathing_data = None

    try:
        radar_data = load_radar(folder, max_duration_s=max_duration_s, verbose=verbose)
    except Exception as e:
        LOGGER.warning("Error loading radar data: %s", e)
        return

    try:
        movesense_data = load_movesense(folder, max_duration_s=max_duration_s, verbose=verbose)
    except Exception as e:
        LOGGER.warning("Error loading movesense data: %s", e)
        return

    try:
        non_breathing_data = load_nbr(folder, verbose=verbose)
    except Exception as e:
        # 脚本可在没有 non-breathing 标记的情况下继续
        _log(f"Error loading non-breathing data: {e}", verbose, level=logging.WARNING)

    _log(" -> [helper_fns.loadTestDataFromDataset] All data loaded successfully", verbose)

    # Extract radar phase data
    raw_phase = []
    r_bin_max = 0

    for rFFT in radar_data['rFFTs']:
        rFFT_t = np.mean(rFFT, axis=0)
        comp = rFFT_t[r_bin_max]
        phase = np.arctan2(comp.imag, comp.real)
        raw_phase.append(phase)

    unwrp = np.unwrap(raw_phase)
    nodc = unwrp - np.mean(unwrp)

    # Plot radar data with acc and ecg
    if plot:
        fig, ax1 = plt.subplots(figsize=(15, 6))

        color = 'tab:red'
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('ECG (mV), Acc. (m/s²)')
        ax1.plot(movesense_data['df_ecg']['Timestamp'], movesense_data['df_ecg']['mV'], label='ECG')
        ax1.plot(movesense_data['df_acc']['Timestamp'], movesense_data['df_acc']['Y: (m/s^2)']*2+0.5, label='ACC')
        if non_breathing_data:
            ax1.fill_betweenx([-1, 2.0], non_breathing_data['begin'], non_breathing_data['end'], color='orange', alpha=0.3, label='Non-Breathing Area')

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:green'
        ax2.set_ylabel('Range FFT', color=color)  # we already handled the x-label with ax1
        ax2.plot(radar_data['timestamps_mmwave'], nodc, color=color, label='Range FFT')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        fig.legend(loc='upper left')

        plt.title('Radar and Movesense Processed Data\nLoading Test')
        plt.show()

    if log_summary and not QUIET_MODE:
        radar_frames = len(radar_data['rFFTs']) if radar_data else 0
        frame_period_s = radar_data.get("frame_period_s") if radar_data else None
        radar_duration_s = radar_frames * frame_period_s if frame_period_s else None
        movesense_ecg = movesense_data['df_ecg'] if movesense_data else pd.DataFrame()
        movesense_acc = movesense_data['df_acc'] if movesense_data else pd.DataFrame()
        summary_parts = [
            f"radar_frames={radar_frames}",
            f"radar_duration_s=~{radar_duration_s:.1f}" if radar_duration_s else "radar_duration_s=unknown",
            f"ecg_samples={len(movesense_ecg)}",
            f"acc_samples={len(movesense_acc)}",
        ]
        truncated_flags = []
        if radar_data and radar_data.get("truncated"):
            truncated_flags.append("radar")
        if movesense_data and movesense_data.get("truncated"):
            truncated_flags.append("movesense")
        if truncated_flags:
            summary_parts.append(f"truncated: {', '.join(truncated_flags)} (max {max_duration_s}s)")
        LOGGER.info("[loadTestDataFromDataset] " + " | ".join(summary_parts))

    return radar_data, movesense_data, non_breathing_data


def get_bandpass_filter(
    Fs: float,
    lowcut: float,
    highcut: float,
    order: int,
    plot_response: bool = False,
    verbose: bool = False,
):
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    if plot_response:
        plot_filter_response(b, a, Fs)

    _log(" -> [helper_fns.get_bandpass_filter] Finished creating bandpass filter", verbose)

    return b, a


def plot_filter_response(b, a, Fs):
    # Compute frequency response of the filter
    wz, hz = signal.freqz(b, a)

    # Calculate Magnitude from hz in dB
    Mag = 20*np.log10(abs(hz))

    # Calculate phase angle in degree from hz
    Phase = np.arctan2(np.imag(hz), np.real(hz))
    Phase = np.unwrap(Phase)*(180/np.pi)

    # Calculate frequency in Hz from wz
    Freq = wz*Fs/(2*np.pi)

    # Plot filter magnitude and phase responses
    fig = plt.figure(figsize=(10, 6))

    # Plot Magnitude response
    sub1 = plt.subplot(2, 1, 1)
    sub1.plot(Freq, Mag, 'r', linewidth=2)
    sub1.set_title('Magnitude Response', fontsize=20)
    sub1.set_xlabel('Frequency [Hz]', fontsize=14)
    sub1.set_ylabel('Magnitude [dB]', fontsize=14)
    sub1.grid()

    # Plot phase angle
    sub2 = plt.subplot(2, 1, 2)
    sub2.plot(Freq, Phase, 'g', linewidth=2)
    sub2.set_ylabel('Phase (degree)', fontsize=14)
    sub2.set_xlabel('Frequency (Hz)', fontsize=14)
    sub2.set_title('Phase response', fontsize=20)
    sub2.grid()

    plt.subplots_adjust(hspace=5)
    fig.tight_layout()
    plt.show(block=False)
