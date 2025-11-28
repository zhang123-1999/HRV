"""
Runtime switches for notebook-friendly runs.

QUIET_MODE silences verbose logs by default. MAX_* values provide
defensive truncation so long recordings don't stall the notebook.
"""
from __future__ import annotations

# Notebook 下默认安静模式，如需调试可手动切换为 False。
QUIET_MODE: bool = True

# 最长 ECG 时长（秒），超出则仅保留前段。
MAX_ECG_DURATION_SEC: int = 600

# RR 序列最多使用的间期数。
MAX_RR_COUNT: int = 600

# 雷达慢时间帧数上限，超过则截断。
MAX_RADAR_FRAMES: int = 4000
