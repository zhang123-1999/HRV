#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from datasets_extensive import (
    _iter_sessions,
    build_radar_ecg_distance_transform_for_session,
)


def main() -> None:
    data_root = Path("data_ExtensivemmWave")
    out_dir = Path("results") / "dt_dataset_120hz"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    ok = fail = 0

    for session_path, participant, posture, state in _iter_sessions(data_root):
        try:
            session_id, t_grid, radar_120, dist_norm = build_radar_ecg_distance_transform_for_session(
                session_path,
                fs_target=120.0,
                max_dist_samples=30,
                max_duration_s=None,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - 便于批处理记录全部错误
            print(f"[skip] {session_path}: {exc}")
            fail += 1
            continue

        npz_path = out_dir / f"{session_id}.npz"
        np.savez_compressed(
            npz_path,
            session_id=session_id,
            t_grid=t_grid,
            radar_120=radar_120,
            dist_norm=dist_norm,
        )

        summary_rows.append(
            {
                "session_id": session_id,
                "participant": participant,
                "posture": posture,
                "state": state,
                "npz_path": str(npz_path),
                "n_samples": int(len(t_grid)),
                "duration_s": float(t_grid[-1] - t_grid[0]) if len(t_grid) else 0.0,
            }
        )
        ok += 1

    pd.DataFrame(summary_rows).to_csv(out_dir / "index.csv", index=False)
    print(f"[done] ok={ok}, fail={fail}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
