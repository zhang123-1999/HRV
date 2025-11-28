from __future__ import annotations

"""
示例入口脚本：调用模块化版本的 Extensive HRV 对比接口。
"""

from pathlib import Path

try:
    from datasets_extensive import compare_extensive_hrv_methods
except ImportError:
    # 兼容直接在同级目录运行的场景
    from datasets_extensive import compare_extensive_hrv_methods


def main() -> None:
    data_root = Path(r"C:\Users\不存在的骑士\Desktop\大创论文\ecg-classification-master\论文\data_ExtensivemmWave")  
    info_xlsx = Path("ParticipantsInfo.xlsx")  
    out_csv = Path("results/extensive_hrv_compare.csv")

    df = compare_extensive_hrv_methods(
        data_root=data_root,
        participants_info_xlsx=info_xlsx,
        out_csv=out_csv,
        verbose=True,
    )
    print(df.head())


if __name__ == "__main__":
    main()
