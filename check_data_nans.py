import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("results/nature_dataset_cleaned")
INDEX_FILE = DATA_DIR / "dataset_index.csv"

def check_nans():
    if not INDEX_FILE.exists():
        print("Index file not found.")
        return

    df = pd.read_csv(INDEX_FILE)
    print(f"Checking {len(df)} files for NaNs...")
    
    nan_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        path = row['path']
        try:
            with np.load(path) as data:
                spec = data['spectrogram']
                ihr = data['ihr_curve']
                
                has_nan_spec = np.isnan(spec).any()
                has_nan_ihr = np.isnan(ihr).any()
                
                if has_nan_spec or has_nan_ihr:
                    nan_files.append({
                        "path": path,
                        "nan_spec": has_nan_spec,
                        "nan_ihr": has_nan_ihr
                    })
        except Exception as e:
            print(f"Error reading {path}: {e}")

    print(f"\nFound {len(nan_files)} files with NaNs.")
    if nan_files:
        print("First 5 examples:")
        for f in nan_files[:5]:
            print(f)

if __name__ == "__main__":
    check_nans()
