
import os
import glob
import h5py
import numpy as np
import multiprocessing
from functools import partial

DATASET_DIR = "/home/hex/Documents/github/compare_ppr/Bin-Picking-Dataset-Generation/data/S_packing_500cycles_80drops/training/h5"

def fix_file(file_path):
    try:
        # Open in read/write mode 'r+'
        with h5py.File(file_path, 'r+') as f:
            labels = f['labels'][:] # Read into memory

            # Check if fix is needed to avoid redundant writes
            if np.all(labels[:, 14] == 3.0):
                return None # Already fixed

            labels[:, 14] = 3.0

            # Write back
            f['labels'][...] = labels
            return None
    except Exception as e:
        return f"ERROR: {file_path} - {e}"

def main():
    print(f"Fixing labels to 3.0 (Class S) in {DATASET_DIR}")
    files = glob.glob(os.path.join(DATASET_DIR, "cycle_*", "*.h5"))
    print(f"Found {len(files)} files.")

    with multiprocessing.Pool(processes=16) as pool:
        # TQDM-like progress could be nice, but simple map is fine
        results = pool.map(fix_file, files)

    failures = [r for r in results if r is not None]

    if len(failures) == 0:
        print("SUCCESS: All files processed successfully.")
    else:
        print(f"FAILURE: Found {len(failures)} errors.")
        for f in failures[:10]:
            print(f)

if __name__ == "__main__":
    main()
