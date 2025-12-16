
import os
import glob
import h5py
import numpy as np
import multiprocessing
from functools import partial

DATASET_DIR = "/home/hex/Documents/github/compare_ppr/Bin-Picking-Dataset-Generation/data/S_packing_500cycles_80drops/training/h5"

def check_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            labels = f['labels'][:]
            cls_ids = labels[:, 14]
            unique_ids = np.unique(cls_ids)

            # Check if all are 3.0
            if not np.all(unique_ids == 3.0):
                 if len(unique_ids) == 0:
                     return f"WARNING: {file_path} empty labels"
                 return f"FAIL: {file_path} contains IDs {unique_ids}"
            return None
    except Exception as e:
        return f"ERROR: {file_path} - {e}"

def main():
    print(f"Verifying labels are 3.0 in {DATASET_DIR}")
    files = glob.glob(os.path.join(DATASET_DIR, "cycle_*", "*.h5"))
    print(f"Found {len(files)} files.")

    with multiprocessing.Pool(processes=16) as pool:
        results = pool.map(check_file, files)

    failures = [r for r in results if r is not None]

    if len(failures) == 0:
        print("SUCCESS: All files contain only Class ID 3.0")
    else:
        print(f"FAILURE: Found {len(failures)} issues.")
        for f in failures[:10]:
            print(f)
        if len(failures) > 10: print("...")

if __name__ == "__main__":
    main()
