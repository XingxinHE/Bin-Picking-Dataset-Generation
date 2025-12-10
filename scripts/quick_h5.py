import h5py
import numpy as np

FILE_PATH = "data/teris/training/h5/cycle_0001/005.h5"

with h5py.File(FILE_PATH, 'r') as f:
    data = f['data'][:]
    labels = f['labels'][:]

    print(f"Data Shape: {data.shape}")
    print(f"Labels Shape: {labels.shape}")

    # Check Class ID (Column 14)
    cls_ids = labels[:, 14]
    unique_cls = np.unique(cls_ids)
    print(f"Unique Class IDs: {unique_cls}")

    # Check Obj ID (Column 13)
    obj_ids = labels[:, 13]
    unique_obj = np.unique(obj_ids)
    print(f"Unique Obj IDs: {unique_obj}")

    # Check Visibility (Column 12)
    vis = labels[:, 12]
    print(f"Visibility Stats: Min={np.min(vis):.4f}, Max={np.max(vis):.4f}, Mean={np.mean(vis):.4f}")
    print(f"Sample Visibilities: {vis[:10]}")
