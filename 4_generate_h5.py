import os
import numpy as np
import h5py
import cv2
import json
import argparse
import glob

# Import shared camera configuration
from camera_config import (
    CAMERA_IMG_WIDTH, CAMERA_IMG_HEIGHT,
    CAMERA_NEAR, CAMERA_FAR,
    F_X, F_Y, C_X, C_Y
)

TARGET_NUM_POINT = 16384

# Class Mapping
# Global constants
# CLASS_MAPPING is now passed as an argument


def read_label_csv(file_path, class_mapping):
    """
    Reads the CSV file and returns a dictionary of poses.
    Returns:
        poses: list of dicts
    """
    poses = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            parts = line.split(',')

            # id,class,x,y,z,rot_x_axis_1...rot_z_axis_3,vis_o,vis_p
            obj_id = int(parts[0])
            cls_name = parts[1]
            trans = np.array([float(x) for x in parts[2:5]])
            rot = np.array([float(x) for x in parts[5:14]])

            # Handle optional vis_o column. vis_p is always the last element.
            # If len(parts) is 15, then vis_p is parts[14].
            # If len(parts) is 16, then vis_o is parts[14] and vis_p is parts[15].
            # In both cases, vis_p is parts[-1].
            vis_p = float(parts[-1])

            cls_id = class_mapping.get(cls_name, 0) # Default to 0 if not found

            poses.append({
                'obj_id': obj_id,
                'cls_id': cls_id,
                'trans': trans,
                'rot': rot,
                'vis': vis_p
            })
    return poses

def depth_to_pointcloud(depth_img, seg_img):
    """
    Convert depth image to point cloud.
    """
    # Create meshgrid
    xs, ys = np.meshgrid(np.arange(CAMERA_IMG_WIDTH), np.arange(CAMERA_IMG_HEIGHT))

    # Filter valid depth (and non-background segmentation)
    # Background in PyBullet usually has specific depth or seg ID.
    # Here we assume seg_img > 0 are objects.
    mask = (seg_img > 0) & (depth_img > 0)

    xs = xs[mask]
    ys = ys[mask]
    zs = depth_img[mask]
    ids = seg_img[mask]

    # Reproject to 3D
    # Z is already depth (Z-coordinate in camera frame)
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy

    # Note: PyBullet Camera looks down -Z?
    # In 1_pybullet_create_n_collect.py, we used a correction matrix to flip Y and Z.
    # So the CSV poses are in OpenCV frame (X right, Y down, Z forward).
    # The depth image from PyBullet is Z-buffer.
    # We need to ensure consistency.

    # Standard Pinhole Model (OpenCV)
    x_3d = (xs - C_X) * zs / F_X
    y_3d = (ys - C_Y) * zs / F_Y
    z_3d = zs

    points = np.stack([x_3d, y_3d, z_3d], axis=-1)

    return points, ids

def process_cycle(cycle_name, data_dir, h5_root, gt_root, target_num_point, class_mapping):
    print(f"Processing {cycle_name}...")

    h5_cycle_dir = os.path.join(h5_root, cycle_name)
    os.makedirs(h5_cycle_dir, exist_ok=True)

    gt_cycle_dir = os.path.join(gt_root, cycle_name)
    gt_files = sorted(glob.glob(os.path.join(gt_cycle_dir, "*.csv")))

    for gt_file in gt_files:
        drop_name = os.path.basename(gt_file).replace(".csv", "")

        depth_path = os.path.join(data_dir, "p_depth", cycle_name, f"{drop_name}_depth.png")
        seg_path = os.path.join(data_dir, "p_segmentation", cycle_name, f"{drop_name}_segmentation.png")

        if not os.path.exists(depth_path) or not os.path.exists(seg_path):
            print(f"Skipping {drop_name}: Missing depth or seg image.")
            continue

        # Load Images
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_m = depth_raw.astype(np.float32) / 65535.0 * (CAMERA_FAR - CAMERA_NEAR) + CAMERA_NEAR

        seg_raw = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

        # Load GT
        poses = read_label_csv(gt_file, class_mapping)

        # Create lookup dict
        pose_lookup = {p['obj_id']: p for p in poses}
        valid_ids = set(pose_lookup.keys())

        # Generate Point Cloud
        points, point_ids = depth_to_pointcloud(depth_m, seg_raw)

        if len(points) == 0:
            print(f"Warning: No foreground points for {drop_name}")
            continue

        # Filter out points that are not in the CSV (e.g. walls, plane, background)
        valid_mask = np.isin(point_ids, list(valid_ids))
        points = points[valid_mask]
        point_ids = point_ids[valid_mask]

        if len(points) == 0:
            print(f"Warning: No valid object points for {drop_name} (after filtering walls/plane)")
            continue

        # Sampling
        if len(points) >= target_num_point:
            choice = np.random.choice(len(points), target_num_point, replace=False)
        else:
            choice = np.random.choice(len(points), target_num_point, replace=True)

        points = points[choice]
        point_ids = point_ids[choice]

        # ----- Recalculate Visibility from Point Counts -----
        # V_i = N_i / N_max (Paper formula)
        unique_ids, counts = np.unique(point_ids, return_counts=True)
        point_count_map = dict(zip(unique_ids, counts))
        n_max = max(counts) if len(counts) > 0 else 1

        # Prepare Label Arrays
        labels = np.zeros((target_num_point, 15), dtype=np.float32)

        for i in range(target_num_point):
            pid = point_ids[i]
            if pid in pose_lookup:
                pose = pose_lookup[pid]
                labels[i, 0:3] = pose['trans']
                labels[i, 3:12] = pose['rot']
                # Use recalculated visibility (N_i / N_max)
                n_i = point_count_map.get(pid, 0)
                labels[i, 12] = n_i / n_max
                labels[i, 13] = float(pid)
                labels[i, 14] = float(pose['cls_id'])

        # Save H5
        h5_path = os.path.join(h5_cycle_dir, f"{drop_name}.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset("data", data=points)
            f.create_dataset("labels", data=labels)

        print(f"Saved {h5_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate H5 Dataset")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to training directory (e.g. data/teris/training)")
    parser.add_argument("--dataset_name", type=str, default="teris", help="Dataset name if data_dir not provided")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    # Determine Data Directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Default structure: data/{dataset_name}/training
        data_dir = os.path.join("data", args.dataset_name, "training")

    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")

    print(f"Processing data in: {data_dir}")

    # Load class mapping
    # Assume model folder is at model/{dataset_name}/class.json
    # Or try to find it relative to data_dir
    # data_dir = .../data/teris/training
    # model_dir = .../model/teris

    # Try generic path first
    mapping_file = os.path.join("model", args.dataset_name, "class.json")
    if not os.path.exists(mapping_file):
        print(f"Warning: Class mapping not found at {mapping_file}. Using default/empty.")
        class_mapping = {}
    else:
        with open(mapping_file, 'r') as f:
            class_mapping = json.load(f)
        print(f"Loaded class mapping: {class_mapping}")

    gt_root = os.path.join(data_dir, "gt")
    h5_root = os.path.join(data_dir, "h5")

    if not os.path.exists(gt_root):
        raise ValueError(f"GT directory not found: {gt_root}")

    cycle_names = sorted([d for d in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, d))])
    print(f"Found {len(cycle_names)} cycles.")

    import multiprocessing
    from functools import partial

    if args.workers > 1:
        with multiprocessing.Pool(processes=args.workers) as pool:
            func = partial(process_cycle,
                           data_dir=data_dir,
                           h5_root=h5_root,
                           gt_root=gt_root,
                           target_num_point=TARGET_NUM_POINT,
                           class_mapping=class_mapping)
            pool.map(func, cycle_names)
    else:
        for cycle in cycle_names:
            # Note: process_cycle signature: cycle_name, data_dir, h5_root, gt_root, target_num_point, class_mapping
            process_cycle(cycle, data_dir, h5_root, gt_root, TARGET_NUM_POINT, class_mapping)

if __name__ == "__main__":
    main()
