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
CLASS_MAPPING = {}

def load_class_mapping(mapping_file):
    global CLASS_MAPPING
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            CLASS_MAPPING = json.load(f)
        print(f"Loaded class mapping: {CLASS_MAPPING}")
    else:
        print("Warning: Class mapping file not found.")

def read_label_csv(file_path):
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

            cls_id = CLASS_MAPPING.get(cls_name, 0) # Default to 0 if not found

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

def process_cycle(cycle_dir, output_dir):
    cycle_name = os.path.basename(cycle_dir)
    print(f"Processing {cycle_name}...")

    # Create output dir
    h5_cycle_dir = os.path.join(output_dir, cycle_name)
    os.makedirs(h5_cycle_dir, exist_ok=True)

    # Find all drops
    gt_files = sorted(glob.glob(os.path.join(cycle_dir, "gt", "*.csv")))

    for gt_file in gt_files:
        drop_name = os.path.basename(gt_file).replace(".csv", "")

        # Paths
        depth_path = os.path.join(cycle_dir, "p_depth", f"{drop_name}_depth.png")
        seg_path = os.path.join(cycle_dir, "p_segmentation", f"{drop_name}_segmentation.png")

        if not os.path.exists(depth_path) or not os.path.exists(seg_path):
            print(f"Skipping {drop_name}: Missing depth or seg image.")
            continue

        # Load Images
        # Depth is saved as uint16, normalized 0-65535 -> CAMERA_NEAR to CAMERA_FAR
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_m = depth_raw.astype(np.float32) / 65535.0 * (CAMERA_FAR - CAMERA_NEAR) + CAMERA_NEAR

        seg_raw = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

        # Load GT
        poses = read_label_csv(gt_file)

        # Generate Point Cloud
        points, point_ids = depth_to_pointcloud(depth_m, seg_raw)

        if len(points) == 0:
            print(f"Warning: No foreground points for {drop_name}")
            continue

        # Sampling
        if len(points) >= TARGET_NUM_POINT:
            choice = np.random.choice(len(points), TARGET_NUM_POINT, replace=False)
        else:
            choice = np.random.choice(len(points), TARGET_NUM_POINT, replace=True)

        points = points[choice]
        point_ids = point_ids[choice]

        # Assign Labels (Sparse-to-Dense)
        # We need to map point_ids (which are object IDs from PyBullet) to indices in our poses list.
        # BUT, poses list is just a list. We need to look up by 'obj_id'.

        # Create lookup dict
        pose_lookup = {p['obj_id']: p for p in poses}

        # Prepare Label Arrays
        # Trans(3), Rot(9), Vis(1), ObjID(1), ClassID(1) -> Total 15
        labels = np.zeros((TARGET_NUM_POINT, 15), dtype=np.float32)

        for i in range(TARGET_NUM_POINT):
            pid = point_ids[i]
            if pid in pose_lookup:
                pose = pose_lookup[pid]
                labels[i, 0:3] = pose['trans']
                labels[i, 3:12] = pose['rot']
                labels[i, 12] = pose['vis']
                labels[i, 13] = float(pid)
                labels[i, 14] = float(pose['cls_id'])
            else:
                # Should not happen if segmentation matches CSV
                pass

        # Save H5
        h5_path = os.path.join(h5_cycle_dir, f"{drop_name}.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset("data", data=points)
            f.create_dataset("labels", data=labels)

        print(f"Saved {h5_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/teris/training", help="Path to training data folder")
    parser.add_argument("--class_map", type=str, default="model/teris/class.json", help="Path to class mapping json")
    args = parser.parse_args()

    load_class_mapping(args.class_map)

    # Process all cycles
    cycle_dirs = sorted(glob.glob(os.path.join(args.data_dir, "cycle_*"))) # Matches cycle_0001, etc.
    # Note: The structure is data/teris/training/gt/cycle_XXXX
    # But the script puts cycle folders INSIDE gt, p_depth, etc.
    # Wait, let's check 1_pybullet_create_n_collect.py structure.
    # RGB_IMG_FOLDER_PATH_ = .../training/p_rgb
    # cycle_rgb_path = .../training/p_rgb/cycle_0001

    # So we should iterate over cycle IDs, not folders directly if they are split.
    # Let's assume we iterate over `gt/cycle_XXXX` and infer others.

    gt_root = os.path.join(args.data_dir, "gt")
    h5_root = os.path.join(args.data_dir, "h5")

    cycle_names = [d for d in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, d))]

    for cycle_name in cycle_names:
        # Construct a "virtual" cycle dir object that has access to all subfolders
        # Actually, process_cycle needs to know where to find gt, p_depth, etc.
        # Let's pass the root data dir and cycle name

        print(f"Processing {cycle_name}...")

        h5_cycle_dir = os.path.join(h5_root, cycle_name)
        os.makedirs(h5_cycle_dir, exist_ok=True)

        gt_cycle_dir = os.path.join(gt_root, cycle_name)
        gt_files = sorted(glob.glob(os.path.join(gt_cycle_dir, "*.csv")))

        for gt_file in gt_files:
            drop_name = os.path.basename(gt_file).replace(".csv", "")

            depth_path = os.path.join(args.data_dir, "p_depth", cycle_name, f"{drop_name}_depth.png")
            seg_path = os.path.join(args.data_dir, "p_segmentation", cycle_name, f"{drop_name}_segmentation.png")

            if not os.path.exists(depth_path) or not os.path.exists(seg_path):
                print(f"Skipping {drop_name}: Missing depth or seg image.")
                continue

            # Load Images
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_m = depth_raw.astype(np.float32) / 65535.0 * (CAMERA_FAR - CAMERA_NEAR) + CAMERA_NEAR

            seg_raw = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

            # Load GT
            poses = read_label_csv(gt_file)

            # Generate Point Cloud
            points, point_ids = depth_to_pointcloud(depth_m, seg_raw)

            if len(points) == 0:
                print(f"Warning: No foreground points for {drop_name}")
                continue

            # Sampling
            if len(points) >= TARGET_NUM_POINT:
                choice = np.random.choice(len(points), TARGET_NUM_POINT, replace=False)
            else:
                choice = np.random.choice(len(points), TARGET_NUM_POINT, replace=True)

            points = points[choice]
            point_ids = point_ids[choice]

            # Create lookup dict
            pose_lookup = {p['obj_id']: p for p in poses}

            # ----- Recalculate Visibility from Point Counts -----
            # V_i = N_i / N_max (Paper formula)
            unique_ids, counts = np.unique(point_ids, return_counts=True)
            point_count_map = dict(zip(unique_ids, counts))
            n_max = max(counts) if len(counts) > 0 else 1

            # Prepare Label Arrays
            labels = np.zeros((TARGET_NUM_POINT, 15), dtype=np.float32)

            for i in range(TARGET_NUM_POINT):
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

if __name__ == "__main__":
    main()
