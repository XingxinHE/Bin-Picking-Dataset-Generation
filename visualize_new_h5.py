"""
H5 Dataset Visualization Tool

Visualize H5 dataset with:
1. Object meshes transformed by 6D pose from H5 labels
2. Full point cloud (all points)
3. Per-object point clouds (subdivided by obj_id)
4. Per-class point clouds (subdivided by class_id)

Usage: uv run visualize_new_h5.py --h5 data/teris/training/h5/cycle_0001/005.h5
"""

import numpy as np
import trimesh
import polyscope as ps
import os
import h5py
import json
import argparse

# Configuration
MODEL_DIR = "model/teris"
CLASS_MAPPING_FILE = "model/teris/class.json"

# Color palette for objects/classes
COLORS = [
    (0.121, 0.466, 0.705),   # Blue
    (1.000, 0.498, 0.054),   # Orange
    (0.172, 0.627, 0.172),   # Green
    (0.839, 0.152, 0.156),   # Red
    (0.580, 0.403, 0.741),   # Purple
    (0.549, 0.337, 0.294),   # Brown
    (0.890, 0.466, 0.760),   # Pink
    (0.498, 0.498, 0.498),   # Gray
    (0.737, 0.741, 0.133),   # Yellow-green
    (0.090, 0.745, 0.811),   # Cyan
]


def load_class_mapping():
    """Load class name to ID mapping"""
    if os.path.exists(CLASS_MAPPING_FILE):
        with open(CLASS_MAPPING_FILE, 'r') as f:
            return json.load(f)
    return {}


def get_class_name_from_id(class_id, class_mapping):
    """Reverse lookup: class_id -> class_name"""
    for name, cid in class_mapping.items():
        if cid == class_id:
            return name
    return None


def add_coordinate_axes():
    """Add XYZ coordinate axes at origin"""
    ps.register_curve_network("origin_x_axis",
                              nodes=np.array([[0, 0, 0], [0.1, 0, 0]]),
                              edges=np.array([[0, 1]]),
                              color=(1, 0, 0),
                              radius=0.002)
    ps.register_curve_network("origin_y_axis",
                              nodes=np.array([[0, 0, 0], [0, 0.1, 0]]),
                              edges=np.array([[0, 1]]),
                              color=(0, 1, 0),
                              radius=0.002)
    ps.register_curve_network("origin_z_axis",
                              nodes=np.array([[0, 0, 0], [0, 0, 0.1]]),
                              edges=np.array([[0, 1]]),
                              color=(0, 0, 1),
                              radius=0.002)


def main():
    parser = argparse.ArgumentParser(description="Visualize H5 dataset")
    parser.add_argument("--h5", type=str, required=True, help="Path to H5 file")
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        print(f"Error: H5 file not found: {args.h5}")
        return

    # Load class mapping
    class_mapping = load_class_mapping()
    print(f"Loaded class mapping: {class_mapping}")

    # Load H5 data
    print(f"Loading H5 file: {args.h5}")
    with h5py.File(args.h5, 'r') as f:
        data = f['data'][:]      # (N, 3) - XYZ
        labels = f['labels'][:]  # (N, 15) - Trans(3), Rot(9), Vis(1), ObjID(1), ClassID(1)

    points = data[:, :3]
    obj_ids = labels[:, 13].astype(int)
    class_ids = labels[:, 14].astype(int)

    unique_obj_ids = np.unique(obj_ids)
    unique_class_ids = np.unique(class_ids)

    print(f"Loaded {len(points)} points")
    print(f"Unique Object IDs: {unique_obj_ids}")
    print(f"Unique Class IDs: {unique_class_ids}")

    # Initialize Polyscope
    ps.init()
    ps.set_up_dir("z_up")
    add_coordinate_axes()

    # Create groups
    ps.create_group("Meshes")
    ps.create_group("All Points")
    ps.create_group("Per-Object Points")
    ps.create_group("Per-Class Points")

    # ----- (1) Register Object Meshes -----
    print("\n--- Registering Object Meshes ---")
    mesh_cache = {}  # Cache loaded meshes by class name

    for obj_id in unique_obj_ids:
        mask = obj_ids == obj_id
        # Get pose from first point with this obj_id
        first_idx = np.where(mask)[0][0]
        trans = labels[first_idx, 0:3]
        rot = labels[first_idx, 3:12].reshape(3, 3)
        cls_id = int(labels[first_idx, 14])

        # Get class name
        cls_name = get_class_name_from_id(cls_id, class_mapping)
        if cls_name is None:
            print(f"Warning: Unknown class ID {cls_id} for obj {obj_id}")
            continue

        # Load mesh (cache it)
        if cls_name not in mesh_cache:
            mesh_path = os.path.join(MODEL_DIR, f"{cls_name}.obj")
            if not os.path.exists(mesh_path):
                print(f"Warning: Mesh not found: {mesh_path}")
                continue
            mesh_cache[cls_name] = trimesh.load(mesh_path)
            print(f"Loaded mesh: {mesh_path}")

        mesh = mesh_cache[cls_name].copy()

        # Build 4x4 transform
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = trans

        # Apply URDF fix: Rotate mesh 180 deg around X axis
        # The URDF has <origin rpy="3.14159 0 0" ... /> for visual mesh
        # We need to apply this to the raw mesh before placing it in the world
        urdf_fix = np.eye(4)
        urdf_fix[:3, :3] = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])[:3, :3]
        mesh.apply_transform(urdf_fix)

        mesh.apply_transform(transform)

        # Register with Polyscope
        color_idx = obj_id % len(COLORS)
        ps_mesh = ps.register_surface_mesh(f"obj_{obj_id}", mesh.vertices, mesh.faces)
        ps_mesh.set_color(COLORS[color_idx])
        ps_mesh.add_to_group("Meshes")

        print(f"Registered obj_{obj_id} (class={cls_name})")

    # ----- (2) Register Full Point Cloud -----
    print("\n--- Registering Full Point Cloud ---")
    pcl_all = ps.register_point_cloud("all_points", points, radius=0.001)
    pcl_all.set_color((0.5, 0.5, 0.5))  # Gray
    pcl_all.add_to_group("All Points")

    # ----- (3) Register Per-Object Point Clouds -----
    print("\n--- Registering Per-Object Point Clouds ---")
    for obj_id in unique_obj_ids:
        mask = obj_ids == obj_id
        obj_points = points[mask]

        color_idx = obj_id % len(COLORS)
        pcl = ps.register_point_cloud(f"points_obj_{obj_id}", obj_points, radius=0.0015)
        pcl.set_color(COLORS[color_idx])
        pcl.add_to_group("Per-Object Points")
        pcl.set_enabled(False)  # Disable by default to avoid clutter

    # ----- (4) Register Per-Class Point Clouds -----
    print("\n--- Registering Per-Class Point Clouds ---")
    for cls_id in unique_class_ids:
        mask = class_ids == cls_id
        cls_points = points[mask]

        cls_name = get_class_name_from_id(cls_id, class_mapping)
        label = cls_name if cls_name else f"class_{cls_id}"

        color_idx = cls_id % len(COLORS)
        pcl = ps.register_point_cloud(f"points_{label}", cls_points, radius=0.0015)
        pcl.set_color(COLORS[color_idx])
        pcl.add_to_group("Per-Class Points")
        pcl.set_enabled(False)  # Disable by default

    print("\n--- Launching Polyscope ---")
    ps.show()


if __name__ == "__main__":
    main()
