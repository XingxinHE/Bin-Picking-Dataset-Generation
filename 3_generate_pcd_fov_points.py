import numpy as np
import open3d as o3d
import os
from multiprocessing import Process
import h5py
import json
import trimesh

# load data & environment generation setting from json file
f = open("data_generation_setting.json")
json_setting = json.load(f)

START_CYCLE_ = json_setting["data_generation"]["start_cycle"]
MAX_CYCLE_ = json_setting["data_generation"]["end_cycle"]
MAX_DROP_ = json_setting["data_generation"]["max_drop"]

DATASET_FOLDER_NAME_ = json_setting["folder_struct"]["dataset_folder_name"]
ITEM_NAME_ = json_setting["folder_struct"]["item_name"]
TRAIN_TEST_FOLDER_NAME_ = json_setting["folder_struct"]["train_test_folder_name"]
SCENE_FOLDER_NAME_ = json_setting["folder_struct"]["scene_folder_name"]
CROPPED_SCENE_FOLDER_NAME_ = json_setting["folder_struct"]["cropped_scene_folder_name"]
H5_FOLDER_NAME_ = json_setting["folder_struct"]["h5_folder_name"]

CURR_DIR_ = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_PATH_ = os.path.join(CURR_DIR_, DATASET_FOLDER_NAME_)
ITEM_FOLDER_PATH_ = os.path.join(CURR_DIR_, DATASET_FOLDER_NAME_, ITEM_NAME_)
TRAIN_TEST_FOLDER_PATH_ = os.path.join(
    CURR_DIR_, DATASET_FOLDER_NAME_, ITEM_FOLDER_PATH_, TRAIN_TEST_FOLDER_NAME_
)
SCENE_FOLDER_PATH_ = os.path.join(
    CURR_DIR_,
    DATASET_FOLDER_NAME_,
    ITEM_FOLDER_PATH_,
    TRAIN_TEST_FOLDER_NAME_,
    SCENE_FOLDER_NAME_,
)
CROPPED_SCENE_FOLDER_PATH_ = os.path.join(
    CURR_DIR_,
    DATASET_FOLDER_NAME_,
    ITEM_FOLDER_PATH_,
    TRAIN_TEST_FOLDER_NAME_,
    CROPPED_SCENE_FOLDER_NAME_,
)
H5_FOLDER_PATH_ = os.path.join(
    CURR_DIR_,
    DATASET_FOLDER_NAME_,
    ITEM_FOLDER_PATH_,
    TRAIN_TEST_FOLDER_NAME_,
    H5_FOLDER_NAME_,
)
GT_MATRIX_FOLDER_PATH_ = os.path.join(
    CURR_DIR_,
    DATASET_FOLDER_NAME_,
    ITEM_FOLDER_PATH_,
    TRAIN_TEST_FOLDER_NAME_,
    json_setting["folder_struct"]["gt_matrix_poses_folder_name"],
)
ITEM_MODEL_FILE_PATH_ = os.path.join(
    CURR_DIR_,
    json_setting["folder_struct"]["model_folder_name"],
    ITEM_NAME_,
    json_setting["model_param"]["model_filename"]
)

# create data folder
if not os.path.exists(CROPPED_SCENE_FOLDER_PATH_):
    os.makedirs(CROPPED_SCENE_FOLDER_PATH_)

if not os.path.exists(H5_FOLDER_PATH_):
    os.makedirs(H5_FOLDER_PATH_)


def crop_pointcloud_with_raycasting(data_points, gt_matrix_file, model_file):
    # Load matrices
    try:
        Trans = np.loadtxt(gt_matrix_file)
        Trans = Trans.reshape([-1, 4, 4])
    except Exception as e:
        print(f"Error loading matrices from {gt_matrix_file}: {e}")
        return data_points, None

    # Load mesh
    try:
        mesh = trimesh.load(model_file)
    except Exception as e:
        print(f"Error loading mesh from {model_file}: {e}")
        return data_points, None

    # Build scene mesh
    meshes = []
    for M in Trans:
        m = mesh.copy()
        m.apply_transform(M)
        meshes.append(m)

    scene_mesh = trimesh.util.concatenate(meshes)

    # Ray casting
    # Points are in Camera Frame. Camera is at (0,0,0).
    points = data_points[:, :3]
    origins = np.zeros_like(points) # Rays start at origin
    vectors = points.copy() # Direction is vector from origin to point

    # Create intersector
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(scene_mesh)

    # Get all intersections
    locations, index_ray, index_tri = intersector.intersects_location(
        origins, vectors, multiple_hits=True
    )

    # We need to find the closest intersection for each ray
    # Calculate distances
    dists = np.linalg.norm(locations, axis=1)

    # Map ray index to min distance
    min_dists = {}
    for i, ray_idx in enumerate(index_ray):
        d = dists[i]
        if ray_idx not in min_dists or d < min_dists[ray_idx]:
            min_dists[ray_idx] = d

    # Filter points
    visible_indices = []
    point_dists = np.linalg.norm(points, axis=1)

    tolerance = 0.005 # 5mm tolerance

    for i in range(len(points)):
        if i in min_dists:
            hit_dist = min_dists[i]
            pt_dist = point_dists[i]

            # If hit distance is close to point distance (within tolerance), it's visible (hit itself)
            # If hit distance is significantly smaller, it's occluded
            if hit_dist >= pt_dist - tolerance:
                visible_indices.append(i)
        else:
            # Ray didn't hit anything? Keep it.
            visible_indices.append(i)

    visible_indices = np.array(visible_indices)
    if len(visible_indices) > 0:
        visible_points = data_points[visible_indices]
    else:
        visible_points = np.zeros((0, data_points.shape[1]))

    pcd = o3d.geometry.PointCloud()
    if len(visible_points) > 0:
        pcd.points = o3d.utility.Vector3dVector(visible_points[:, :3])

    return visible_points, pcd


def fpcc_save_h5(h5_filename, data, data_dtype="float32", label_dtype="uint8"):
    h5_fout = h5py.File(h5_filename, "w")

    p_xyz = data[..., :6]
    gid = data[:, :, -2]
    center_score = data[:, :, -1]

    h5_fout.create_dataset(
        "data", data=p_xyz, compression="gzip", compression_opts=4, dtype=data_dtype
    )
    h5_fout.create_dataset(
        "center_score",
        data=center_score,
        compression="gzip",
        compression_opts=4,
        dtype=data_dtype,
    )
    h5_fout.create_dataset(
        "gid", data=gid, compression="gzip", compression_opts=4, dtype=label_dtype
    )
    h5_fout.close()


def samples(data, sample_num_point, dim=None):
    if dim is None:
        dim = data.shape[-1]
    N = data.shape[0]
    order = np.arange(N)
    np.random.shuffle(order)
    data = data[order, :]
    batch_num = int(np.ceil(N / float(sample_num_point)))
    sample_datas = np.zeros((batch_num, sample_num_point, dim))

    for i in range(batch_num):
        beg_idx = i * sample_num_point
        end_idx = min((i + 1) * sample_num_point, N)
        num = end_idx - beg_idx
        sample_datas[i, 0:num, :] = data[beg_idx:end_idx, :]

        if num < sample_num_point:
            makeup_indices = np.random.choice(N, sample_num_point - num)
            sample_datas[i, num:, :] = data[makeup_indices, :]
    return sample_datas


def samples_plus_normalized(data_label, num_point=4096):
    data = data_label.copy()
    dim = data.shape[-1]

    # Calculate min/max for normalization features
    xyz_min = np.amin(data[:, 0:3], axis=0)
    data_shifted = data[:, 0:3] - xyz_min
    max_vals = np.max(data_shifted, axis=0)

    # Avoid divide by zero
    max_vals[max_vals == 0] = 1.0

    norm_xyz = data_shifted / max_vals

    # Construct extended data: [x, y, z, nx, ny, nz, idx, score]
    extended_data = np.zeros((data.shape[0], dim + 3))
    extended_data[:, 0:3] = data[:, 0:3] # Original XYZ (Camera Frame)
    extended_data[:, 3:6] = norm_xyz     # Normalized XYZ (0-1)
    extended_data[:, 6:] = data[:, 3:]   # Rest (idx, score)

    # Sample from the extended data
    data_batch = samples(extended_data, num_point)

    return data_batch


for cycle_idx in range(START_CYCLE_, MAX_CYCLE_ + 1):
    cycle_scene_path = os.path.join(SCENE_FOLDER_PATH_, "cycle_%04d" % cycle_idx)
    cycle_cropped_scene_path = os.path.join(
        CROPPED_SCENE_FOLDER_PATH_, "cycle_%04d" % cycle_idx
    )
    cycle_h5_path = os.path.join(H5_FOLDER_PATH_, "cycle_%04d" % cycle_idx)
    cycle_gt_matrix_path = os.path.join(GT_MATRIX_FOLDER_PATH_, "cycle_%04d" % cycle_idx)

    if not os.path.exists(os.path.join(cycle_cropped_scene_path)):
        os.makedirs(os.path.join(cycle_cropped_scene_path))

    if not os.path.exists(os.path.join(cycle_h5_path)):
        os.makedirs(os.path.join(cycle_h5_path))

    for item_count in range(1, MAX_DROP_ + 1):
        filename = str("%03d" % item_count) + ".txt"
        h5_filename = os.path.join(cycle_h5_path, str("%03d" % item_count) + ".h5")
        cropped_pc_filename = os.path.join(
            cycle_cropped_scene_path, str("%03d" % item_count) + ".txt"
        )
        gt_matrix_filename = os.path.join(cycle_gt_matrix_path, filename)

        point = np.loadtxt(os.path.join(cycle_scene_path, filename))

        # Use ray casting for cropping/occlusion culling
        visible_points, pcd = crop_pointcloud_with_raycasting(
            point, gt_matrix_filename, ITEM_MODEL_FILE_PATH_
        )

        data = samples_plus_normalized(visible_points, 4096)

        # save cropped scene point cloud data
        with open(cropped_pc_filename, "w") as f:
            for i in range(visible_points.shape[0]):
                f.write(
                    "%.3f %.3f %.3f %d %.3f\n"
                    % (
                        visible_points[i][0],
                        visible_points[i][1],
                        visible_points[i][2],
                        visible_points[i][3],
                        visible_points[i][4],
                    )
                )

        # save data into H5 format
        fpcc_save_h5(h5_filename, data)
        print("Saved h5 data : " + str(h5_filename))
