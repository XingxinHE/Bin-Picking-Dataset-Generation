import numpy as np
import trimesh
import polyscope as ps
import os
import h5py

# Paths
MODEL_PATH = "model/teris/T.obj"
MATRIX_FILE = "data/teris/training/gt_matrix/cycle_0001/003.txt"
PT_FILE = "data/teris/training/scene/cycle_0001/003.txt"
PT_CROPPED_FILE = "data/teris/training/pointcloud/cycle_0001/003.txt"
H5_FILE = "data/teris/training/h5/cycle_0001/003.h5"

def load_matrices(file_path):
    print(f"Loading matrices from {file_path}")
    with open(file_path, 'r') as f:
        content = f.read().strip()

    lines = content.split('\n')
    matrices = []
    current_matrix = []
    for line in lines:
        line = line.strip()
        if not line:
            if current_matrix:
                matrices.append(np.array(current_matrix))
                current_matrix = []
            continue
        try:
            current_matrix.append([float(x) for x in line.split()])
        except ValueError:
            continue

    if current_matrix:
        matrices.append(np.array(current_matrix))

    return np.array(matrices)

def main():
    # Initialize polyscope
    ps.init()
    ps.set_up_dir("z_up")

    ps.create_group("Point Cloud Full")
    ps.create_group("Point Cloud Cropped")
    ps.create_group("Point Cloud H5")

    # Add coordinate system (gumball) at origin
    # X-axis (Red)
    ps.register_curve_network("origin_x_axis",
                              nodes=np.array([[0, 0, 0], [0.1, 0, 0]]),
                              edges=np.array([[0, 1]]),
                              color=(1, 0, 0),
                              radius=0.002)
    # Y-axis (Green)
    ps.register_curve_network("origin_y_axis",
                              nodes=np.array([[0, 0, 0], [0, 0.1, 0]]),
                              edges=np.array([[0, 1]]),
                              color=(0, 1, 0),
                              radius=0.002)
    # Z-axis (Blue)
    ps.register_curve_network("origin_z_axis",
                              nodes=np.array([[0, 0, 0], [0, 0, 0.1]]),
                              edges=np.array([[0, 1]]),
                              color=(0, 0, 1),
                              radius=0.002)

    # Load mesh
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Loading mesh from {MODEL_PATH}")
    mesh = trimesh.load(MODEL_PATH)

    # Load matrices
    if not os.path.exists(MATRIX_FILE):
        print(f"Error: Matrix file not found at {MATRIX_FILE}")
        return

    matrices = load_matrices(MATRIX_FILE)
    print(f"Loaded {len(matrices)} matrices.")

    # Create collision manager
    try:
        manager = trimesh.collision.CollisionManager()
        print("Collision manager initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize CollisionManager: {e}")
        print("Falling back to visualization only.")
        manager = None

    transformed_meshes = []

    # Register all meshes
    for i, matrix in enumerate(matrices):
        # Apply transform
        tm = mesh.copy()
        tm.apply_transform(matrix)

        name = f"obj_{i}"

        # Add to polyscope
        ps.register_surface_mesh(name, tm.vertices, tm.faces)

        # Add to collision manager if available
        if manager:
            manager.add_object(name, tm)

        transformed_meshes.append(tm)

    # Check for collisions
    if manager:
        print("Checking for collisions...")
        is_collision, contact_names = manager.in_collision_internal(return_names=True)

        if is_collision:
            print("COLLISION DETECTED!")
            print(f"Found {len(contact_names)} colliding pairs.")
            # Highlight colliding objects in polyscope
            for pair in contact_names:
                print(f"Collision between {pair[0]} and {pair[1]}")
                ps.get_surface_mesh(pair[0]).set_color((1, 0, 0)) # Red
                ps.get_surface_mesh(pair[1]).set_color((1, 0, 0))
        else:
            print("No collisions detected.")

    print("Showing visualization...")


    # Load point cloud
    if not os.path.exists(PT_FILE):
        print(f"Error: Point cloud file not found at {PT_FILE}")
        return

    print(f"Loading point cloud from {PT_FILE}")
    try:
        # Format: x y z label score
        data = np.loadtxt(PT_FILE)
        if data.size > 0:
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            points = data[:, :3]
            # Register point cloud
            # Use a distinct color (e.g., Cyan) to distinguish from mesh
            pcl_full = ps.register_point_cloud("scene_pcd", points, radius=0.002, color=(0, 1, 1))
            pcl_full.add_to_group("Point Cloud Full")

            print(f"Loaded {points.shape[0]} points.")
        else:
            print("Point cloud file is empty.")
    except Exception as e:
        print(f"Error loading point cloud: {e}")


    # Load point cloud cropped
    if not os.path.exists(PT_CROPPED_FILE):
        print(f"Error: Point cloud file not found at {PT_CROPPED_FILE}")
        return

    print(f"Loading point cloud from {PT_CROPPED_FILE}")
    try:
        # Format: x y z label score
        data = np.loadtxt(PT_CROPPED_FILE)
        if data.size > 0:
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            points = data[:, :3]
            pcl_cropped = ps.register_point_cloud("scene_pcd_cropped", points, radius=0.002, color=(0, 1, 1))
            pcl_cropped.add_to_group("Point Cloud Cropped")

            print(f"Loaded {points.shape[0]} points.")
        else:
            print("Point cloud file is empty.")
    except Exception as e:
        print(f"Error loading point cloud: {e}")


    # Load pointcloud from h5
    if not os.path.exists(H5_FILE):
        print(f"Error: H5 file not found at {H5_FILE}")
        return

    print(f"Loading point cloud from {H5_FILE}")
    try:
        with h5py.File(H5_FILE, 'r') as f:
            # data shape is expected to be (N, 6) or (1, N, 6) based on generation script
            # The generation script saves: p_xyz = data[...,:6]
            # And data comes from samples_plus_normalized which returns (batch, num_point, dim)

            h5_data = f['data'][:]

            # Reshape if necessary (handle batch dimension)
            if len(h5_data.shape) == 3:
                h5_data = h5_data.reshape(-1, h5_data.shape[-1])

            points = h5_data[:, :3]

            # Register point cloud
            # Use a distinct color (e.g., Magenta)
            pcl_h5 = ps.register_point_cloud("scene_pcd_h5", points, radius=0.002, color=(1, 0, 1))
            pcl_h5.add_to_group("Point Cloud H5")

            print(f"Loaded {points.shape[0]} points from H5.")

    except Exception as e:
        print(f"Error loading H5 point cloud: {e}")


    ps.show()

if __name__ == "__main__":
    main()
