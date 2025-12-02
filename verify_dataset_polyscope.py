import polyscope as ps
import numpy as np
import h5py
import os
import open3d as o3d
import csv

# Paths
DATA_ROOT = 'data_custom'
ITEM_NAME = 'teris'
CYCLE = 'cycle_0001'
SCENE_ID = '002' # Using 002 as it has 2 objects

H5_PATH = os.path.join(DATA_ROOT, ITEM_NAME, 'h5', CYCLE, f'{SCENE_ID}.h5')
GT_PATH = os.path.join(DATA_ROOT, ITEM_NAME, 'training', 'gt', CYCLE, f'{SCENE_ID}.csv')
MESH_PATH = os.path.join('model', 'teris', 'T.obj') 

def main():
    ps.init()
    ps.set_up_dir("y_up")
    # 1. Load Point Cloud from H5
    if not os.path.exists(H5_PATH):
        print(f"H5 file not found: {H5_PATH}")
        return

    print(f"Loading Point Cloud from {H5_PATH}")
    with h5py.File(H5_PATH, 'r') as f:
        points = f['data'][:]
        # labels = f['labels'][:] 

    ps_cloud = ps.register_point_cloud("Scene Point Cloud", points)
    ps_cloud.set_radius(0.002)

    # 2. Load Ground truth poses
    if not os.path.exists(GT_PATH):
        print(f"GT file not found: {GT_PATH}")
        return

    print(f"Loading GT Poses from {GT_PATH}")
    poses = []
    with open(GT_PATH, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            # id,class,x,y,z,r11,r12...
            obj_id = int(row[0])
            cls = row[1]
            tx, ty, tz = float(row[2]), float(row[3]), float(row[4])
            rot = [float(x) for x in row[5:14]]
            
            # Reconstruct Matrix
            # The CSV stores flattened rotation matrix.
            # In 1_pybullet...custom.py: R_flat = new_rot.flatten('F') (Column-Major)
            # So we need to reshape carefully.
            R = np.array(rot).reshape(3, 3, order='F')
            t = np.array([tx, ty, tz])
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            poses.append(T)

    # 3. Load Mesh
    if not os.path.exists(MESH_PATH):
        print(f"Mesh file not found: {MESH_PATH}")
        return
    
    print(f"Loading Mesh from {MESH_PATH}")
    mesh = o3d.io.read_triangle_mesh(MESH_PATH)
    if not mesh.has_triangles():
        print("Failed to load mesh or mesh has no triangles.")
        return
        
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # 4. Visualize Meshes
    for i, T in enumerate(poses):
        verts_homog = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        verts_transformed = (T @ verts_homog.T).T[:, :3]
        
        ps.register_surface_mesh(f"Object_{i+1}", verts_transformed, faces)

    print("Showing Polyscope...")
    ps.show()

if __name__ == "__main__":
    main()
