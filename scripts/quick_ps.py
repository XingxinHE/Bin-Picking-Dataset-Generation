import numpy as np
import trimesh
import polyscope as ps
import os
import h5py


ps.init()
ps.set_up_dir("neg_y_up")
ps.set_front_dir("neg_z_front")


# load mesh
mesh_t = trimesh.load("model/teris/T.obj")

ps.register_surface_mesh("T", mesh_t.vertices, mesh_t.faces)

mesh_s = trimesh.load("model/teris/S.obj")
ps.register_surface_mesh("S", mesh_s.vertices, mesh_s.faces)

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
ps.show()
