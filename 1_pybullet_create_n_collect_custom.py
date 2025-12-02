import pybullet as p
import time
import pybullet_data
import random
import os
import matplotlib.image as mp
import numpy as np
import json
import cv2

# load data & environment generation setting from json file
f = open("data_generation_setting.json")
json_setting = json.load(f)

# '''pybullet env parameters'''
useGUI = True
useRealTimeSimulation = 1  # 0 will freeze the simualtion?
TIMESTEP_ = 1.0 / 240.0  # Time in seconds.

if useGUI:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

# ''' virtual camera parameter in pybullet'''
CAMERA_IMG_WIDTH_ = 2448  # px
CAMERA_IMG_HEIGHT_ = 2048  # px

CAMERA_FOV_ = 43  # Zivid 2+ MR60 vertical FOV approx 43 degrees
CAMERA_ASPECT_ = (
    CAMERA_IMG_WIDTH_ / CAMERA_IMG_HEIGHT_
)  # describes the camera aspect ratio
CAMERA_NEAR_ = 0.02
CAMERA_FAR_ = 2.0  # describe the minimum and maximum distance which the camera will render objects
CAMERA_EYE_POSITION_ = [
    0,
    0,
    0.65,
]  # physical location of the camera in x, y, and z coordinates (65cm)
CAMERA_TARGET_POSITION_ = [
    0,
    0,
    0,
]  # the point that we wish the camera to face. [0, 0, 0] is origin
CAMERA_UP_VECTOR_ = [0, 1, 0]  # describe the orientation of the camera

CAMERA_VIEW_MATRIX_ = p.computeViewMatrix(
    cameraEyePosition=CAMERA_EYE_POSITION_,
    cameraTargetPosition=CAMERA_TARGET_POSITION_,
    cameraUpVector=CAMERA_UP_VECTOR_,
)
CAMERA_PROJ_MATRIX_ = p.computeProjectionMatrixFOV(
    CAMERA_FOV_, CAMERA_ASPECT_, CAMERA_NEAR_, CAMERA_FAR_
)


# '''Box/Container parameters'''
# BOX_MODEL_PATH_ = "model/tote_box/tote_box.urdf" # Removed for optical table setup
# BOX_WIDTH_X_ = 0.6  # meters
# BOX_WIDTH_Y_ = 0.4
# BOX_SCALING_ = 2.0  # adjust scaling factor if box is too small

# ''' Dropping parameters'''
ITEM_MODEL_PATH_ = "model/teris/T.urdf"
# Drop range adjusted for Zivid 2+ MR60 FOV (approx 58x47cm)
# We will place items within this range
DROP_X_MIN_ = -0.25
DROP_X_MAX_ = 0.25
DROP_Y_MIN_ = -0.2
DROP_Y_MAX_ = 0.2
# Z height for placement (just above table surface at z=0)
DROP_Z_MIN_ = 0.05
DROP_Z_MAX_ = 0.05

# ''' data collection cycle and drop setting '''
START_CYCLE_ = json_setting["data_generation"][
    "start_cycle"
]  # starting count of cycle to run
MAX_CYCLE_ = json_setting["data_generation"][
    "end_cycle"
]  # maximum count of cycle to run
MAX_DROP_ = json_setting["data_generation"][
    "max_drop"
]  # max number of item drop in each cycle

# '''path for save img data'''
# HXX: Changed to data_custom to avoid overwriting
DATASET_FOLDER_NAME_ = "data_custom" 
ITEM_NAME_ = json_setting["folder_struct"]["item_name"]
TRAIN_TEST_FOLDER_NAME_ = json_setting["folder_struct"]["train_test_folder_name"]
RGB_IMG_FOLDER_NAME_ = json_setting["folder_struct"]["rgb_img_folder_name"]
DEPTH_IMG_FOLDER_NAME_ = json_setting["folder_struct"]["depth_img_folder_name"]
SYN_SEG_IMG_FOLDER_NAME_ = json_setting["folder_struct"]["syn_seg_img_folder_name"]
GT_POSES_FOLDER_NAME_ = json_setting["folder_struct"]["gt_poses_folder_name"]
GT_MATRIX_POSES_FOLDER_NAME_ = json_setting["folder_struct"][
    "gt_matrix_poses_folder_name"
]

CURR_DIR_ = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_PATH_ = os.path.join(CURR_DIR_, DATASET_FOLDER_NAME_)
ITEM_FOLDER_PATH_ = os.path.join(CURR_DIR_, DATASET_FOLDER_NAME_, ITEM_NAME_)
TRAIN_TEST_FOLDER_PATH_ = os.path.join(
    CURR_DIR_, DATASET_FOLDER_NAME_, ITEM_FOLDER_PATH_, TRAIN_TEST_FOLDER_NAME_
)
RGB_IMG_FOLDER_PATH_ = os.path.join(
    CURR_DIR_,
    DATASET_FOLDER_NAME_,
    ITEM_FOLDER_PATH_,
    TRAIN_TEST_FOLDER_NAME_,
    RGB_IMG_FOLDER_NAME_,
)
DEPTH_IMG_FOLDER_PATH_ = os.path.join(
    CURR_DIR_,
    DATASET_FOLDER_NAME_,
    ITEM_FOLDER_PATH_,
    TRAIN_TEST_FOLDER_NAME_,
    DEPTH_IMG_FOLDER_NAME_,
)
SEG_IMG_FOLDER_PATH_ = os.path.join(
    CURR_DIR_,
    DATASET_FOLDER_NAME_,
    ITEM_FOLDER_PATH_,
    TRAIN_TEST_FOLDER_NAME_,
    SYN_SEG_IMG_FOLDER_NAME_,
)
GT_POSES_FOLDER_PATH_ = os.path.join(
    CURR_DIR_,
    DATASET_FOLDER_NAME_,
    ITEM_FOLDER_PATH_,
    TRAIN_TEST_FOLDER_NAME_,
    GT_POSES_FOLDER_NAME_,
)
GT_MATRIX_POSES_FOLDER_PATH_ = os.path.join(
    CURR_DIR_,
    DATASET_FOLDER_NAME_,
    ITEM_FOLDER_PATH_,
    TRAIN_TEST_FOLDER_NAME_,
    GT_MATRIX_POSES_FOLDER_NAME_,
)

# create /data/item/training folder
if not os.path.exists(RGB_IMG_FOLDER_PATH_):
    os.makedirs(RGB_IMG_FOLDER_PATH_)
if not os.path.exists(DEPTH_IMG_FOLDER_PATH_):
    os.makedirs(DEPTH_IMG_FOLDER_PATH_)
if not os.path.exists(SEG_IMG_FOLDER_PATH_):
    os.makedirs(SEG_IMG_FOLDER_PATH_)
if not os.path.exists(GT_POSES_FOLDER_PATH_):
    os.makedirs(GT_POSES_FOLDER_PATH_)


def setup_env():
    p.resetSimulation()
    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )  # default pybullet model library
    p.setGravity(0, 0, -9.87)
    p.setRealTimeSimulation(useRealTimeSimulation)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.33,
        cameraYaw=0.0,
        cameraPitch=-65.0,
        cameraTargetPosition=[0.0, 0.0, -0.16],
    )
    p.setPhysicsEngineParameter(numSolverIterations=30)
    p.setPhysicsEngineParameter(fixedTimeStep=TIMESTEP_)

    # # HXX: table
    # table_dims = [0.58, 0.47, 0.02] # Width, Length, Height (thickness)
    # table_half_extents = [d/2 for d in table_dims]
    # colId = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
    # visId = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half_extents)
    # planeId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colId, baseVisualShapeIndex=visId, basePosition=[0, 0, -table_half_extents[2]])


    new_position = [5.5, 5.5, 0.0]
    scale_factor = 0.5
    #current_orientation = p.getQuaternionFromEuler()

    planeId = p.loadURDF(
        "plane.urdf",
        basePosition=new_position,
        #baseOrientation=current_orientation,
        globalScaling=scale_factor)
    textureId = p.loadTexture("assets/optical-table-texture.png")
    p.changeVisualShape(
        objectUniqueId=planeId,
        linkIndex=-1,
        textureUniqueId=textureId,
        specularColor=[0.8, 0.8, 0.8]
    )

    if useRealTimeSimulation:
        p.setRealTimeSimulation(1)


for cycle_idx in range(START_CYCLE_, MAX_CYCLE_ + 1):
    # create sub cycle folders
    cycle_rgb_path = os.path.join(RGB_IMG_FOLDER_PATH_, "cycle_%04d" % cycle_idx)
    cycle_depth_path = os.path.join(DEPTH_IMG_FOLDER_PATH_, "cycle_%04d" % cycle_idx)
    cycle_seg_path = os.path.join(SEG_IMG_FOLDER_PATH_, "cycle_%04d" % cycle_idx)
    cycle_gt_path = os.path.join(GT_POSES_FOLDER_PATH_, "cycle_%04d" % cycle_idx)
    cycle_gt_matrix_path = os.path.join(
        GT_MATRIX_POSES_FOLDER_PATH_, "cycle_%04d" % cycle_idx
    )

    if not os.path.exists(os.path.join(cycle_rgb_path)):
        os.makedirs(os.path.join(cycle_rgb_path))

    if not os.path.exists(os.path.join(cycle_depth_path)):
        os.makedirs(os.path.join(cycle_depth_path))

    if not os.path.exists(os.path.join(cycle_seg_path)):
        os.makedirs(os.path.join(cycle_seg_path))

    if not os.path.exists(os.path.join(cycle_gt_path)):
        os.makedirs(os.path.join(cycle_gt_path))

    if not os.path.exists(os.path.join(cycle_gt_matrix_path)):
        os.makedirs(os.path.join(cycle_gt_matrix_path))

    for item_count in range(1, MAX_DROP_ + 1):
        # '''reset the environemnt'''
        setup_env()

        # '''place a box at the middle'''
        # Removed box for optical table setup
        # boxStartPos = [0, 0, 0.01]
        # boxStartOrientation = p.getQuaternionFromEuler([1.571, 0, 0])
        # boxId = p.loadURDF(
        #     BOX_MODEL_PATH_,
        #     boxStartPos,
        #     boxStartOrientation,
        #     useFixedBase=1,
        #     globalScaling=BOX_SCALING_,
        # )
        # boxPos, boxQuat = p.getBasePositionAndOrientation(boxId)
        time.sleep(0.1)

        # '''start the dropping loop'''
        obj_id = []
        count = 0
        for count in range(1, item_count + 1):
            pose = [
                random.uniform(DROP_X_MIN_, DROP_X_MAX_),
                random.uniform(DROP_Y_MIN_, DROP_Y_MAX_),
                random.uniform(DROP_Z_MIN_, DROP_Z_MAX_),
            ]
            orientation = p.getQuaternionFromEuler(
                # rotate the rpy dropping to the plane: random.uniform(0.01, 3.0142)
                # I don't consider drop right now. Just randomly in xy plane
                [
                    random.uniform(0.01, 0.1),
                    random.uniform(0.01, 0.1),
                    random.uniform(0.01, 3.0142)
                ]
            )
            obj_id.append(p.loadURDF(ITEM_MODEL_PATH_, pose, orientation))
            time.sleep(0.25)  # to prevent all objects drop at the same time
            count += 1
            images = p.getCameraImage(
                CAMERA_IMG_WIDTH_,
                CAMERA_IMG_HEIGHT_,
                viewMatrix=CAMERA_VIEW_MATRIX_,
                projectionMatrix=CAMERA_PROJ_MATRIX_,
                lightDirection=[-0.0, -1.0, -1.0],
                lightColor=[1.0, 1.0, 1.0],
                lightDistance=2,
                shadow=1,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            if useRealTimeSimulation:
                time.sleep(TIMESTEP_)
            else:
                p.stepSimulation()

        print("End of cycle %04d_%03d" % (cycle_idx, item_count))
        print("Total item drop:" + str(len(obj_id)))

        # '''give it some time (3sec) to let the physics settle down'''
        time_start = time.time()
        while time.time() < (time_start + 3.0):
            images = p.getCameraImage(
                CAMERA_IMG_WIDTH_,
                CAMERA_IMG_HEIGHT_,
                viewMatrix=CAMERA_VIEW_MATRIX_,
                projectionMatrix=CAMERA_PROJ_MATRIX_,
                lightDirection=[-0.0, -1.0, -1.0],
                lightColor=[1.0, 1.0, 1.0],
                lightDistance=2,
                shadow=1,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            if useRealTimeSimulation:
                time.sleep(TIMESTEP_)
            else:
                p.stepSimulation()

        # ''' end of this dropping cycle, start of data saving process '''
        # ''' image convertion '''
        rgb_opengl = (
            np.reshape(images[2], (CAMERA_IMG_HEIGHT_, CAMERA_IMG_WIDTH_, 4))
            * 1.0
            / 255.0
        )
        depth_buffer_opengl = np.reshape(
            images[3], [CAMERA_IMG_HEIGHT_, CAMERA_IMG_WIDTH_]
        )
        depth_opengl = (
            CAMERA_FAR_
            * CAMERA_NEAR_
            / (CAMERA_FAR_ - (CAMERA_FAR_ - CAMERA_NEAR_) * depth_buffer_opengl)
        )
        
        # HXX: Save depth as uint16
        # Normalize to 0-65535
        depth_uint16 = (depth_opengl - CAMERA_NEAR_) / (CAMERA_FAR_ - CAMERA_NEAR_) * 65535
        depth_uint16 = np.clip(depth_uint16, 0, 65535).astype(np.uint16)

        # HXX: Save raw segmentation
        seg_raw = np.reshape(images[4], [CAMERA_IMG_HEIGHT_, CAMERA_IMG_WIDTH_]).astype(np.uint16)

        # ''' save rgb,depth,seg images '''
        # RGB is fine as is (matplotlib saves as png)
        mp.imsave(
            os.path.join(cycle_rgb_path, str("%03d_rgb.png" % item_count)), rgb_opengl
        )
        
        # Save depth as uint16 png using cv2
        cv2.imwrite(
            os.path.join(cycle_depth_path, str("%03d_depth.png" % item_count)),
            depth_uint16,
        )
        
        # Save segmentation as raw png using cv2
        cv2.imwrite(
            os.path.join(cycle_seg_path, str("%03d_segmentation.png" % item_count)),
            seg_raw,
        )

        # ''' save each cycle with different number of object poses (target object only, without bin pose) into .csv '''
        # ''' format : id,class,x,y,z,rot_x_axis_1,rot_x_axis_2,rot_x_axis_3,rot_y_axis_1,rot_y_axis_2,rot_y_axis_3,rot_z_axis_1,rot_z_axis_2,rot_z_axis_3,visibility_score_o,visibility_score_p '''

        gt_filename = str("%03d" % item_count) + ".csv"
        gt_csv_str = "id,class,x,y,z,rot_x_axis_1,rot_x_axis_2,rot_x_axis_3,rot_y_axis_1,rot_y_axis_2,rot_y_axis_3,rot_z_axis_1,rot_z_axis_2,rot_z_axis_3,visibility_score_o,visibility_score_p\n"
        
        # Get View Matrix (World -> Camera)
        # PyBullet returns column-major list
        view_matrix = np.array(CAMERA_VIEW_MATRIX_).reshape(4, 4, order='F')
        
        obj_idx_counter = 1
        for idx in obj_id:
            boxPos, boxQuat = p.getBasePositionAndOrientation(idx)
            
            # Get object matrix (Model -> World)
            # p.getMatrixFromQuaternion returns row-major list of 9 floats
            rot_mat = np.array(p.getMatrixFromQuaternion(boxQuat)).reshape(3, 3)
            obj_matrix = np.eye(4)
            obj_matrix[:3, :3] = rot_mat
            obj_matrix[:3, 3] = boxPos
            
            # Transform to Camera Frame: T_cam = T_view * T_world
            # Note: PyBullet matrices are column-major, but numpy is row-major.
            # If view_matrix is already correct 4x4 numpy array (row-major storage of the matrix), we can multiply.
            # Let's verify view_matrix construction.
            # If CAMERA_VIEW_MATRIX_ is [col1, col2, col3, col4] flattened.
            # reshape(4,4, order='F') will put col1 into first column. Correct.
            
            obj_matrix_cam = np.matmul(view_matrix, obj_matrix)
            
            # Extract new pos and rot
            new_pos = obj_matrix_cam[:3, 3]
            new_rot = obj_matrix_cam[:3, :3]
            
            # Flatten column-major (Fortran style) to get x-axis, y-axis, z-axis vectors sequentially
            R_flat = new_rot.flatten('F')
            
            # In Camera frame, Z is negative?
            # PyBullet camera looks down -Z axis usually?
            # Let's check CAMERA_TARGET_POSITION_ = [0,0,0], EYE = [0,0,0.65].
            # View matrix transforms [0,0,0] to [0,0,-0.65] (approx).
            # So objects on table (z=0) should have negative Z in camera frame.
            # But depth is positive distance.
            # H5DataGenerator expects Z to be positive depth?
            # "Zcs = (clip_start + ...)" -> Zcs is positive.
            # "Xcs = -(us - cx) * Zcs / fx"
            # This implies standard computer vision frame: Z forward (positive), X right, Y down.
            # PyBullet view matrix usually aligns with OpenGL: Z backward (positive towards viewer), Y up.
            # So PyBullet Camera Frame: -Z is forward.
            # If H5DataGenerator expects Z forward (positive), we might need to flip axes.
            # Let's check H5DataGenerator:
            # Zcs is calculated from depth buffer. Depth buffer in OpenGL is non-linear Z.
            # The formula converts it to linear Z (distance from camera plane).
            # This Zcs is positive.
            # If PyBullet View Matrix gives negative Z for objects in front of camera (OpenGL convention),
            # then we need to flip Z (and maybe Y) to match CV convention.
            # CV Frame: X right, Y down, Z forward.
            # OpenGL Frame: X right, Y up, Z backward.
            # Transformation OpenGL -> CV: Rotate 180 deg around X axis.
            # Matrix: [[1, 0, 0], [0, -1, 0], [0, 0, -1]].
            
            # Let's apply this correction to obj_matrix_cam.
            # correction = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # obj_matrix_cv = np.matmul(correction, obj_matrix_cam)
            
            # new_pos = obj_matrix_cv[:3, 3]
            # new_rot = obj_matrix_cv[:3, :3]
            # R_flat = new_rot.flatten('F')
            
            # HXX: Debugging alignment.
            # The point cloud is generated by H5DataGenerator using:
            # Xcs = -(us - cx) * Zcs / fx
            # Ycs = -(vs - cy) * Zcs / fy
            # This implies:
            # +X in point cloud corresponds to -u (left in image)
            # +Y in point cloud corresponds to -v (up in image)
            # +Z is depth (forward)
            # So Point Cloud Frame is: X-Left, Y-Up, Z-Forward.
            
            # PyBullet View Matrix (OpenGL):
            # X-Right, Y-Up, Z-Backward (Camera looks down -Z)
            
            # We need to transform PyBullet Pose (OpenGL) to Point Cloud Frame (X-Left, Y-Up, Z-Forward).
            # OpenGL: [r, u, b] (right, up, back)
            # Target: [l, u, f] (left, up, forward)
            # Transformation:
            # X_new = -X_old
            # Y_new = Y_old
            # Z_new = -Z_old
            # Matrix: [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
            
            correction = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            obj_matrix_cv = np.matmul(correction, obj_matrix_cam)
            
            new_pos = obj_matrix_cv[:3, 3]
            new_rot = obj_matrix_cv[:3, :3]
            R_flat = new_rot.flatten('F')
            
            gt_csv_str += f"{obj_idx_counter},{ITEM_NAME_},{new_pos[0]:.8f},{new_pos[1]:.8f},{new_pos[2]:.8f},"
            gt_csv_str += ",".join([f"{val:.8f}" for val in R_flat])
            gt_csv_str += ",1.0,1.0\n" # Visibility scores
            
            obj_idx_counter += 1

        # write csv
        with open(os.path.join(cycle_gt_path, gt_filename), "w") as f:
            f.write(gt_csv_str)

