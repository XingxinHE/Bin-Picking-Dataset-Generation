import pybullet as p
import time
import pybullet_data
import random
import os
import matplotlib.image as mp
import numpy as np
import json
import cv2
import math

# Import shared camera configuration
from camera_config import (
    CAMERA_IMG_WIDTH, CAMERA_IMG_HEIGHT,
    CAMERA_FOV_V_DEG, CAMERA_ASPECT,
    CAMERA_NEAR, CAMERA_FAR,
    CAMERA_EYE_POSITION, CAMERA_TARGET_POSITION, CAMERA_UP_VECTOR,
    TARGET_DISTANCE, FOV_WIDTH_M, FOV_HEIGHT_M,
    print_camera_config
)

# Print camera config for verification
print_camera_config()

# Alias for backward compatibility
CAMERA_IMG_WIDTH_ = CAMERA_IMG_WIDTH
CAMERA_IMG_HEIGHT_ = CAMERA_IMG_HEIGHT
CAMERA_FOV_ = CAMERA_FOV_V_DEG
CAMERA_ASPECT_ = CAMERA_ASPECT
CAMERA_NEAR_ = CAMERA_NEAR
CAMERA_FAR_ = CAMERA_FAR
CAMERA_EYE_POSITION_ = CAMERA_EYE_POSITION
CAMERA_TARGET_POSITION_ = CAMERA_TARGET_POSITION
CAMERA_UP_VECTOR_ = CAMERA_UP_VECTOR

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

CAMERA_VIEW_MATRIX_ = p.computeViewMatrix(
    cameraEyePosition=CAMERA_EYE_POSITION_,
    cameraTargetPosition=CAMERA_TARGET_POSITION_,
    cameraUpVector=CAMERA_UP_VECTOR_,
)
CAMERA_PROJ_MATRIX_ = p.computeProjectionMatrixFOV(
    CAMERA_FOV_, CAMERA_ASPECT_, CAMERA_NEAR_, CAMERA_FAR_
)

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
DATASET_FOLDER_NAME_ = json_setting["folder_struct"]["dataset_folder_name"]
MODEL_FOLDER_NAME_ = json_setting["folder_struct"]["model_folder_name"]
ITEM_NAME_ = json_setting["folder_struct"]["item_name"]
TRAIN_TEST_FOLDER_NAME_ = json_setting["folder_struct"]["train_test_folder_name"]
RGB_IMG_FOLDER_NAME_ = json_setting["folder_struct"]["rgb_img_folder_name"]
DEPTH_IMG_FOLDER_NAME_ = json_setting["folder_struct"]["depth_img_folder_name"]
SYN_SEG_IMG_FOLDER_NAME_ = json_setting["folder_struct"]["syn_seg_img_folder_name"]
GT_POSES_FOLDER_NAME_ = json_setting["folder_struct"]["gt_poses_folder_name"]
GT_MATRIX_POSES_FOLDER_NAME_ = json_setting["folder_struct"][
    "gt_matrix_poses_folder_name"
]

TYPES_TERIS_ = json_setting["model_param"]["types_teris"]

CURR_DIR_ = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_PATH_ = os.path.join(CURR_DIR_, DATASET_FOLDER_NAME_)
ITEM_FOLDER_PATH_ = os.path.join(CURR_DIR_, DATASET_FOLDER_NAME_, ITEM_NAME_)
MODEL_FOLDER_PATH_ = os.path.join(CURR_DIR_, MODEL_FOLDER_NAME_, ITEM_NAME_)
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

# Load Class Mapping
CLASS_MAPPING_FILE = os.path.join(ITEM_FOLDER_PATH_, "class.json")
CLASS_MAPPING = {}
if os.path.exists(CLASS_MAPPING_FILE):
    with open(CLASS_MAPPING_FILE, 'r') as f:
        CLASS_MAPPING = json.load(f)
else:
    print("Warning: class.json not found. Defaulting to empty mapping.")

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

    new_position = [5.5, 5.5, 0.0]
    scale_factor = 0.5

    planeId = p.loadURDF(
        "plane.urdf",
        basePosition=new_position,
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

    create_virtual_frustum()


def create_virtual_frustum():
    # Calculate FOV bounds at Z=0
    # Camera is at Z = CAMERA_EYE_POSITION_[2]
    # Looking down at Z=0

    z_height = CAMERA_EYE_POSITION_[2]
    fov_rad = np.deg2rad(CAMERA_FOV_)

    # Vertical height of the view at Z=0
    # tan(fov/2) = (h/2) / z
    view_height = 2 * z_height * np.tan(fov_rad / 2)

    # Horizontal width of the view at Z=0
    view_width = view_height * CAMERA_ASPECT_

    print(f"Virtual Frustum at Z=0: Width={view_width:.3f}, Height={view_height:.3f}")

    # Create 4 walls
    # Wall thickness
    thickness = 0.1
    height = 0.5 # Wall height

    # Wall positions (centers)
    # Top (Y+)
    pos_top = [0, view_height/2 + thickness/2, height/2]
    # Bottom (Y-)
    pos_bottom = [0, -view_height/2 - thickness/2, height/2]
    # Right (X+)
    pos_right = [view_width/2 + thickness/2, 0, height/2]
    # Left (X-)
    pos_left = [-view_width/2 - thickness/2, 0, height/2]

    # Collision shapes
    # Top/Bottom walls: Width = view_width + 2*thickness, Thickness = thickness
    col_top_bottom = p.createCollisionShape(p.GEOM_BOX, halfExtents=[view_width/2 + thickness, thickness/2, height/2])

    # Left/Right walls: Length = view_height, Thickness = thickness
    col_left_right = p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness/2, view_height/2, height/2])

    # Create bodies (Invisible, so no visual shape)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_top_bottom, basePosition=pos_top)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_top_bottom, basePosition=pos_bottom)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_left_right, basePosition=pos_right)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_left_right, basePosition=pos_left)


def get_camera_image():
    return p.getCameraImage(
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

def calculate_visibility_new(obj_ids, full_seg_img):
    """
    Calculate visibility score: V_i = N_i / N_max
    where N_i is the number of pixels for object i,
    and N_max is the max number of pixels for any object in the scene.
    """
    visibility_scores = {}

    # Count pixels for each object
    pixel_counts = {}
    max_pixels = 0

    for obj_id in obj_ids:
        count = np.sum(full_seg_img == obj_id)
        pixel_counts[obj_id] = count
        if count > max_pixels:
            max_pixels = count

    # Calculate scores
    for obj_id in obj_ids:
        if max_pixels > 0:
            score = pixel_counts[obj_id] / max_pixels
        else:
            score = 0.0
        visibility_scores[obj_id] = score

    return visibility_scores


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

    heuristic_max_items = None

    for item_count in range(1, MAX_DROP_ + 1):
        # '''reset the environemnt'''
        setup_env()

        time.sleep(0.1)

        # '''start the dropping loop'''
        # ''' drop items '''
        # Get list of items to drop from settings
        dropping_option = json_setting.get("dropping_option", "falling")
        print(f"Dropping option: {dropping_option}")

        obj_id = []
        obj_id_to_class = {} # Map obj_id to class_name

        if dropping_option == "falling":
            # Falling Mode: Incremental drops, stacked vertically
            # Loop from 1 to item_count (incremental)
            # The outer loop is: for item_count in range(1, MAX_DROP_ + 1):
            # So inside here we just need to generate 'item_count' items.

            for _ in range(item_count):
                # Randomly select a class
                class_name = random.choice(TYPES_TERIS_)
                urdf_path = os.path.join(MODEL_FOLDER_PATH_, f"{class_name}.urdf")

                if not os.path.exists(urdf_path):
                    raise ValueError(f"URDF file not found: {urdf_path}")

                # Random position and orientation (falling)
                pos = [
                    random.uniform(DROP_X_MIN_, DROP_X_MAX_),
                    random.uniform(DROP_Y_MIN_, DROP_Y_MAX_),
                    random.uniform(DROP_Z_MIN_, DROP_Z_MAX_),
                ]
                orn = p.getQuaternionFromEuler(
                    [
                        random.uniform(0.01, 0.1),
                        random.uniform(0.01, 0.1),
                        random.uniform(0.01, 3.0142)
                    ]
                )

                new_id = p.loadURDF(urdf_path, pos, orn)
                obj_id.append(new_id)
                obj_id_to_class[new_id] = class_name

                # Let physics settle
                for _ in range(10):
                    p.stepSimulation()

        elif dropping_option == "packing":
            # Packing Mode: Flat placement, no overlap
            # Determine target count based on heuristic (only for packing mode)
            target_count = item_count
            if heuristic_max_items is not None:
                target_count = heuristic_max_items
                print(f"Using heuristic max items: {target_count} (instead of {item_count})")

            # Z height for flat placement
            PACK_Z = 0.05

            for _ in range(target_count):
                class_name = random.choice(TYPES_TERIS_)
                urdf_path = os.path.join(MODEL_FOLDER_PATH_, f"{class_name}.urdf")

                if not os.path.exists(urdf_path):
                    raise ValueError(f"URDF file not found: {urdf_path}")

                # Try to place without collision
                max_retries = 30
                placed = False

                for _ in range(max_retries):
                    # Random position in X/Y
                    pos = [
                        random.uniform(DROP_X_MIN_, DROP_X_MAX_),
                        random.uniform(DROP_Y_MIN_, DROP_Y_MAX_),
                        PACK_Z,
                    ]

                    # Flat orientation (random Yaw only)
                    orn = p.getQuaternionFromEuler(
                        [
                            0, # Roll = 0
                            0, # Pitch = 0
                            random.uniform(-math.pi, math.pi), # Random Yaw
                        ]
                    )

                    temp_id = p.loadURDF(urdf_path, pos, orn)
                    p.performCollisionDetection()
                    contact_points = p.getContactPoints(temp_id)

                    has_collision = False
                    for cp in contact_points:
                        # cp[2] is bodyB unique id
                        other_id = cp[2]
                        # Ignore collision with plane (which is usually the first body loaded)
                        # We can check if other_id is in obj_id list (existing objects)
                        if other_id in obj_id:
                            has_collision = True
                            break

                    if not has_collision:
                        # Valid placement
                        obj_id.append(temp_id)
                        obj_id_to_class[temp_id] = class_name
                        placed = True
                        break
                    else:
                        # Collision, remove and retry
                        p.removeBody(temp_id)

                if not placed:
                    print(f"Warning: Could not pack item {len(obj_id)+1}/{target_count}. Scene might be full.")
                    break # Stop adding items for this scene if we can't fit more

            # Optimization: Update heuristic if we couldn't place all items
            if len(obj_id) < target_count:
                if heuristic_max_items is None:
                    heuristic_max_items = len(obj_id)
                    print(f"Optimization: Packing limit reached at {heuristic_max_items}. Capping future drops.")
                elif len(obj_id) < heuristic_max_items:
                     # Should not happen if logic is correct, but just in case
                     heuristic_max_items = len(obj_id)

        else:
            print(f"Error: Unknown dropping_option: {dropping_option}")
            continue

        # Simulation loop to settle
        count = 0
        while count < 200:
            count += 1
            if useRealTimeSimulation:
                time.sleep(TIMESTEP_)
            else:
                p.stepSimulation()

        print("End of cycle %04d_%03d" % (cycle_idx, item_count))
        print("Total item drop:" + str(len(obj_id)))

        # '''give it some time (3sec) to let the physics settle down'''
        time_start = time.time()
        while time.time() < (time_start + 3.0):
            if useRealTimeSimulation:
                time.sleep(TIMESTEP_)
            else:
                p.stepSimulation()

        # ''' end of this dropping cycle, start of data saving process '''
        images = get_camera_image()

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
        seg_opengl = (
            np.reshape(images[4], [CAMERA_IMG_HEIGHT_, CAMERA_IMG_WIDTH_])
        )

        # Calculate Visibility Scores (New Logic)
        vis_scores = calculate_visibility_new(obj_id, seg_opengl)

        # ''' save rgb,depth,seg images '''
        mp.imsave(
            os.path.join(cycle_rgb_path, str("%03d_rgb.png" % item_count)), rgb_opengl
        )

        # Save depth as uint16
        # Normalize to 0-65535
        depth_uint16 = (depth_opengl - CAMERA_NEAR_) / (CAMERA_FAR_ - CAMERA_NEAR_) * 65535
        depth_uint16 = np.clip(depth_uint16, 0, 65535).astype(np.uint16)

        cv2.imwrite(
            os.path.join(cycle_depth_path, str("%03d_depth.png" % item_count)),
            depth_uint16,
        )

        # Save segmentation as uint16 (to support > 255 IDs if needed)
        # Note: IDs start from 5, so values are low (5, 6, 7...).
        # Image will appear black in standard viewers.
        cv2.imwrite(
            os.path.join(cycle_seg_path, str("%03d_segmentation.png" % item_count)),
            seg_opengl.astype(np.uint16),
        )

        # ''' save each cycle with different number of object poses (target object only, without bin pose) into .txt '''
        # ''' format : x y z quat_x quat_y quat_z quat_w'''
        # ''' format : matrix 4x4 '''

        gt_filename = str("%03d" % item_count) + ".csv" # Changed to .csv

        # CSV Header (Removed visibility_score_o)
        gt_poses_str = "id,class,x,y,z,rot_x_axis_1,rot_x_axis_2,rot_x_axis_3,rot_y_axis_1,rot_y_axis_2,rot_y_axis_3,rot_z_axis_1,rot_z_axis_2,rot_z_axis_3,visibility_score_p\n"

        # HXX: Correction Matrix (OpenGL -> OpenCV)
        # Rotate 180 deg around X axis: Y -> -Y, Z -> -Z
        correction_matrix = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        # HXX: Get View Matrix (World -> Camera)
        # PyBullet returns column-major list, reshape to 4x4
        view_matrix = np.array(CAMERA_VIEW_MATRIX_).reshape(4, 4, order='F')

        for idx in obj_id:
            boxPos, boxQuat = p.getBasePositionAndOrientation(idx)

            # Create World Matrix
            world_matrix = np.eye(4)
            world_matrix[:3, :3] = np.array(p.getMatrixFromQuaternion(boxQuat)).reshape((3, 3))
            world_matrix[:3, 3] = boxPos

            # Transform to Camera Frame
            # T_cam = T_correction * T_view * T_world * T_visual_offset
            # We need to account for the URDF visual offset (RotX 180)
            # Hack?
            visual_correction_matrix = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])

            cam_matrix = correction_matrix @ view_matrix @ world_matrix @ visual_correction_matrix

            # Extract new pos and rot
            new_pos = cam_matrix[:3, 3]
            new_rot_mat = cam_matrix[:3, :3]

            # Flatten Rotation Matrix
            rot_flat = new_rot_mat.flatten() # x1, x2, x3, y1, y2, y3, z1, z2, z3

            # Get Class Name and ID
            class_name = obj_id_to_class[idx]

            # Get Visibility
            vis_p = vis_scores.get(idx, 0.0)

            # Append to CSV String
            # id,class,x,y,z,rot...,vis_p
            gt_poses_str += f"{idx},{class_name},{new_pos[0]:.5f},{new_pos[1]:.5f},{new_pos[2]:.5f},"
            gt_poses_str += f"{rot_flat[0]:.5f},{rot_flat[1]:.5f},{rot_flat[2]:.5f},"
            gt_poses_str += f"{rot_flat[3]:.5f},{rot_flat[4]:.5f},{rot_flat[5]:.5f},"
            gt_poses_str += f"{rot_flat[6]:.5f},{rot_flat[7]:.5f},{rot_flat[8]:.5f},"
            gt_poses_str += f"{vis_p:.5f}\n"

        # write csv
        f = open(os.path.join(cycle_gt_path, gt_filename), "w")
        f.write(gt_poses_str)
        f.close()
