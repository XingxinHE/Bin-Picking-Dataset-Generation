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
import argparse
import multiprocessing
from functools import partial

# Import shared camera configuration
from camera_config import (
    CAMERA_IMG_WIDTH, CAMERA_IMG_HEIGHT,
    CAMERA_FOV_V_DEG, CAMERA_ASPECT,
    CAMERA_NEAR, CAMERA_FAR,
    CAMERA_EYE_POSITION, CAMERA_TARGET_POSITION, CAMERA_UP_VECTOR,
    TARGET_DISTANCE, FOV_WIDTH_M, FOV_HEIGHT_M,
    print_camera_config
)

# Global constants (will be available to workers)
CAMERA_IMG_WIDTH_ = CAMERA_IMG_WIDTH
CAMERA_IMG_HEIGHT_ = CAMERA_IMG_HEIGHT
CAMERA_FOV_ = CAMERA_FOV_V_DEG
CAMERA_ASPECT_ = CAMERA_ASPECT
CAMERA_NEAR_ = CAMERA_NEAR
CAMERA_FAR_ = CAMERA_FAR
CAMERA_EYE_POSITION_ = CAMERA_EYE_POSITION
CAMERA_TARGET_POSITION_ = CAMERA_TARGET_POSITION
CAMERA_UP_VECTOR_ = CAMERA_UP_VECTOR

def setup_env(use_gui, use_real_time):
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.87)
    p.setRealTimeSimulation(use_real_time)

    if use_gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=2.33,
            cameraYaw=0.0,
            cameraPitch=-65.0,
            cameraTargetPosition=[0.0, 0.0, -0.16],
        )


    p.setPhysicsEngineParameter(numSolverIterations=30)
    # Increase simulation substeps for better stability (prevents tunneling)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0, numSubSteps=20)

    new_position = [5.5, 5.5, 0.0]
    scale_factor = 0.5

    planeId = p.loadURDF(
        "plane.urdf",
        basePosition=new_position,
        globalScaling=scale_factor)
    print(f"Plane loaded at {new_position} with scale {scale_factor}. ID: {planeId}")
    if planeId < 0:
        raise RuntimeError("Failed to load plane.urdf!")
    textureId = p.loadTexture("assets/optical-table-texture.png")
    p.changeVisualShape(
        objectUniqueId=planeId,
        linkIndex=-1,
        textureUniqueId=textureId,
        specularColor=[0.8, 0.8, 0.8]
    )

    if use_real_time:
        p.setRealTimeSimulation(1)

    create_virtual_frustum()

def create_virtual_frustum():
    # Calculate FOV bounds at Z=0
    z_height = CAMERA_EYE_POSITION_[2]
    fov_rad = np.deg2rad(CAMERA_FOV_)
    view_height = 2 * z_height * np.tan(fov_rad / 2)
    view_width = view_height * CAMERA_ASPECT_

    # Wall thickness
    thickness = 0.1
    height = 0.5 # Wall height

    # Wall positions (centers)
    pos_top = [0, view_height/2 + thickness/2, height/2]
    pos_bottom = [0, -view_height/2 - thickness/2, height/2]
    pos_right = [view_width/2 + thickness/2, 0, height/2]
    pos_left = [-view_width/2 - thickness/2, 0, height/2]

    # Collision shapes
    col_top_bottom = p.createCollisionShape(p.GEOM_BOX, halfExtents=[view_width/2 + thickness, thickness/2, height/2])
    col_left_right = p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness/2, view_height/2, height/2])

    # Create bodies (Invisible)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_top_bottom, basePosition=pos_top)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_top_bottom, basePosition=pos_bottom)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_left_right, basePosition=pos_right)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_left_right, basePosition=pos_left)

def get_camera_image(view_matrix, proj_matrix, renderer):
    return p.getCameraImage(
        CAMERA_IMG_WIDTH_,
        CAMERA_IMG_HEIGHT_,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        lightDirection=[-0.0, -1.0, -1.0],
        lightColor=[1.0, 1.0, 1.0],
        lightDistance=2,
        shadow=1,
        renderer=renderer,
    )

def calculate_visibility_new(obj_ids, full_seg_img):
    visibility_scores = {}
    pixel_counts = {}
    max_pixels = 0

    for obj_id in obj_ids:
        count = np.sum(full_seg_img == obj_id)
        pixel_counts[obj_id] = count
        if count > max_pixels:
            max_pixels = count

    for obj_id in obj_ids:
        if max_pixels > 0:
            score = pixel_counts[obj_id] / max_pixels
        else:
            score = 0.0
        visibility_scores[obj_id] = score

    return visibility_scores

def process_cycle(cycle_idx, json_setting, args):
    # Seed random number generator for this process
    np.random.seed(cycle_idx)
    random.seed(cycle_idx)

    # Connect to PyBullet
    if args.mode == 'gui':
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    # Load EGL if requested and in direct mode
    if args.renderer == 'egl' and args.mode == 'direct':
        import pkgutil
        egl = pkgutil.get_loader('eglRenderer')
        if egl:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")

    # Compute matrices (per process to be safe)
    view_matrix_tuple = p.computeViewMatrix(
        cameraEyePosition=CAMERA_EYE_POSITION_,
        cameraTargetPosition=CAMERA_TARGET_POSITION_,
        cameraUpVector=CAMERA_UP_VECTOR_,
    )
    proj_matrix_tuple = p.computeProjectionMatrixFOV(
        CAMERA_FOV_, CAMERA_ASPECT_, CAMERA_NEAR_, CAMERA_FAR_
    )

    # Paths
    CURR_DIR_ = os.path.dirname(os.path.abspath(__file__))
    DATASET_FOLDER_NAME_ = json_setting["folder_struct"]["dataset_folder_name"]
    ITEM_NAME_ = json_setting["folder_struct"]["item_name"]
    MODEL_FOLDER_NAME_ = json_setting["folder_struct"]["model_folder_name"]

    # Use explicit model name if available, otherwise fallback to item_name (legacy behavior)
    # But main() guarantees model_name is set now.
    MODEL_NAME_ = json_setting["folder_struct"].get("model_name", "teris")

    TRAIN_TEST_FOLDER_NAME_ = json_setting["folder_struct"]["train_test_folder_name"]
    RGB_IMG_FOLDER_NAME_ = json_setting["folder_struct"]["rgb_img_folder_name"]
    DEPTH_IMG_FOLDER_NAME_ = json_setting["folder_struct"]["depth_img_folder_name"]
    SYN_SEG_IMG_FOLDER_NAME_ = json_setting["folder_struct"]["syn_seg_img_folder_name"]
    GT_POSES_FOLDER_NAME_ = json_setting["folder_struct"]["gt_poses_folder_name"]
    GT_MATRIX_POSES_FOLDER_NAME_ = json_setting["folder_struct"]["gt_matrix_poses_folder_name"]

    ITEM_FOLDER_PATH_ = os.path.join(CURR_DIR_, DATASET_FOLDER_NAME_, ITEM_NAME_)
    MODEL_FOLDER_PATH_ = os.path.join(CURR_DIR_, MODEL_FOLDER_NAME_, MODEL_NAME_)
    TRAIN_TEST_FOLDER_PATH_ = os.path.join(CURR_DIR_, DATASET_FOLDER_NAME_, ITEM_FOLDER_PATH_, TRAIN_TEST_FOLDER_NAME_)

    RGB_IMG_FOLDER_PATH_ = os.path.join(TRAIN_TEST_FOLDER_PATH_, RGB_IMG_FOLDER_NAME_)
    DEPTH_IMG_FOLDER_PATH_ = os.path.join(TRAIN_TEST_FOLDER_PATH_, DEPTH_IMG_FOLDER_NAME_)
    SEG_IMG_FOLDER_PATH_ = os.path.join(TRAIN_TEST_FOLDER_PATH_, SYN_SEG_IMG_FOLDER_NAME_)
    GT_POSES_FOLDER_PATH_ = os.path.join(TRAIN_TEST_FOLDER_PATH_, GT_POSES_FOLDER_NAME_)
    GT_MATRIX_POSES_FOLDER_PATH_ = os.path.join(TRAIN_TEST_FOLDER_PATH_, GT_MATRIX_POSES_FOLDER_NAME_)

    # Create cycle folders
    cycle_rgb_path = os.path.join(RGB_IMG_FOLDER_PATH_, "cycle_%04d" % cycle_idx)
    cycle_depth_path = os.path.join(DEPTH_IMG_FOLDER_PATH_, "cycle_%04d" % cycle_idx)
    cycle_seg_path = os.path.join(SEG_IMG_FOLDER_PATH_, "cycle_%04d" % cycle_idx)
    cycle_gt_path = os.path.join(GT_POSES_FOLDER_PATH_, "cycle_%04d" % cycle_idx)
    cycle_gt_matrix_path = os.path.join(GT_MATRIX_POSES_FOLDER_PATH_, "cycle_%04d" % cycle_idx)

    for path in [cycle_rgb_path, cycle_depth_path, cycle_seg_path, cycle_gt_path, cycle_gt_matrix_path]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    # Parameters
    MAX_DROP_ = json_setting["data_generation"]["max_drop"]
    TYPES_TERIS_ = json_setting["model_param"]["types_active"]
    DROP_X_MIN_ = -0.25
    DROP_X_MAX_ = 0.25
    DROP_Y_MIN_ = -0.2
    DROP_Y_MAX_ = 0.2
    DROP_Z_MIN_ = 0.05
    DROP_Z_MAX_ = 0.05

    heuristic_max_items = None

    # Use GUI only for first process if in GUI mode, or if running single process?
    # Actually, if multiprocessing, GUI mode is tricky. We usually assume GUI is for single process debugging.
    # For this script, we'll assume if mode=gui, workers=1.

    use_real_time = (args.mode == 'gui')

    for item_count in range(1, MAX_DROP_ + 1):
        setup_env(args.mode == 'gui', use_real_time)
        time.sleep(0.1)

        dropping_option = json_setting.get("dropping_option", "falling")
        obj_id = []
        obj_id_to_class = {}

        if dropping_option == "falling":
            for _ in range(item_count):
                class_name = random.choice(TYPES_TERIS_)
                urdf_path = os.path.join(MODEL_FOLDER_PATH_, f"{class_name}.urdf")
                if not os.path.exists(urdf_path):
                    continue

                pos = [
                    random.uniform(DROP_X_MIN_, DROP_X_MAX_),
                    random.uniform(DROP_Y_MIN_, DROP_Y_MAX_),
                    random.uniform(DROP_Z_MIN_, DROP_Z_MAX_),
                ]
                orn = p.getQuaternionFromEuler([
                    random.uniform(0.01, 0.1),
                    random.uniform(0.01, 0.1),
                    random.uniform(0.01, 3.0142)
                ])

                new_id = p.loadURDF(urdf_path, pos, orn)
                # Enable Continuous Collision Detection (CCD)
                p.changeDynamics(new_id, -1, ccdSweptSphereRadius=0.005)
                obj_id.append(new_id)
                obj_id_to_class[new_id] = class_name

                for _ in range(10):
                    p.stepSimulation()

        elif dropping_option == "packing":
            target_count = item_count
            if heuristic_max_items is not None:
                target_count = heuristic_max_items

            PACK_Z = 0.05

            for _ in range(target_count):
                class_name = random.choice(TYPES_TERIS_)
                urdf_path = os.path.join(MODEL_FOLDER_PATH_, f"{class_name}.urdf")
                if not os.path.exists(urdf_path):
                    raise ValueError(f"URDF file not found: {urdf_path}")

                max_retries = 30
                placed = False

                for _ in range(max_retries):
                    pos = [
                        random.uniform(DROP_X_MIN_, DROP_X_MAX_),
                        random.uniform(DROP_Y_MIN_, DROP_Y_MAX_),
                        PACK_Z,
                    ]
                    orn = p.getQuaternionFromEuler([
                        0, 0, random.uniform(-math.pi, math.pi)
                    ])

                    temp_id = p.loadURDF(urdf_path, pos, orn)
                    # Enable Continuous Collision Detection (CCD)
                    p.changeDynamics(temp_id, -1, ccdSweptSphereRadius=0.005)
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
                        obj_id.append(temp_id)
                        obj_id_to_class[temp_id] = class_name
                        placed = True
                        break
                    else:
                        p.removeBody(temp_id)

                if not placed:
                    break

            if len(obj_id) < target_count:
                if heuristic_max_items is None:
                    heuristic_max_items = len(obj_id)
                elif len(obj_id) < heuristic_max_items:
                    heuristic_max_items = len(obj_id)

        # Settle
        count = 0
        while count < 200:
            count += 1
            if use_real_time:
                time.sleep(1.0/240.0)
            else:
                p.stepSimulation()

        # Wait for settle
        time_start = time.time()
        while time.time() < (time_start + 3.0):
            if use_real_time:
                time.sleep(1.0/240.0)
            else:
                p.stepSimulation()

        # Capture
        renderer = p.ER_BULLET_HARDWARE_OPENGL
        # If using TinyRenderer (CPU)
        if args.renderer == 'tiny':
            renderer = p.ER_TINY_RENDERER

        images = get_camera_image(view_matrix_tuple, proj_matrix_tuple, renderer)

        # Process Images
        rgb_opengl = np.reshape(images[2], (CAMERA_IMG_HEIGHT_, CAMERA_IMG_WIDTH_, 4)) * 1.0 / 255.0
        depth_buffer_opengl = np.reshape(images[3], [CAMERA_IMG_HEIGHT_, CAMERA_IMG_WIDTH_])
        depth_opengl = CAMERA_FAR_ * CAMERA_NEAR_ / (CAMERA_FAR_ - (CAMERA_FAR_ - CAMERA_NEAR_) * depth_buffer_opengl)
        seg_opengl = np.reshape(images[4], [CAMERA_IMG_HEIGHT_, CAMERA_IMG_WIDTH_])

        vis_scores = calculate_visibility_new(obj_id, seg_opengl)

        # Save Images
        mp.imsave(os.path.join(cycle_rgb_path, "%03d_rgb.png" % item_count), rgb_opengl)

        depth_uint16 = (depth_opengl - CAMERA_NEAR_) / (CAMERA_FAR_ - CAMERA_NEAR_) * 65535
        depth_uint16 = np.clip(depth_uint16, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(cycle_depth_path, "%03d_depth.png" % item_count), depth_uint16)

        cv2.imwrite(os.path.join(cycle_seg_path, "%03d_segmentation.png" % item_count), seg_opengl.astype(np.uint16))

        # Save GT
        gt_filename = "%03d.csv" % item_count
        gt_poses_str = "id,class,x,y,z,rot_x_axis_1,rot_x_axis_2,rot_x_axis_3,rot_y_axis_1,rot_y_axis_2,rot_y_axis_3,rot_z_axis_1,rot_z_axis_2,rot_z_axis_3,visibility_score_p\n"

        correction_matrix = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        view_matrix = np.array(view_matrix_tuple).reshape(4, 4, order='F')

        visual_correction_matrix = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for idx in obj_id:
            boxPos, boxQuat = p.getBasePositionAndOrientation(idx)
            z_val = boxPos[2]
            print(f"Object {idx} ({obj_id_to_class[idx]}) World Z: {z_val:.4f}m")
            if z_val < 0.0:
                 print(f"⚠️ WARNING: Object {idx} is below table! Z={z_val:.4f}")

            world_matrix = np.eye(4)
            world_matrix[:3, :3] = np.array(p.getMatrixFromQuaternion(boxQuat)).reshape((3, 3))
            world_matrix[:3, 3] = boxPos

            cam_matrix = correction_matrix @ view_matrix @ world_matrix @ visual_correction_matrix

            new_pos = cam_matrix[:3, 3]
            new_rot_mat = cam_matrix[:3, :3]
            rot_flat = new_rot_mat.flatten()

            class_name = obj_id_to_class[idx]
            vis_p = vis_scores.get(idx, 0.0)

            gt_poses_str += f"{idx},{class_name},{new_pos[0]:.5f},{new_pos[1]:.5f},{new_pos[2]:.5f},"
            gt_poses_str += f"{rot_flat[0]:.5f},{rot_flat[1]:.5f},{rot_flat[2]:.5f},"
            gt_poses_str += f"{rot_flat[3]:.5f},{rot_flat[4]:.5f},{rot_flat[5]:.5f},"
            gt_poses_str += f"{rot_flat[6]:.5f},{rot_flat[7]:.5f},{rot_flat[8]:.5f},"
            gt_poses_str += f"{vis_p:.5f}\n"

        with open(os.path.join(cycle_gt_path, gt_filename), "w") as f:
            f.write(gt_poses_str)

    p.disconnect()
    print(f"Cycle {cycle_idx} completed.")

def main():
    parser = argparse.ArgumentParser(description="Generate Bin Picking Dataset")
    parser.add_argument("--start_cycle", type=int, default=None, help="Start cycle index")
    parser.add_argument("--end_cycle", type=int, default=None, help="End cycle index")
    parser.add_argument("--mode", type=str, default="direct", choices=["gui", "direct"], help="Simulation mode")
    parser.add_argument("--renderer", type=str, default="tiny", choices=["tiny", "egl"], help="Renderer type")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")

    # New arguments for decoupling
    parser.add_argument("--max_drop", type=int, default=None, help="Max items to drop")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset (folder name)")
    parser.add_argument("--model_name", type=str, default="teris", help="Name of the model folder to use (default: teris)")
    parser.add_argument("--object_types", nargs='+', default=None, help="List of object types to drop")
    parser.add_argument("--dropping", type=str, default=None, choices=["falling", "packing"], help="Dropping style")

    args = parser.parse_args()

    # Load settings
    with open("data_generation_setting.json") as f:
        json_setting = json.load(f)

    # CLI Overrides
    if args.start_cycle is not None:
        json_setting["data_generation"]["start_cycle"] = args.start_cycle
    if args.end_cycle is not None:
        json_setting["data_generation"]["end_cycle"] = args.end_cycle
    if args.max_drop is not None:
        json_setting["data_generation"]["max_drop"] = args.max_drop

    if args.dataset_name is not None:
        json_setting["folder_struct"]["item_name"] = args.dataset_name

    # Set model name (source of URDFs)
    # If not in json, we use the default from CLI ("teris") or whatever is in json if exists
    # But json usually assumes item_name == model_name. We splits it now.
    json_setting["folder_struct"]["model_name"] = args.model_name

    # Handle object types
    if args.object_types is not None:
        json_setting["model_param"]["types_active"] = args.object_types
    else:
        # Fallback logic
        # We should use model_name to find types, not dataset_name
        model_name = json_setting["folder_struct"]["model_name"]

        # Try to find specific key
        specific_key = f"types_{model_name}"
        if specific_key in json_setting["model_param"]:
             json_setting["model_param"]["types_active"] = json_setting["model_param"][specific_key]
        elif "types_teris" in json_setting["model_param"]:
             # Default fallback if model-specific types not found
             json_setting["model_param"]["types_active"] = json_setting["model_param"]["types_teris"]
        else:
             # Just use whatever is available if it matches?
             raise ValueError(f"No object types defined for {model_name}")

    if args.dropping is not None:
        json_setting["dropping_option"] = args.dropping

    start_cycle = json_setting["data_generation"]["start_cycle"]
    end_cycle = json_setting["data_generation"]["end_cycle"]

    print_camera_config()
    print(f"Generating cycles {start_cycle} to {end_cycle}")
    print(f"Dataset: {json_setting['folder_struct']['item_name']}")
    print(f"Mode: {args.mode}, Renderer: {args.renderer}, Workers: {args.workers}")
    print(f"Dropping: {json_setting['dropping_option']}, Max Drop: {json_setting['data_generation']['max_drop']}")
    print(f"Object Types: {json_setting['model_param']['types_active']}")

    cycles = list(range(start_cycle, end_cycle + 1))

    if args.workers > 1:
        with multiprocessing.Pool(processes=args.workers) as pool:
            func = partial(process_cycle, json_setting=json_setting, args=args)
            pool.map(func, cycles)
    else:
        for cycle in cycles:
            process_cycle(cycle, json_setting, args)

if __name__ == "__main__":
    main()
