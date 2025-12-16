"""
Shared Camera Configuration for Zivid 2+ MR460

All camera-related parameters should be imported from this module
to ensure consistency across the pipeline.
"""

import math
import json
import os

# --- Zivid 2+ MR460 Specs ---
# Fixed resolution
CAMERA_IMG_WIDTH = 2448  # px
CAMERA_IMG_HEIGHT = 2048  # px

# Working distance and FOV
TARGET_DISTANCE = 0.600  # meters (600mm)

def _interpolate_fov(specs: dict, target_mm: float) -> tuple:
    """
    Interpolate FOV width and height for a given working distance.
    Uses linear interpolation between the two closest defined distances.

    Args:
        specs: Dict with distance keys (str) and width/height values
        target_mm: Target distance in mm

    Returns:
        (width_m, height_m) tuple in meters
    """
    # Get sorted list of available distances (filter out non-integer keys like 'near_limit')
    available = []
    for k in specs.keys():
        try:
            available.append(int(k))
        except ValueError:
            continue
    available.sort()

    # Exact match
    if int(target_mm) in available:
        key = str(int(target_mm))
        return specs[key]["width"] / 1000.0, specs[key]["height"] / 1000.0

    # Find bracketing distances
    lower = None
    upper = None
    for d in available:
        if d < target_mm:
            lower = d
        elif d > target_mm and upper is None:
            upper = d
            break

    # Handle edge cases (out of range)
    if lower is None:
        # Below minimum - use minimum
        key = str(available[0])
        print(f"Warning: Target distance {target_mm}mm is below minimum {available[0]}mm. Using minimum.")
        return specs[key]["width"] / 1000.0, specs[key]["height"] / 1000.0
    if upper is None:
        # Above maximum - use maximum
        key = str(available[-1])
        print(f"Warning: Target distance {target_mm}mm is above maximum {available[-1]}mm. Using maximum.")
        return specs[key]["width"] / 1000.0, specs[key]["height"] / 1000.0

    # Linear interpolation
    t = (target_mm - lower) / (upper - lower)
    lower_key, upper_key = str(lower), str(upper)

    width_m = (specs[lower_key]["width"] * (1 - t) + specs[upper_key]["width"] * t) / 1000.0
    height_m = (specs[lower_key]["height"] * (1 - t) + specs[upper_key]["height"] * t) / 1000.0

    print(f"Interpolated FOV for {target_mm}mm (between {lower}mm and {upper}mm): "
          f"{width_m*1000:.1f}mm x {height_m*1000:.1f}mm")

    return width_m, height_m


# Load FOV dimensions from JSON with interpolation support
# Use abspath to ensure it works regardless of CWD
_ZIVID_SPEC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zivid", "mr_460.json")
if os.path.exists(_ZIVID_SPEC_FILE):
    with open(_ZIVID_SPEC_FILE, 'r') as f:
        _specs = json.load(f)
    FOV_WIDTH_M, FOV_HEIGHT_M = _interpolate_fov(_specs, TARGET_DISTANCE * 1000)
else:
    # Fallback: Calculate from pinhole model (less accurate)
    print("Warning: Zivid spec file not found. Using pinhole approximation.")
    # Approximate based on typical Zivid 2+ MR460 specs
    _SENSOR_WIDTH_MM = 23.0  # Typical sensor width
    _FOCAL_LENGTH_MM = 35.0  # Approximate focal length
    FOV_WIDTH_M = TARGET_DISTANCE * _SENSOR_WIDTH_MM / _FOCAL_LENGTH_MM
    FOV_HEIGHT_M = FOV_WIDTH_M * (CAMERA_IMG_HEIGHT / CAMERA_IMG_WIDTH)

# Calculate FOV angles
# FOV = 2 * arctan( (dimension/2) / distance )
CAMERA_FOV_V_RAD = 2 * math.atan((FOV_HEIGHT_M / 2) / TARGET_DISTANCE)  # Vertical FOV in radians
CAMERA_FOV_V_DEG = math.degrees(CAMERA_FOV_V_RAD)  # Vertical FOV in degrees
CAMERA_FOV_H_RAD = 2 * math.atan((FOV_WIDTH_M / 2) / TARGET_DISTANCE)   # Horizontal FOV in radians
CAMERA_FOV_H_DEG = math.degrees(CAMERA_FOV_H_RAD)  # Horizontal FOV in degrees

# Aspect ratio
CAMERA_ASPECT = CAMERA_IMG_WIDTH / CAMERA_IMG_HEIGHT

# Depth range
CAMERA_NEAR = 0.02  # meters
CAMERA_FAR = 2.0    # meters

# Camera position (should match TARGET_DISTANCE!)
CAMERA_EYE_POSITION = [0, 0, TARGET_DISTANCE]  # Camera at working distance
CAMERA_TARGET_POSITION = [0, 0, 0]
CAMERA_UP_VECTOR = [0, 1, 0]

# Camera intrinsics (for depth -> point cloud conversion)
# f_y = H / (2 * tan(FOV_V/2))
F_Y = CAMERA_IMG_HEIGHT / (2 * math.tan(CAMERA_FOV_V_RAD / 2))
F_X = F_Y  # Square pixels
C_X = CAMERA_IMG_WIDTH / 2
C_Y = CAMERA_IMG_HEIGHT / 2


def print_camera_config():
    """Print current camera configuration for debugging"""
    print("=" * 50)
    print("Camera Configuration (Zivid 2+ MR460)")
    print("=" * 50)
    print(f"Resolution: {CAMERA_IMG_WIDTH} x {CAMERA_IMG_HEIGHT}")
    print(f"Target Distance: {TARGET_DISTANCE * 1000:.0f} mm")
    print(f"FOV at target: {FOV_WIDTH_M * 1000:.0f} mm x {FOV_HEIGHT_M * 1000:.0f} mm")
    print(f"Vertical FOV: {CAMERA_FOV_V_DEG:.2f} deg")
    print(f"Horizontal FOV: {CAMERA_FOV_H_DEG:.2f} deg")
    print(f"Camera Position: {CAMERA_EYE_POSITION}")
    print(f"Intrinsics: fx={F_X:.2f}, fy={F_Y:.2f}, cx={C_X:.2f}, cy={C_Y:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    print_camera_config()
