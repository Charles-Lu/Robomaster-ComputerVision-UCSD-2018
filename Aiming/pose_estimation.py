import cv2
import numpy as np
import math


def estimate_pose(dtct, calib=None, mode="pixel"):
    x, y, w, h = dtct

    if mode == "angle":
        # vector of 3 elements for multiplying with camera matrix inverse
        box_avg = np.array([x / 512 * 1920, y / 512 * 1080, 1])

        # get camera matrix
        calibration = calib
        _, mtx, _, _, _ = calibration

        mtx_inv = np.linalg.inv(mtx)
        angle_vec = np.dot(mtx_inv, box_avg)

        # angle offset from armor plate as ( roll_offset, pitch_offset )
        pose = (math.atan(angle_vec[0]), math.atan(angle_vec[1]))
        pose = (math.degrees(pose[0]), math.degrees(pose[1]))

    else:
        pose = (-float(x - 256), float(y - 256))

    return pose

