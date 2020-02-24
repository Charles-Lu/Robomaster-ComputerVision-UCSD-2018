import cv2
import time
from globals import device
from Camera.calibration import calibrate_camera, undistort


def video_capture(mem, calibrate=False, timing=False, imshow=False):
    print(">>>>>Process Video_Capture Starts.<<<<<\nCalibration:", calibrate,
          "\nVerbose:", "time;" if timing else "no time;", "image;" if imshow else "no image;")
    if calibrate:
        ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='CheckerBoard')
        mem["calibration"] = (ret, mtx, dist, rvecs, tvecs)
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    t0 = time.time()
    while True:
        _, frame = cap.read()
        if calibrate:
            frame = undistort(frame, mtx, dist)
        mem["current_frame"] = (frame, time.time())
        if timing:
            print("Capture Time: %.4fs" % (time.time() - t0))
            t0 = time.time()
        if imshow:
            cv2.imshow("VideoCapture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    test = {}
    video_capture(test, False, True, True)


