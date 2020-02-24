# Robomaster-ComputerVision-UCSD-2018  
## Desciption
This is the Python prototype of our computer vision code for Armor detection. The camera we used at that time has a very long exposure time, so a large part of the code focused on the handling of blurry image. Now we use this repository mostly for member training and demo purpose, as it demonstrates the basic use of OpenCV and Keras.  
  
## Structure
- `Main/main.py`: Main entry. It will start the camera and detection threads.  
- `Camera/video.py`: Camera thread. Initialize camera, calibrate image, and write to dict as shared memory.
- `Camera/calibration.py`: Calibrate image with chessboard and cache the result.
- `Aiming/detect.py` (well documented): Detection thread. It is a simple two-stage object detector that generates ROIs (based on manually designed rules for speed concern) and passes them to CNN.  
- `Aiming/classifier.py`: the CNN class for model training and predicting.

## P.S.
- The class design in `Aiming/detect.py` is not very good.
- The implementation of multi-threading is not very good.
