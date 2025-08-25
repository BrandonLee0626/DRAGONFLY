from python.drone import Drone
import cv2

drone = Drone()

cnt = 1

while True:
    frame = drone.take_picture()

    cv2.imshow('streaming', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(1) & 0xFF

    if key == 13:
        path = f'data/speed_estimation/camera_calibration/charuco_chessboard_{cnt}'
        cv2.imwrite(path, frame)
        cnt += 1

    if key == 27:
        break