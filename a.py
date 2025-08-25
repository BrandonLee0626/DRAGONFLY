import cv2
from cv2 import aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((7, 5), 1.0, 0.8, aruco_dict)
cv2.imwrite('data/speed_estimation/camera_calibration/charuco_board.png', board.generateImage((2000, 2000)))