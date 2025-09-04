import cv2
from cv2 import aruco
import os
import numpy as np
import pickle

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((5, 5), 1.0, 0.8, aruco_dict)
cv2.imwrite('data/speed_estimation/camera_calibration/charuco_board.png', board.generateImage((2000, 2000)))

detector_params = aruco.DetectorParameters()
charuco_params = aruco.CharucoParameters()
charuco_detector = aruco.CharucoDetector(board, charuco_params, detector_params)

def read_chessboard(images):
    """
    Charuco base pose estimation (for OpenCV 4.7+)
    Args:
        images: image path iterable
    Returns:
        allCorners: list of (N,1,2) float32
        allIds    : list of (N,1)   int32
        imsize    : (width, height)
    """
    allCorners, allIds = [], []
    im_w = im_h = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    for p in images:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if im_w is None:
            im_h, im_w = gray.shape[:2]

        charucoCorners, charucoIds, _, _ = charuco_detector.detectBoard(gray)

        if charucoCorners is not None and charucoIds is not None and len(charucoIds) >= 4:
            cv2.cornerSubPix(gray, charucoCorners, (3, 3), (-1, -1), criteria)
            allCorners.append(charucoCorners)
            allIds.append(charucoIds)

    return allCorners, allIds, (im_w, im_h)  # (width, height)

def calibrate_camera(allCorners, allIds, imsize):
    """
    Calibrate using ChArUco detections (OpenCV 4.7+)
    """
    w, h = imsize
    cameraMatrixInit = np.array([[1000.,    0., w/2.],
                                 [   0., 1000., h/2.],
                                 [   0.,    0.,   1. ]], dtype=np.float64)
    distCoeffsInit = np.zeros((5, 1), dtype=np.float64)

    flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
             cv2.CALIB_RATIONAL_MODEL |
             cv2.CALIB_FIX_ASPECT_RATIO)

    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)

    (ret, camera_matrix, dist_coeffs,
     rvecs, tvecs,
     stdDevInt, stdDevExt,
     perViewErr) = aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=board,
        imageSize=imsize,                 # (width, height)
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=term_criteria
    )

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

img_path = 'data/speed_estimation/camera_calibration/charuco_chessboard/'
images = [img_path+filename for filename in os.listdir(img_path)]

allCorners, allIds, imsize = read_chessboard(images)

ret, mat, dist, rvec, tvec = calibrate_camera(allCorners, allIds, imsize)

with open('data/speed_estimation/camera_calibration/drone_camera_parameters.pkl', 'wb') as f:
    pickle.dump({'camera_matrix': mat, 'dist_coeffs': dist})
