import cv2
import numpy as np
from cv2 import aruco
import pickle

def distance_to_charuco(
        img_path: str,
        mat: np.ndarray, dist: np.ndarray, 
        board_block_length: float,
        marker_length: float
):
    """
    Get distance to detected marker from camera

    Args:
        image(str): path of image for detecting marker
        mat(np.ndarray): camera matrix from calibration
        dist(np.ndarray): distortion coefficients from calibration
        board_block_length(float): real length of a block of chessboard (m)
        marker_length(float): real length of aruco marker (m)
    
    Returns:
        frame drawn image and distance to camera
    """
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard((5, 5), board_block_length, marker_length, aruco_dict)

    detector_params = aruco.DetectorParameters()
    charuco_params = aruco.CharucoParameters()
    charuco_detector = aruco.CharucoDetector(board, charuco_params, detector_params)

    charucoCorners, charucoIds, markerCorners, markerIds = charuco_detector.detectBoard(gray)
    if charucoCorners is None or charucoIds is None or len(charucoIds) < 4:
        return None
    
    ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
        charucoCorners, charucoIds, board, mat, dist, None, None
    )
    if not ok:
        return None
    
    cv2.drawFrameAxes(image, mat, dist, rvec, tvec, board_block_length*0.5)

    distance = float(np.linalg.norm(tvec)) #in meter
    return image, distance

if __name__ == '__main__':
    img_path = 'data/speed_estimation/drone_aruco.jpg'

    with open('data/speed_estimation/camera_calibration/drone_camera_parameters.pkl') as f:
        drone_camera_parameter = pickle.load(f)

    mat = drone_camera_parameter['camera_matrix']
    dist = drone_camera_parameter['dist_coeffs']

    image, distance = distance_to_charuco(img_path, mat, dist, )