import cv2
import numpy as np
import pickle

def take_distance_aruco(frame, marker_size):
    """
    get distance to aruco marker from camera
    Args:
        frame (ndarray): frame to find aruco marker
        marker_size (int): real size of aruco marker in mm
    
    Return: real distance to aruco marker from camera in mm
    """
    with open('data/speed_estimation/camera_calibration/drone_camera_parameters.pkl', 'rb') as f:
        calibration_data = pickle.load(f)

    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
    aruco_params = cv2.aruco.DetectorParameters()
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    marker_3d_edges = np.array([
        [0, 0, 0],
        [0, marker_size, 0],
        [marker_size, marker_size, 0],
        [marker_size, 0, 0]
    ], dtype='float32').reshape((4, 1, 3))

    corners, ids, rejected = detector.detectMarkers(frame)

    if corners:
        corner = corners[0]

    else:
        return None

    ret, rvec, tvec = cv2.solvePnP(
        marker_3d_edges,
        corner,
        camera_matrix,
        dist_coeffs
    )

    if ret:
        x = round(tvec[0][0], 2)
        y = round(tvec[1][0], 2)
        z = round(tvec[2][0], 2)

    return np.linalg.norm(np.array([x, y, z]))