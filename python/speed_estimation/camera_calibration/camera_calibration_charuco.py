import cv2
import os
import numpy as np
import pickle

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)

def read_chessboard(images):
    """
    Charuco base pose estimation
    Args:
        images(iterative): iterative structure(e.g. list, tuple, np.ndarray etc) with paths of images
    """
    allCorners = list ()
    allIds = list()
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    cv2.imwrite('data/speed_estimation/camera_calibration/charuco_board.png', board.draw((2000, 2000)))

    for image in images:
        frame = cv2.imread(image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3,3),
                                 zeroZone=(-1,-1),
                                 criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])
        
        decimator += 1
    
    imsize = gray.shape
    return allCorners, allIds, imsize

def calibrate_camera(allCorners, allIds, imsize):
    """
    Calibrate the camera using the detected corners
    """
    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])
    
    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

img_path = 'data/speed_estimation/camera_calibration/charuco_chessboard/'
images = [img_path+filename for filename in os.listdir(img_path)]

allCorners, allIds, imsize = read_chessboard(images)

ret, mat, dist, rvec, tvec = calibrate_camera(allCorners, allIds, imsize)

with open('data/speed_estimation/camera_calibration/drone_camera_parameters.pkl', 'wb') as f:
    pickle.dump({'camera_matrix': mat, 'dist_coeffs': dist})
