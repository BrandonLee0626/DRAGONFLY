from ultralytics import YOLO
import cv2
import numpy as np

import python.helmet_detection.box as box

class HelmetDetection(YOLO):
    """
    Inheritance YOLO and add helmet detection
    """
    def __init__(self, model):
        """
        Same with constructor of YOLO

        Args:
            model (str): path of yolo model file
        """

        super().__init__(model)
        self.boxes = None

    def get_boxes(self, image: np.ndarray, threshold: float, marking: bool = False):
        """
        Predict from source and return boxes of detected people

        Args: 
            image (np.ndarray): only one ndarray image
            threshold (float): threshold of helmet detection
            marking (bool): To return image that bounding boxes are marked

        Returns:
            list of boxes of people who wear helmet, list of boxes of people who don't wear helmet, frame with box
        """

        person_bbs = list()
        helmet_bbs = list()

        with_helmets = list()
        without_helmets = list()

        for bb in super().__call__(image, verbose=False)[0].boxes:
            if int(bb.cls[0]==1):
                person_bbs.append(tuple(bb.xyxy[0].tolist()))
            else:
                helmet_bbs.append(tuple(bb.xyxy[0].tolist()))

        for person_bb in person_bbs:
            helmet_flag = False
            for helmet_bb in helmet_bbs:
                if box.intersection_area(helmet_bb, person_bb):
                    if box.intersection_area(helmet_bb, person_bb) / box.box_area(helmet_bb) > threshold:
                        helmet_flag = True
                        break
            if helmet_flag:
                with_helmets.append(person_bb)
            else:
                without_helmets.append(person_bb)

            if marking:
                image = box.draw_box(image, person_bb, caption='With Helmet' if helmet_flag else 'Without Helmet', color='b' if helmet_flag else 'r')
        
        if marking:
            return with_helmets, without_helmets, image
        
        else:
            return with_helmets, without_helmets