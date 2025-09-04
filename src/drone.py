from djitellopy.tello import Tello
import numpy as np
import cv2

class Drone(Tello):
    def __init__(self):
        super.__init__()

        self.connect()
        self.streamon()
    
    def take_picture(self) -> np.ndarray:
        """
        Return frame that drone is watching now and converted to BGR
        
        Returns:
            frame (np.ndarray): The frame that drone is watching now
        """
        frame = self.get_frame_read().frame
        
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)