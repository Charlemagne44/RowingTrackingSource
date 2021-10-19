import cv2
import numpy as np

movement_threshold = 0.02

class Calcs:
    def calculate_angle(self, a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle
    
    def detect_end(self, x_hist, y_hist):
        if x_hist[0] - x_hist[4] < movement_threshold and x_hist[0] - x_hist[4] > -movement_threshold and y_hist[0] - y_hist[4] < movement_threshold and y_hist[0] - y_hist[4] > -movement_threshold:
            return True
        else:
            return False

    def angle_diff_from_normal(self, a, b, normal=None):
        a = np.array(a) # start
        b = np.array(b) # end
        #if normal == "horizontal":
        radians = np.arctan2(b[1] - a[1], b[0] - a[0])
        angle = 180 - np.abs(radians * 180.0/np.pi)
        if angle >180.0:
            angle = 360-angle
        
        return angle
        

    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)
