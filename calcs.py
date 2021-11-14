import cv2
import numpy as np
from math import sqrt
import mediapipe as mp

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

    def detect_side(self, landmarks):
        mp_pose = mp.solutions.pose
        lshoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
        rshoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
        if lshoulder < rshoulder:
            return "left"
        elif lshoulder > rshoulder:
            return "right"
        else:
            return "same"
        
    
    def detect_end(self, x_hist, y_hist):
        if x_hist[0] - x_hist[-1] < movement_threshold and x_hist[0] - x_hist[-1] > -movement_threshold: #and y_hist[0] - y_hist[-1] < movement_threshold and y_hist[0] - y_hist[-1] > -movement_threshold:
            return True
        #if sum(x_hist)/len(x_hist) <= movement_threshold and sum(y_hist)/len(y_hist):
        #    return True
        else:
            return False

    def angle_diff_from_normal(self, a, b, frame_width, frame_height):
        a = np.array(a) # First shoulder
        b = np.array(b) # Mid hip
        #rescaling for aspect ratio
        a[0] *= frame_width
        b[0] *= frame_width
        a[1] *= frame_height
        b[1] *= frame_height
        c = [b[0], 0] #normal
        #print(frame_height, frame_width)
        #print("shoulder x: ", a[0], "shoulder y ", a[1])
        #print("hip x: ", b[0], "hip y: ", b[1]) 
        #print("normal x: ", c[0], "normal y: ", c[1])  
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        print("angle: ", angle)
        return angle

    def angle_test(self, a, b):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array([b[1], 0])      
        #arccos((P12^2 + P13^2 - P23^2) / (2 * P12 * P13))
        #where P12 is the length of the segment from P1 to P2, calculated by sqrt((P1x - P2x)2 + (P1y - P2y)2)
        # 1 = b 2 = a 3 = c
        #radians = arctan2(a[1], a[0])
        #print(c[0], c[1])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

        '''
        pba = sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
        pbc = sqrt((b[0] - c[0])**2 + (c[1] - c[1])**2)
        pac = sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2)
        radians = np.arccos((pba**2 + pbc**2 - pac**2) / (2 * pba * pbc))
        
        angle = np.abs(radians*180.0/np.pi)
        if angle >180.0:
            angle = 360-angle
        
        return angle
        '''

    def three_dimensional_angle(a, b, frame_width, frame_height):
        p0 = [3.5, 6.7]
        p1 = [7.9, 8.4]
        p2 = [10.8, 4.8]

        ''' 
        compute angle (in degrees) for p0p1p2 corner
        Inputs:
            p0,p1,p2 - points in the form of [x,y]
        '''

        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)

        angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
        print np.degrees(angle)

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
