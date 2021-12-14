import cv2
import numpy as np
from math import sqrt
import mediapipe as mp

movement_threshold = 0.02
hip_body_catch_avg_threshold = 1
hip_body_finish_avg_threshold = 1
catch_treshold = 50
finish_threshold = 90

class Calcs:
    def calculate_angle(a,b,c):
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
            return False
        

    def end_detect(self, hip_body, hip_hist, frames):
        tot_diff = 0
        for i in range(0, len(hip_hist)  - 1):
            tot_diff += hip_hist[i] - hip_hist[i+1]
        
        #print("hip body: ", hip_body)
        avg = tot_diff / frames
        #print(abs(avg))
        if hip_body > finish_threshold:
            if abs(avg) < hip_body_finish_avg_threshold:
                return 'finish'
        elif hip_body < catch_treshold:
            if abs(avg) < hip_body_catch_avg_threshold:
                return 'catch'
        else:
            return False
    

    #DEPRECATED
    def detect_end(self, knee_hist, shoulder_hist, frame_width, frame_height):
        #print(knee_hist[0])
        #print(shoulder_hist[0])

        #unsclased euclidian distance
        delta1 = sqrt((knee_hist[0][0] - shoulder_hist[0][0])**2 + (knee_hist[0][1] - shoulder_hist[0][1])**2 + (knee_hist[0][2] - shoulder_hist[0][2])**2)
        delta2 = sqrt((knee_hist[-1][0] - shoulder_hist[-1][0])**2 + (knee_hist[-1][1] - shoulder_hist[-1][1])**2 + (knee_hist[-1][2] - shoulder_hist[-1][2])**2)
        
        '''
        for coords in knee_hist:
            coords[0] *= frame_width
            coords[1] *= frame_height
            coords[2] *= ((frame_height + frame_width) / 2)

        for coords in shoulder_hist:
            coords[0] *= frame_width
            coords[1] *= frame_height
            coords[2] *= ((frame_height + frame_width) / 2)
        '''

        #scaled euclidian distance
        #sdelta1 = sqrt((knee_hist[0][0] - shoulder_hist[0][0])**2 + (knee_hist[0][1] - shoulder_hist[0][1])**2 + (knee_hist[0][2] - shoulder_hist[0][2])**2)
        #sdelta2 = sqrt((knee_hist[-1][0] - shoulder_hist[-1][0])**2 + (knee_hist[-1][1] - shoulder_hist[-1][1])**2 + (knee_hist[-1][2] - shoulder_hist[-1][2])**2)
        
        #print("unscaled euclidian: ", delta1)
        #print("scaled euclidian: " , sdelta1)
        
        #calculate avg x y z difference for first and last frame of range
        #first_avg = ((knee_hist[0][0] - shoulder_hist[0][0]) + (knee_hist[0][1] - shoulder_hist[0][1]) + (knee_hist[0][2] - shoulder_hist[0][2])) / 3
        #last_avg = ((knee_hist[-1][0] - shoulder_hist[-1][0]) + (knee_hist[-1][1] - shoulder_hist[-1][1]) + (knee_hist[-1][2] - shoulder_hist[-1][2])) / 3

        #print(first_avg)
 
    def two_dimensional_one_side(self, a, b, frame_width, frame_height, norm):
        a = np.array(a) # First shoulder
        b = np.array(b) # Mid hip
        #rescaling for aspect ratio
        a[0] *= frame_width
        b[0] *= frame_width
        a[1] *= frame_height
        b[1] *= frame_height
        if norm == "y":
            c = [b[0], 0] #normal
        elif norm == "x":
            c = [0, b[0]]
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

    
    def three_dimensional_one_side(self, a, b, c, frame_width, frame_height, norm):
        a = np.array(a)
        b = np.array(b)

        a[0] *= frame_width
        b[0] *= frame_width
        a[1] *= frame_height
        b[1] *= frame_height

        if c != None:
            c = np.array(c)
            c[0] *= frame_width
            c[1] *= frame_height
        else:
            if norm == 'y':
                c = [b[0], 0, b[2]] #normal
            elif norm == 'x':
                c = [0, b[1], b[2]]
        #rescaling for aspect ratio
        
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        #print("angle: ", np.degrees(angle))
        return np.degrees(angle)    

    def three_dimensional_angle(self, La, Lb, Ra, Rb, frame_width, frame_height):
        La = np.array(La) # First shoulder
        Lb = np.array(Lb) # Mid hip
        Ra = np.array(Ra) # First shoulder
        Rb = np.array(Rb) # Mid hip 
        #rescaling for aspect ratio

        #taking average of left and right joints and reassigning them to L values as those are used for calculations
        La[0] = La[0] + Ra[0] / 2
        La[1] = La[1] + Ra[1] / 2
        La[2] = La[2] + Ra[2] / 2
        Lb[0] = Lb[0] + Rb[0] / 2
        Lb[1] = Lb[1] + Rb[1] / 2
        Lb[2] = Lb[2] + Rb[2] / 2

        La[0] *= frame_width
        Lb[0] *= frame_width
        La[1] *= frame_height
        Lb[1] *= frame_height
        #La[2] *= frame_height
        #Lb[2] *= frame_width

        c = [Lb[0], 0, Lb[2]] #normal
        c = np.array(c)

        ba = La - Lb
        bc = c - Lb

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        print("angle: ", np.degrees(angle))
        return np.degrees(angle)

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

    '''
    if the body and shin angles are outside of the perscribed windows, then mark them as an out of window stroke
    return a dictionary of different coached aspect of each end of the stroke
    '''
    def coaching(self, body_angle, shin_angle, stage, body_finish_window, body_catch_window, shin_catch_window):
        if stage == 'finish':
            if body_angle < body_finish_window[0] or body_angle > body_finish_window[1]:
                print("finish body angle: " + str(body_angle) + " out of finish body angle range " + str(body_finish_window))
                return
            else:
                print("finish body angle: " + str(body_angle) + " in finish body angle range " + str(body_finish_window))
                return
        else:
            if body_angle < body_catch_window[0] or body_angle > body_catch_window[1]:
                print("catch body angle: " + str(body_angle) + " out of catch body angle range " + str(body_catch_window))
            else:
                print("catch body angle: " + str(body_angle) + " in catch body angle range " + str(body_catch_window))
            if shin_angle < shin_catch_window[0] or shin_angle > shin_catch_window[1]:
                print("catch shin angle: " + str(shin_angle) + " out of catch shin angle range " + str(shin_catch_window))
                return
            else:
                print("catch shin angle: " + str(shin_angle) + " in catch shin angle range " + str(shin_catch_window))
        return

