import cv2
import mediapipe as mp
from PIL import Image, ImageFont, ImageDraw 
import numpy as np

class Render:

    

    def render_detections(self, image, hip_normal_angle, results):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        ) 
        return


    '''
    render different coaching data on screen
    show hip angle, shin angle
    show whether or not this is in or out of window
    '''    
    def render_text_catch(self, image, hip_normal_angle, shin_angle, frame_width, frame_height, hip, knee, end):
        area = frame_height * frame_width
        font_size = area / 1500000
        hip = np.array(hip)
        knee = np.array(knee)
        hip[0] *= frame_width
        hip[1] *= frame_height
        knee[0] *= frame_width
        knee[1] *= frame_height
        cv2.putText(img=image, text=end + ' angle: ' + str(hip_normal_angle), 
            #tuple(np.multiply(hip, [2000, 2000]).astype(int)), 
            org=(int(hip[0]) - int(frame_width / 10), int(hip[1])),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        cv2.putText(img=image, text='shin angle: ' + str(shin_angle), 
            #tuple(np.multiply(hip, [2000, 2000]).astype(int)), 
            org=(int(knee[0]) - int(frame_width / 10), int(knee[1])),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        return
    
    def render_text_finish(self, image, hip_normal_angle, frame_width, frame_height, hip, end):
        area = frame_height * frame_width
        font_size = area / 1500000
        hip = np.array(hip)
        hip[0] *= frame_width
        hip[1] *= frame_height
        cv2.putText(img=image, text=end + ' angle: ' + str(hip_normal_angle), 
            #tuple(np.multiply(hip, [2000, 2000]).astype(int)), 
            org=(int(hip[0]) - int(frame_width / 10), int(hip[1])),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        return
