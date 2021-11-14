import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
from calcs import *
from render import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#angle histories for stroke count detection
nose_x_history = []
nose_y_history = []
frames_to_consider = 10
#number of frames for standstills
pause_frames = 40

#end of stroke switch value
prev_stroke = False
filename = 'snap.mp4'

cap = cv2.VideoCapture('Video/' + filename)
## Setup mediapipe instance
output = filename[0:-4]
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

#Get vid length to recalc speed
fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        #close video if over
        if ret == True:
        
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            #print(type(results.pose_landmarks))
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            
            # Extract landmarks
            #try:
            landmarks = results.pose_landmarks.landmark

            #detect front side
            side = Calcs().detect_side(landmarks)
            #print(side)

            # Get coordinates
            #shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            #elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            #wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            if side == "left":
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            elif side == "right":
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            else:
                print("no perceived depth")
                exit(1)

            hip_normal_angle = Calcs().angle_diff_from_normal(shoulder, hip, frame_width=frame_width, frame_height=frame_height)
            hip_normal_angle = round(hip_normal_angle)

            # add to history
            # history of last 20 frames for calculations
            if len(nose_x_history) >= frames_to_consider:
                nose_x_history.pop(-1)
                nose_x_history.insert(0,nose[0])
            else:
                nose_x_history.insert(0,nose[0])

            if len(nose_y_history) >= frames_to_consider:
                nose_y_history.pop(-1)
                nose_y_history.insert(0,nose[0])
            else:
                nose_y_history.insert(0,nose[0])
            
            end = False

            # DETERMINE IF STROKE IS AT CATCH OR FINISH
            if len(nose_x_history) >= 5:
                end = Calcs().detect_end(nose_x_history, nose_y_history)

            # Pause and overlay at end of stroke
            if not prev_stroke == end: #if change from false to true or true to false for end of stroke
                if end == True and prev_stroke != end:
                    #print('reached end')
                    #render detection of same image a few times for a standstill
                    for i in range(50):                        
                        Render().render_text(image, hip_normal_angle)                        
                        Render().render_detections(image, hip_normal_angle, results)
                        out.write(image)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            out.write(image)
                            shutil.move("C:/Users/colli/Code/RowingTrackingSource/" + output + '.avi', "C:/Users/colli/Code/RowingTrackingSource/Results/" + output + '.avi')
                            cap.release()
                            out.release()
                            cv2.destroyAllWindows()
                            exit            
            prev_stroke = end
            
            Render().render_detections(image, hip_normal_angle, results)

            #except:
            im2 = Calcs().ResizeWithAspectRatio(image, height=750)
            cv2.imshow('Mediapipe Feed', im2)
                        
            out.write(image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

shutil.move("C:/Users/colli/Code/RowingTrackingSource/" + output + '.avi', "C:/Users/colli/Code/RowingTrackingSource/Results/" + output + '.avi')