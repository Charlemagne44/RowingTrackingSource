import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
from calcs import *
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#angle histories for stroke count detection
nose_x_history = []
nose_y_history = []

#end of stroke switch value
prev_stroke = False

cap = cv2.VideoCapture('Video/erg.mp4')
## Setup mediapipe instance
output = input("name of video to write: ")
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

            # Get coordinates
            #shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            #elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            #wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            # Calculate angle
            #angle = calculate_angle(shoulder, elbow, wrist)
            #relative_angle = Calcs().calculate_angle(knee, hip, shoulder)
            # remove decimal values
            #relative_angle = round(relative_angle)
            #print(relative_angle)
            # Visualize relative_angle

            hip_normal_angle = Calcs().angle_diff_from_normal(hip, shoulder, normal="vertical")
            hip_normal_angle = round(hip_normal_angle)
            #print(hip_normal_angle)

            #add to history
            # history of last 20 frames for calculations
            
            if len(nose_x_history) >= 5:
                nose_x_history.pop(-1)
                nose_x_history.insert(0,nose[0])
            else:
                nose_x_history.insert(0,nose[0])

            if len(nose_y_history) >= 5:
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
                    print('reached end')
                    
                    #render detection of same image a few times for a standstill
                    for i in range(20):
                        cv2.putText(image, 'hip angle: ' + str(hip_normal_angle), 
                            #tuple(np.multiply(hip, [2000, 2000]).astype(int)), 
                            (300, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA
                        )
                 
                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                        ) 
                        # FOR DEBUGGING
                        #im2 = cv2.resize(image, (1080, 1920))
                        im2 = Calcs().ResizeWithAspectRatio(image, height=750)
                        cv2.imshow('Mediapipe Feed', im2)
                        
                        
                        out.write(image)

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                        
            prev_stroke = end



            
            '''
            cv2.putText(image, str(relative_angle), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA
            )
            '''
            
            cv2.putText(image, 'hip angle: ' + str(hip_normal_angle), 
                #tuple(np.multiply(hip, [2000, 2000]).astype(int)), 
                (300, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA
            )
            
                    
            #except:
                #pass

            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            ) 
            # FOR DEBUGGING
            #im2 = cv2.resize(image, (1080, 1920))
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

shutil.move("C:/Users/colli/Code/VS Code Python/Rowing Tracking/" + output + '.avi', "C:/Users/colli/Code/VS Code Python/Rowing Tracking/Results/" + output + '.avi')