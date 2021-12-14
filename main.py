import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
#from moviepy.editor import *
from calcs import *
from render import *


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#angle histories for stroke count detection
frames_to_consider = 10
hip_hist = []
#shin_hist = []

#number of frames for standstills
pause_frames = 40

#end of stroke switch value
prev_end = ''
filename = 'garage.mov'

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

counter = 0
stage = 'catch'
stages = []

body_finish_window = [30, 40]
body_catch_window = [20, 30] 
shin_catch_window = [0, 10]

eval_limit = 3

calcs = Calcs()
render = Render()

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
            side = calcs.detect_side(landmarks)
            #print(side)

            # Get coordinates
            #shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            #elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            #wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y, landmarks[mp_pose.PoseLandmark.NOSE.value].z]

            # if side == "left":
            lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            #elif side == "right":
            rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
            rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
        
            #3D USE FOREFRONT SIDE
            if side == "left":
                hip_normal_angle = calcs.three_dimensional_one_side(a=lshoulder, b=lhip, c=None, frame_width=frame_width, frame_height=frame_height, norm='y')
                hip_body_angle =  calcs.three_dimensional_one_side(a=lshoulder,  b=lhip, c=lknee, frame_width=frame_width,  frame_height=frame_height, norm=None)
                shin_angle = calcs.three_dimensional_one_side(a=lknee, b = lankle, c=None, frame_width=frame_width, frame_height=frame_height, norm='y')
            else:
                hip_normal_angle = calcs.three_dimensional_one_side(a=rshoulder, b=rhip, c=None, frame_width=frame_width, frame_height=frame_height, norm='y')
                hip_body_angle =  calcs.three_dimensional_one_side(a=rshoulder,  b=rhip, c=rknee, frame_width=frame_width,  frame_height=frame_height, norm=None)
                shin_angle = calcs.three_dimensional_one_side(a=rknee, b = rankle, c=None, frame_width=frame_width, frame_height=frame_height, norm='y')

            hip_normal_angle = round(hip_normal_angle)
            hip_body_angle = round(hip_body_angle)
            shin_angle = round(shin_angle)


            if len(hip_hist) >= frames_to_consider:
                hip_hist.pop(-1)
                hip_hist.insert(0, hip_body_angle)
            else:
                hip_hist.insert(0, hip_body_angle)

            # stroke counter logic
            if hip_body_angle < 40:
                stage = "catch"
            if hip_body_angle > 110 and stage=='catch':
                stage= "finish"
                counter +=1

            #angle history end detection            

            end = calcs.end_detect(hip_body=hip_body_angle,  hip_hist=hip_hist, frames=frames_to_consider)
            
            # Pause and overlay at end of stroke if stroke changed and at end
            #print(end, prev_end)
            if end and prev_end != end:
                # run coaching func every end
                print("Stroke " + str(counter))
                calcs.coaching(body_angle=hip_normal_angle, shin_angle=shin_angle, stage=stage, body_finish_window=body_finish_window, body_catch_window=body_catch_window, shin_catch_window=shin_catch_window)

                # render the coaching texts at the ends
                for i in range(30):
                    if side == 'left':
                        if stage == 'catch':
                            render.render_text_catch(image, hip_normal_angle, shin_angle=shin_angle, frame_width=frame_width, frame_height=frame_height, hip=lhip, knee=lknee, end=stage)
                        else:
                            render.render_text_finish(image, hip_normal_angle, frame_width, frame_height, hip=lhip, end=stage)
                    else:
                        if stage == 'catch':
                            render.render_text_catch(image, hip_normal_angle, shin_angle=shin_angle, frame_width=frame_width, frame_height=frame_height, hip=rhip, knee=rknee, end=stage)                      
                        else:
                            render.render_text_finish(image, hip_normal_angle, frame_width, frame_height, hip=rhip, end=stage)
                    render.render_detections(image, hip_normal_angle, results)
                    out.write(image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        out.write(image)
                        shutil.move("C:/Users/colli/Code/RowingTrackingSource/" + output + '.avi', "C:/Users/colli/Code/RowingTrackingSource/Results/" + output + '.avi')
                        cap.release()
                        out.release()
                        cv2.destroyAllWindows()
                        exit       
            if end != None and end != False:
                prev_end = end
            
            render.render_detections(image, hip_normal_angle, results)

            #except:
            #    continue

            im2 = calcs.ResizeWithAspectRatio(image, height=700)
            cv2.imshow('Mediapipe Feed', im2)
                        
            out.write(image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            if counter > eval_limit:
                break
                
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

shutil.move("C:/Users/colli/Code/RowingTrackingSource/" + output + '.avi', "C:/Users/colli/Code/RowingTrackingSource/Results/" + output + '.avi')
#clip = VideoFileClip("C:/Users/colli/Code/RowingTrackingSource/Results/" + output + '.avi')
#clipSpeed = clip.speedx(2)
#clipSpeed.write_videofile(r"")