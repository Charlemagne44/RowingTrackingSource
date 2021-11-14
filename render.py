import cv2
import mediapipe as mp

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

    def render_text(self, image, hip_normal_angle):
        cv2.putText(image, 'hip angle: ' + str(hip_normal_angle), 
            #tuple(np.multiply(hip, [2000, 2000]).astype(int)), 
            (100,100),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA
        )
        return