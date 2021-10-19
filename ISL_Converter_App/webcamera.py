import cv2
from matplotlib import pyplot as plt
import mediapipe as mp
import numpy as np
from threading import Thread
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_objectron = mp.solutions.objectron

# cap = cv2.VideoCapture(0)
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         success, image = cap.read()
#         image.flags.writeable = False
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = holistic.process(image)

#         # Draw the hand annotations on the image.
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         if results.left_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image,
#                 results.left_hand_landmarks,
#                 mp_holistic.HAND_CONNECTIONS,
#                 # mp_drawing_styles.get_default_hand_landmarks_style(),
#                 # mp_drawing_styles.get_default_hand_connections_style()
#                 mp_drawing.DrawingSpec(color=(252, 169, 3), thickness=2, circle_radius=2),
#                 mp_drawing.DrawingSpec(color=(31, 0, 209), thickness=2, circle_radius=2))

#         if results.right_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image,
#                 results.right_hand_landmarks,
#                 mp_holistic.HAND_CONNECTIONS,
#                 # mp_drawing_styles.get_default_hand_landmarks_style(),
#                 # mp_drawing_styles.get_default_hand_connections_style()
#                 mp_drawing.DrawingSpec(color=(252, 169, 3), thickness=2, circle_radius=2),
#                 mp_drawing.DrawingSpec(color=(31, 0, 209), thickness=2, circle_radius=2))
#         # Flip the image horizontally for a selfie-view display.
#         if results.face_landmarks:
#             mp_drawing.draw_landmarks(
#                 image,
#                 results.face_landmarks,
#                 mp_holistic.FACEMESH_CONTOURS,
#                 # mp_drawing_styles.get_default_hand_landmarks_style(),
#                 # mp_drawing_styles.get_default_hand_connections_style()
#                 mp_drawing.DrawingSpec(color=(252, 169, 3), thickness=1, circle_radius=1),
#                 mp_drawing.DrawingSpec(color=(227, 227, 227), thickness=1, circle_radius=1))
        
#         # cv2.imshow('Hands Keypoints', cv2.flip(image, 1))
#         # if cv2.waitKey(10) & 0xFF == ord('q'):
#         #     break

# cap.release()
# cv2.destroyAllWindows()

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.get_frame, args=())
        self.thread.daemon = True
        self.thread.start()

    # def update(self):
    #     while True:
    #         if self.capture.isOpened():
    #             (self.status, self.frame) = self.capture.read()
    #         time.sleep(self.FPS)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        with  mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                success, image = self.video.read()
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        # mp_drawing_styles.get_default_hand_landmarks_style(),
                        # mp_drawing_styles.get_default_hand_connections_style()
                        mp_drawing.DrawingSpec(color=(252, 169, 3), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(31, 0, 209), thickness=2, circle_radius=2))

                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        # mp_drawing_styles.get_default_hand_landmarks_style(),
                        # mp_drawing_styles.get_default_hand_connections_style()
                        mp_drawing.DrawingSpec(color=(252, 169, 3), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(31, 0, 209), thickness=2, circle_radius=2))
                # Flip the image horizontally for a selfie-view display.
                
                if results.face_landmarks:
                    pass
                    # mp_drawing.draw_landmarks(
                    #     image,
                    #     results.face_landmarks,
                    #     mp_holistic.FACEMESH_CONTOURS,
                    #     # mp_drawing_styles.get_default_hand_landmarks_style(),
                    #     # mp_drawing_styles.get_default_hand_connections_style()
                    #     mp_drawing.DrawingSpec(color=(252, 169, 3), thickness=1, circle_radius=1),
                    #     mp_drawing.DrawingSpec(color=(227, 227, 227), thickness=1, circle_radius=1))
                # cv2.imshow('frame', self.frame)
                # cv2.waitKey(self.FPS_MS)    
                ret, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes()
