import cv2
import threading
import mediapipe as mp
from threading import Thread
import time
import numpy as np
import tensorflowjs as tfjs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import backend as K

# from tensorflow.keras.callbacks import TensorBoard

# config = tf.compat.v1.ConfigProto(log_device_placement=True)
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_objectron = mp.solutions.objectron


class VideoCamera(threading.Thread):
    def __init__(self):
        self.video = cv2.VideoCapture(1000)

    def __del__(self):
        self.video.release()

    # @tf.function(experimental_relax_shapes=True)
    def process(self):
        # while True:
        with  mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            time.sleep(5)
            if self.video.isOpened():
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
            _ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes(), results

    def text_p(self, results):
        def extract_keypoints(results):
                if results.left_hand_landmarks:
                    leftHandKeypoints = np.array([[keypoint.x, keypoint.y, keypoint.z] for keypoint in results.left_hand_landmarks.landmark]).flatten()
                else:
                    leftHandKeypoints = np.zeros(21*3)

                if results.right_hand_landmarks:
                    rightHandKeypoints = np.array([[keypoint.x, keypoint.y, keypoint.z] for keypoint in results.right_hand_landmarks.landmark]).flatten()
                else:
                    rightHandKeypoints = np.zeros(21*3)
                return np.concatenate([leftHandKeypoints, rightHandKeypoints])

        with  mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            if self.video.isOpened():
                success, image = self.video.read()
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                sequence = []
                threshold = 0.7

                # Actions that we try to detect
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                actions = np.array(['help', 'thief', 'pain'])

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                model = Sequential()
                model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(2,126)))
                model.add(LSTM(128, return_sequences=True, activation='relu'))
                model.add(LSTM(64, return_sequences=False, activation='relu'))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(actions.shape[0], activation='softmax'))
                tfjs.converters.save_keras_model(model, "./")
                model.load_weights('final_accuracy.h5')

                #model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # K.clear_session()
                output_text = actions[np.argmax(res)]
                print(output_text)
                return output_text

# sequence = []
# actions = np.array(['help', 'thief', 'pain'])

# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(2,126)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# # model.fit(X_train, y_train, epochs=800, callbacks=[tb_callback])
# model.load_weights('./final_accuracy.h5')

# tfjs.converters.save_keras_model(model, "./")

