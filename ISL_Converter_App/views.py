from django.shortcuts import render

# Create your views here.

import cv2
from matplotlib import pyplot as plt
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_objectron = mp.solutions.objectron

from django.http.response import StreamingHttpResponse
from ISL_Converter_App.webcamera import VideoCamera
# Create your views here.

def index(request):
	return render(request, 'ISL_Converter_App/index.html')

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import TensorBoard
