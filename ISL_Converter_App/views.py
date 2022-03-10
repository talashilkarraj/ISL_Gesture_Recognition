from django.shortcuts import render
import time
# Create your views here.
from threading import Thread
from matplotlib.style import context
import mediapipe as mp
from django.views.decorators import gzip
from django.http.response import StreamingHttpResponse
# from ISL_Converter_App.webcamera import VideoCamera
from ISL_Converter_App.webcamtesting import VideoCamera
# Create your views here.

camera = VideoCamera()

camera.thread = Thread(target=camera.process, args=())
camera.thread.daemon = True
camera.thread.start()

def gen(request, camera):
	while True:
		frame, results = camera.process()
		y=1
		if y == 1:
			camera.thread_text = Thread(target=camera.text_p, args=(results))
			camera.thread_text.daemon = True
			camera.thread_text.start()
			y+=1
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def video_feed(request):
	return StreamingHttpResponse(gen(request, VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')


@gzip.gzip_page
def text_pred(request):
	while True:
		context = {}
		frame, results = camera.process()
		text = camera.text_p(results)
		context['text'] = text
		time.sleep(0.01)
		return render(request, 'ISL_Converter_App/templates/ISL_Converter_App/VtoT.html', context)

	