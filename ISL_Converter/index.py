from django.http import HttpResponse
from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def video_to_text(request):
    return render(request, 'VtoT.html')

def text_to_video(request):
    videopath=""
    if request.method == "POST":
        # print("method POST")
        keyword = request.POST.get('keyword')
        # if keyword == "help":
        #     videopath="static/Video/bgvideo.mp4"
            # print("Video found")
        if keyword == "thief":
            videopath="static/Video/thief.mp4"
        if keyword == "pain":
            videopath="static/Video/pain.mp4"  
        if keyword == "help":
            videopath="static/Video/help.mp4"       

        return render(request, 'TtoV.html', {"videopath":videopath})
    return render(request, 'TtoV.html')

def about(request):
    return render(request, 'about.html')