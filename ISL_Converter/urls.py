"""ISL_Converter URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from os import name
from django.contrib import admin
from django.urls import path
from ISL_Converter_App import views

from . import index

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index.home, name="home"),
    path('video_feed', views.video_feed, name="video_feed"),
    path('text_pred', views.text_pred, name="text_pred"),
    path("VtoT", index.video_to_text, name='video_to_text'),
    path("TtoV", index.text_to_video, name='text_to_video'),
    path("about", index.about, name='about'),
]
