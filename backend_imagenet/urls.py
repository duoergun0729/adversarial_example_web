from django.conf.urls import url
import backend_imagenet.views as views

urlpatterns = [
    url(r'drawinput_imagenet$', views.drawinput_imagenet, ),
    url(r'upload_imagenet$', views.upload_imagenet, ),
]
