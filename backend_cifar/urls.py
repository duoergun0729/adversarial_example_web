from django.conf.urls import url
import backend_cifar.views as views

urlpatterns = [
    url(r'upload_cifar$', views.upload_cifar, ),
    url(r'drawinput_cifar', views.drawinput_cifar, ),
]
