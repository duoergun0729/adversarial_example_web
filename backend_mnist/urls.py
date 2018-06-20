from django.conf.urls import url
import backend_mnist.views as views

urlpatterns = [
    url(r'check$', views.check, ),
    url(r'drawinput_mnist$', views.drawinput_mnist, ),
]
