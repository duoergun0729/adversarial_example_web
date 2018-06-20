"""recognition_attack URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from backend import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from django.views.generic import TemplateView
import backend_mnist.urls
# import backend_cifar.urls
# import backend_imagenet.urls

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^api_mnist/', include(backend_mnist.urls)),
    # url(r'^api_cifar/', include(backend_cifar.urls)),
    # url(r'^api_imagenet/', include(backend_imagenet.urls)),

    url(r'^$', TemplateView.as_view(template_name="index.html")),
]
