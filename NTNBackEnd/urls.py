from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.conf.urls import url

from . import views

urlpatterns = [
    path('', views.login, name = 'login'),
    path('signup/', views.signup, name = 'signup'),
    path('upload/', views.upload, name = 'upload'),
    path('signup/upload.html', views.upload, name = 'upload'),
    path('upload/upload/', views.upload, name = 'upload'),
    path('signup/upload/', views.upload, name = 'upload'),
    path('result/', views.result, name = 'result')
]