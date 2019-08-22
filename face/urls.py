from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='face-home'),
    path('predict/', views.predict, name='face-predict')
]
