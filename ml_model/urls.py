"""
URL configuration for ml_model app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('health/', views.health, name='health'),
]
