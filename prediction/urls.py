from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('/predict', views.predict_charges, name='predict_charges'),
    path('analytics/', views.dashboard, name='analytics_dashboard'),
]
