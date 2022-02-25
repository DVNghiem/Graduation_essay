from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login),
    path('signin/', views.signin),
    path('profile/', views.profile),
    path('update/', views.update)
]
