from django.urls import path
from . import views

urlpatterns = [
    path('add/', views.addPost),
    path('update/<slug:slug>', views.updatePost),
    path('delete/<slug:slug>', views.deletePost),
    path('get-post/<slug:slug>', views.getPost),
    path('get-all/', views.getAllPost)
]
