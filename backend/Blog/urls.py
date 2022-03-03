from django.urls import path
from . import views

urlpatterns = [
    # post
    path('add/', views.addPost),
    path('update/<slug:slug>', views.updatePost),
    path('delete/<slug:slug>', views.deletePost),
    path('get-post/<slug:slug>', views.getPost),
    path('get-all/', views.getAllPost),
    # comment
    path('comment/<slug:slug>/add/', views.addComment),
    path('comment/<int:id>/edit/', views.editComment),
    path('comment/<int:id>/delete/', views.deleteComment),
    # reply
    path('reply/<int:id>/add/', views.addReply),
    path('reply/<int:id>/edit/', views.editReply),
    path('reply/<int:id>/delete/', views.deleteReply),
]
