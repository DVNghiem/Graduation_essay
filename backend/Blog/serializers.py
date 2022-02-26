from dataclasses import field
from pyexpat import model
from rest_framework import serializers
from . import models
from User.serializers import ProfileSerializer


class PostSerializer_(serializers.ModelSerializer):
    class Meta:
        model = models.Post
        fields = ['title', 'content']


class PostSerializer(serializers.ModelSerializer):
    user = ProfileSerializer()

    class Meta:
        model = models.Post
        fields = '__all__'
