from dataclasses import field
from pyexpat import model
from rest_framework import serializers
from . import models

class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ChatContent
        field = '__all__'

