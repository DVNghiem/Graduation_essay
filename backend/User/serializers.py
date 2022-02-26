from . import models
from rest_framework import serializers


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.CustomUser
        fields = '__all__'


class ProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.CustomUser
        fields = ['username', 'avatar', 'first_name',
                  'last_name', ]


class LoginSerializer(serializers.ModelSerializer):
    access_token = serializers.CharField(max_length=200)
    refresh_token = serializers.CharField(max_length=200)

    class Meta:
        model = models.CustomUser
        fields = ['username', 'avatar', 'first_name',
                  'last_name', 'access_token', 'refresh_token']
