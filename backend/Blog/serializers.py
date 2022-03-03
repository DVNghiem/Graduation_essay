from rest_framework import serializers
from . import models
from User.serializers import ProfileSerializer, UserSerializer


class ReplySerializer(serializers.ModelSerializer):
    user = ProfileSerializer()

    class Meta:
        model = models.Reply
        fields = '__all__'


class CommentSerializer(serializers.ModelSerializer):
    user = ProfileSerializer()
    reply = ReplySerializer(many=True, read_only=True)

    class Meta:
        model = models.Comment
        fields = '__all__'


class PostSerializer(serializers.ModelSerializer):
    user = ProfileSerializer()
    comment = CommentSerializer(many=True, read_only=True)

    class Meta:
        model = models.Post
        fields = '__all__'


class PostSerializer_(serializers.ModelSerializer):
    class Meta:
        model = models.Post
        fields = ['title', 'content']
