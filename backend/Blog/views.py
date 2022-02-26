from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from . import serializers
from . import models


@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        title='Add Post',
        type=openapi.TYPE_OBJECT,
        properties={
            'title': openapi.Schema(type=openapi.TYPE_STRING),
            'content': openapi.Schema(type=openapi.TYPE_STRING)
        }
    ),
    manual_parameters=[
        openapi.Parameter('Authorization', in_=openapi.IN_HEADER,
                          type=openapi.TYPE_STRING),
    ],
    responses={
        200: 'SUCCESS',
        400: 'FAIL'
    }
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def addPost(request):
    data = request.data
    try:
        post = models.Post.objects.create(
            user=request.user,
            title=data['title'],
            content=data['content']
        )
        post.save()
        return Response(data='SUCCESS', status=status.HTTP_200_OK)
    except:
        return Response(data='FAIL', status=status.HTTP_400_BAD_REQUEST)


@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        title='Update Post',
        type=openapi.TYPE_OBJECT,
        properties={
            'title': openapi.Schema(type=openapi.TYPE_STRING),
            'content': openapi.Schema(type=openapi.TYPE_STRING)
        }
    ),
    manual_parameters=[
        openapi.Parameter('Authorization', in_=openapi.IN_HEADER,
                          type=openapi.TYPE_STRING),
        openapi.Parameter('slug', in_=openapi.IN_PATH,
                          type=openapi.TYPE_STRING)
    ],
    responses={
        200: 'SUCCESS',
        400: 'FAIL'
    }
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def updatePost(request, slug):
    post = models.Post.objects.get(slug=slug)
    if request.user.id != post.user.id:
        return Response(data='FAIL', status=status.HTTP_400_BAD_REQUEST)
    rs = serializers.PostSerializer_(instance=post, data=request.data)
    if rs.is_valid():
        rs.save()
        return Response(data='SUCCESS', status=status.HTTP_200_OK)
    return Response(data='FAIL', status=status.HTTP_400_BAD_REQUEST)


@swagger_auto_schema(
    method='post',

    manual_parameters=[
        openapi.Parameter('slug', in_=openapi.IN_PATH,
                          type=openapi.TYPE_STRING),
        openapi.Parameter('Authorization',
                          in_=openapi.IN_HEADER, type=openapi.TYPE_STRING)
    ],
    responses={
        200: 'SUCCESS',
        400: 'FAIL'
    }
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def deletePost(request, slug):
    try:
        post = models.Post.objects.get(slug=slug)
        if request.user.id != post.user.id:
            return Response(data='FAIL', status=status.HTTP_400_BAD_REQUEST)
        post.delete()
        return Response(data='SUCCESS', status=status.HTTP_200_OK)
    except:
        return Response(data='FAIL', status=status.HTTP_400_BAD_REQUEST)


@swagger_auto_schema(
    method='get',
    manual_parameters=[
        openapi.Parameter('slug', in_=openapi.IN_PATH,
                          type=openapi.TYPE_STRING)
    ],
    responses={
        200: serializers.PostSerializer,
        404: 'Not found'
    }
)
@api_view(['GET'])
@permission_classes([AllowAny])
def getPost(request, slug):
    try:
        post = models.Post.objects.get(slug=slug)
        rs = serializers.PostSerializer(post)
        return Response(data=rs.data, status=status.HTTP_200_OK)
    except:
        return Response(data='Not found', status=status.HTTP_404_NOT_FOUND)


@swagger_auto_schema(
    method='get',
    responses={
        200: serializers.PostSerializer(many=True),
        400: 'Fail'
    }
)
@api_view(['get'])
@permission_classes([AllowAny])
def getAllPost(request):
    post = models.Post.objects.all()
    rs = serializers.PostSerializer(post, many=True)
    return Response(data=rs.data, status=status.HTTP_200_OK)
