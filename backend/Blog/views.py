from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.core.paginator import Paginator

from . import serializers
from . import models

# -----------------------------------------------------------------------


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

# -----------------------------------------------------------------------


@swagger_auto_schema(
    method='put',
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
@api_view(['PUT'])
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

# -----------------------------------------------------------------------


@swagger_auto_schema(
    method='delete',

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
@api_view(['DELETE'])
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

# -----------------------------------------------------------------------


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

# -----------------------------------------------------------------------


@swagger_auto_schema(
    method='get',
    manual_parameters=[
        openapi.Parameter('page', in_=openapi.IN_QUERY,
                          type=openapi.TYPE_INTEGER),
    ],
    responses={
        200: serializers.PostSerializer(many=True),
        400: 'Fail'
    }
)
@api_view(['GET'])
@permission_classes([AllowAny])
def getAllPost(request):
    post = models.Post.objects.all().order_by('id')
    pagination = Paginator(post, 2)
    rs = serializers.PostSerializer(
        pagination.get_page(request.GET['page']), many=True)
    return Response(data=rs.data, status=status.HTTP_200_OK)


# -----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def addComment(request, slug):
    user = request.user
    post = models.Post.objects.get(slug=slug)
    comment = models.Comment.objects.create(
        user=user,
        post=post,
        content=request.data['content']
    )
    comment.save()
    return Response(data='Success', status=status.HTTP_200_OK)
# -----------------------------------------------------------------------


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def deleteComment(request, id):
    user = request.user
    comment = models.Comment.objects.get(id=id)
    if user.id != comment.user.id:
        return Response(data='Fail', status=status.HTTP_401_UNAUTHORIZED)
    comment.delete()
    return Response(data='Success', status=status.HTTP_200_OK)

# -----------------------------------------------------------------------


@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def editComment(request, id):
    user = request.user
    comment = models.Comment.objects.get(id=id)
    if user.id != comment.user.id:
        return Response(data='Fail', status=status.HTTP_401_UNAUTHORIZED)

    rs = serializers.CommentSerializer(instance=comment, data={
        'user': user.id,
        'post': comment.post.id,
        'content': request.data['content']
    })
    if rs.is_valid(raise_exception=True):
        rs.save()
        return Response(data=rs.data, status=status.HTTP_200_OK)
    return Response(data='Fail', status=status.HTTP_400_BAD_REQUEST)

# -----------------------------------------------------------------------


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def addReply(request, id):
    user = request.user
    comment = models.Comment.objects.get(id=id)
    reply = models.Reply.objects.create(
        user=user,
        comment=comment,
        content=request.data['content']
    )
    reply.save()
    return Response(data='Success', status=status.HTTP_200_OK)


# -----------------------------------------------------------------------
@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def editReply(request, id):
    user = request.user
    reply = models.Reply.objects.filter(id=id)
    # if user.id != reply.user.id:
    #     return Response(data='Fail', status=status.HTTP_401_UNAUTHORIZED)
    reply.update(content=request.data['content'])
    return Response(data='Success', status=status.HTTP_200_OK)
    # return Response(data='Fail', status=status.HTTP_400_BAD_REQUEST)


# -----------------------------------------------------------------------


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def deleteReply(request, id):
    reply = models.Reply.objects.get(id=id)
    reply.delete()
    return Response(data='Success', status=status.HTTP_200_OK)
