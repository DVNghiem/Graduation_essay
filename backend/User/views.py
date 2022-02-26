from email import header
from django.contrib.auth import authenticate, get_user_model
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

#
from .serializers import UserSerializer, LoginSerializer, ProfileSerializer

User = get_user_model()


@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'username': openapi.Schema(type=openapi.TYPE_STRING),
            'password': openapi.Schema(type=openapi.TYPE_STRING)
        }

    ),
    responses={
        200: openapi.Response(
            description='SUCCESS',
            schema=LoginSerializer

        ), 401: 'UNAUTHORIZED'})
@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    user = request.data
    auth = authenticate(username=user['username'],
                        password=user['password'])

    if auth is None:
        return Response(status=status.HTTP_401_UNAUTHORIZED)

    token = TokenObtainPairSerializer().get_token(user=auth)
    data = UserSerializer(instance=auth).data
    data['refresh_token'] = str(token)
    data['access_token'] = str(token.access_token)
    rs = LoginSerializer(data=data)
    rs.is_valid()
    return Response(data=rs.data, status=status.HTTP_200_OK)


@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        title='Signin',
        type=openapi.TYPE_OBJECT,
        properties={
            'username': openapi.Schema(type=openapi.TYPE_STRING),
            'password': openapi.Schema(type=openapi.TYPE_STRING),
            'first_name': openapi.Schema(type=openapi.TYPE_STRING),
            'last_name': openapi.Schema(type=openapi.TYPE_STRING),
            'email': openapi.Schema(type=openapi.TYPE_STRING),
            'avatar': openapi.Schema(type=openapi.TYPE_FILE),
        }
    ),

    responses={
        201: "SUCCESS",
        400: "FAIL"
    })
@api_view(['POST'])
@permission_classes([AllowAny])
def signin(request):
    data = request.data
    if User.objects.filter(username=data['username']).exists():
        return Response(data='Username exist', status=status.HTTP_400_BAD_REQUEST)
    if User.objects.filter(email=data['email']).exists():
        return Response(data='Email exist', status=status.HTTP_400_BAD_REQUEST)

    user = User.objects.create_user(
        username=data['username'],
        password=data['password'],
        first_name=data['first_name'],
        last_name=data['last_name'],
        email=data['email'],
        avatar=data['avatar'],

    )
    user.save()
    return Response(status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='get',
    manual_parameters=[
        openapi.Parameter(
            'Authorization', in_=openapi.IN_HEADER, type=openapi.TYPE_STRING)
    ],
    responses={
        200: ProfileSerializer
    }
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile(request):
    user = request.user
    rs = ProfileSerializer(user)
    return Response(data=rs.data, status=status.HTTP_200_OK)


@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        title='Update',
        type=openapi.TYPE_OBJECT,
        properties={
            'first_name': openapi.Schema(type=openapi.TYPE_STRING),
            'last_name': openapi.Schema(type=openapi.TYPE_STRING),
            'email': openapi.Schema(type=openapi.TYPE_STRING),
            'avatar': openapi.Schema(type=openapi.TYPE_FILE),
        }),
    manual_parameters=[
        openapi.Parameter('Authorization', in_=openapi.IN_HEADER,
                          type=openapi.TYPE_STRING)
    ]
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update(request):
    rs = ProfileSerializer(instance=request.user,
                           data=request.data, partial=True)

    if rs.is_valid():
        rs.save()
        return Response(status=status.HTTP_200_OK)
    return Response(status=status.HTTP_400_BAD_REQUEST)
