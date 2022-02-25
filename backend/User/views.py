from django.contrib.auth import authenticate, get_user_model
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi


from .serializers import UserSerializer, LoginSerializer

User = get_user_model()


@swagger_auto_schema(method='post',
                     request_body=openapi.Schema(
                         type=openapi.TYPE_OBJECT,
                         required=['username', 'password'],
                         properties={
                             'username': openapi.Schema(type=openapi.TYPE_STRING),
                             'password': openapi.Schema(type=openapi.TYPE_STRING)
                         }

                     ), responses={
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


@swagger_auto_schema(method='post',
                     request_body=openapi.Schema(
                         title='Signin',
                         type=openapi.TYPE_OBJECT,
                         required=[
                             'username', 'password', 'first_name', 'last_name', 'email', 'avatar'],
                         properties={
                             'username': openapi.Schema(type=openapi.TYPE_STRING),
                             'password': openapi.Schema(type=openapi.TYPE_STRING),
                             'first_name': openapi.Schema(type=openapi.TYPE_STRING),
                             'last_name': openapi.Schema(type=openapi.TYPE_STRING),
                             'email': openapi.Schema(type=openapi.TYPE_STRING),
                             'avatar': openapi.Schema(type=openapi.TYPE_FILE),
                         }
                     ), responses={
                         201: "SUCCESS",
                         400: "FAIL"
                     })
@api_view(['POST'])
@permission_classes([AllowAny])
def signin(request):
    data = request.data
    try:
        user = User.objects.create_user(
            username=data['username'],
            password=data['password'],
            first_name=data['first_name'],
            last_name=data['last_name'],
            email=data['email'],
            avatar=data['avatar'],

        )
        print('okeeeeeeeeeeeee')
        user.save()
        return Response(status=status.HTTP_201_CREATED)
    except:
        return Response(status=status.HTTP_400_BAD_REQUEST)
