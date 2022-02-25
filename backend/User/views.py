from django.contrib.auth import get_user_model, authenticate
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

from .serializers import UserSerializer


@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    user = request.data
    auth = authenticate(username=user['username'],
                        password=user['password'])

    if auth is None:
        return Response(status=status.HTTP_401_UNAUTHORIZED)

    s = UserSerializer(auth)
    token = TokenObtainPairSerializer().get_token(user=auth)
    data = {
        'refresh_token': str(token),
        'access_token': str(token.access_token),
        'user': s.data
    }
    return Response(data=data, status=status.HTTP_200_OK)
