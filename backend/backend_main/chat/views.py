from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework import status
from rest_framework.response import Response
from . import models
from . import serializers
# Create your views here.


@api_view(['POST'])
@permission_classes([AllowAny])
def addChat(request):
    d = request.data['content']
    try:
        models.ChatContent.objects.create(
            content=d
        )
        return Response(data='oke', status=status.HTTP_200_OK)
    except:
        return Response(data='no oke', status=status.HTTP_400_BAD_REQUEST)


# @api_view(['POST'])
# @permission_classes([AllowAny])
# def updateChat(request):
#     d = request.data['content']
#     ids = request.data['id']
#     try:
#         models.ChatContent.objects.update(
#             id = ids
#             content = d
#         )
#         return Response(data='oke', status = status.HTTP_200_OK)
#     except:
#         return Response(data='no oke', status = status.HTTP_400_BAD_REQUEST)
