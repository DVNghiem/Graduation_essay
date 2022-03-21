import imp
from statistics import mode
from django.db import models
from django.contrib.auth import get_user_model
from datetime import datetime
from User.models import UserRelationship
# Create your models here.


class ChatContent(models.Model):
    content = models.TextField()
