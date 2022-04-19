from django.db import models
# Create your models here.


class ChatContent(models.Model):
    content = models.TextField()
