from django.db import models
from django.contrib.auth import get_user_model
from datetime import datetime
from django.utils.text import slugify
# Create your models here.


class Post(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    content = models.TextField()
    last_update = models.DateTimeField(default=datetime.now, blank=True)
    slug = models.SlugField()

    def save(self, *args, **kwargs):
        self.slug = slugify(
            self.title+'-'+self.last_update.strftime("%m-%d-%Y-%H-%M-%S"))
        super(Post, self).save(*args, **kwargs)

    def __str__(self) -> str:
        return self.title


class Comment(models.Model):
    user = models.ForeignKey(
        get_user_model(), on_delete=models.CASCADE)
    post = models.ForeignKey(
        Post, related_name='comment', on_delete=models.CASCADE)
    content = models.TextField()
    time = models.DateTimeField(default=datetime.now, blank=True)


class Reply(models.Model):
    user = models.ForeignKey(
        get_user_model(), on_delete=models.CASCADE)
    comment = models.ForeignKey(
        Comment, related_name='reply', on_delete=models.CASCADE)
    content = models.TextField()
    time = models.DateTimeField(default=datetime.now, blank=True)
