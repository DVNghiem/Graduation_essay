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
