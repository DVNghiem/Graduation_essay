# Generated by Django 3.2.12 on 2022-02-26 07:23

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Blog', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='last_update',
            field=models.DateTimeField(blank=True, default=datetime.datetime.now),
        ),
        migrations.AlterField(
            model_name='post',
            name='time_upload',
            field=models.DateTimeField(blank=True, default=datetime.datetime.now),
        ),
    ]
