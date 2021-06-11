from django.db import models

# Create your models here.

class Image(models.Model):
    Input_Image = models.ImageField(upload_to = 'input_imgs')

class UserDetails (models.Model):
    username = models.CharField(max_length = 50)
    first_name = models.CharField(max_length = 30)
    last_name = models.CharField(max_length = 30)
    email = models.EmailField()
    password = models.CharField(max_length = 20)
    profileimg = models.ImageField(upload_to = 'profile')