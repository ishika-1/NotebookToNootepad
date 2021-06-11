from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth.models import User, auth
from django.contrib.auth import authenticate, logout
from django.contrib.auth import login as auth_login
from django.contrib import messages
from django.forms import ModelForm
from django.urls import reverse

from .forms import *
from .models import *
from NTNFrontEnd.settings import MEDIA_ROOT, MEDIA_URL
import os
from django.conf import settings
from django.templatetags.static import static
from pathlib import Path

from .Model import Model
from .SpellChecker import correct_sentence
from .main import FilePaths, infer

# Create your views here.

def login (request) :
    if request.method=='POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username is None:
            return render(request, 'login.html')
        user = auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request, user)
            return HttpResponseRedirect('upload')
        else:
            messages.info(request, 'Sorry, wrong username or password. Please try again.')
            return render(request, 'login.html', {'err':'Invalid User Credentials!'})
    else :
        return render(request, 'login.html')

def signup (request):
    if (request.method=='POST'):
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        email = request.POST['email']

        if password1==password2:
            if User.objects.filter(username=username).exists():
                print('username taken')
                messages.info(request, 'Sorry, this username is taken. Please try again with another username.')
                return render (request, 'signup.html')
            elif User.objects.filter(email=email).exists():
                print('email taken')
                messages.info(request, 'Sorry, an account with this Email ID already exists. Please try again with another one, or Login with this one.')
                return render (request, 'signup.html')
            user = User.objects.create_user(username=username, password=password1, email=email, first_name=first_name, last_name=last_name)
            user.save()
            print('user created')
            return redirect('upload.html')
        else:
            print('password not matching')
            messages.info(request, 'Passwords not matching. Please try again.')
            return render (request, 'signup.html')
    else:
        return render(request,'signup.html')

def upload (request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            print("upload successful")
            return redirect('result')
    else:
        form = ImageForm()
    print('no file to be uploaded')
    return render(request,'upload.html', {'form': form})

def result (request):
    a = get_inp_img(request)
    print(a)
    model = Model(open(FilePaths.fnCharList).read(), mustRestore=False)
    string = infer(model, a)
    print(string)
    return render(request, 'result.html', {'ans':string})

def get_inp_img (request):
    path = MEDIA_ROOT + '/input_imgs'
    img_list = os.listdir(path)
    paths = sorted(Path(path).iterdir(), key=os.path.getmtime)
    paths.reverse()
    a = str(paths[0])
    return a