from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.exceptions import ObjectDoesNotExist
from .models import Audio
# Create your views here.
class FetchAudio(APIView):
    def post(self,request,*args,**kwargs):
        try:
            audio = request.data.get('audio')
        except ObjectDoesNotExist as e:
            return Response({'error':e})
        
        Audio.objects.create(audio_file=audio)
        return Response(
            {
                'data':'succesfully passed audio'
            },
            status=status.HTTP_201_CREATED)