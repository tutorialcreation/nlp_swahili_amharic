from django.shortcuts import get_object_or_404, render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Audio, Performance
from django.core.exceptions import ObjectDoesNotExist
from jiwer import wer
import tensorflow as tf
import pickle

import os,sys
sys.path.append(os.path.abspath(os.path.join('scripts')))
sys.path.append(os.path.abspath(os.path.join('logs')))
from scripts.clean import AM_ALPHABET, EN_ALPHABET, Clean
from scripts.logger import logger
from scripts.utils import decode_batch_predictions
# Create your views here.

class FetchAudio(APIView):
    def post(self,request,*args,**kwargs):
        try:
            audio = request.data.get('audio')
        except ObjectDoesNotExist as e:
            logger.error(e)
            return Response({'error':e})
        
        Audio.objects.create(audio_file=audio)
        return Response(
            {
                'data':'succesfully passed audio'
            },
            status=status.HTTP_201_CREATED)

class FetchLanguage(APIView):
    def get(self,request,*args,**kwargs):
        alphabets={
            'swahili':EN_ALPHABET,
            'amharic':AM_ALPHABET
        }

        return Response(data=alphabets,
                        status=status.HTTP_200_OK)

class PredictView(APIView):
    model=open("models/model.pkl","rb")
    deep_model=pickle.load(model)

    def post(self,request,*args,**kwargs):
        audio_pk = request.data.get('pk')
        alphabet = request.data.get('alphabet')
        audio = get_object_or_404(Audio,pk=audio_pk)
        batch_size = 1
        cleaner = Clean()
        dataset = tf.data.Dataset.from_tensor_slices(
            (list(audio.audio_file.url))
        )
        dataset = (
            dataset.map(cleaner.convert_spectogram, num_parallel_calls=tf.data.AUTOTUNE)
            .padded_batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        predictions = []
        for batch in dataset:
            X = batch
            batch_predictions = self.deep_model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions,alphabet=alphabet)
            predictions.extend(batch_predictions)

        stringed_predictions = ' '.join(map(str,predictions))
        Performance.objects.create(audio=audio,prediction=stringed_predictions)
        return Response({
            'data':predictions
        })