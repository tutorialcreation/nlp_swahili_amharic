from django.shortcuts import get_object_or_404, render
from rest_framework import status,viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Audio, Performance
from .serializers import AudioFileSerializer
from django.core.files.base import ContentFile
from django.core.exceptions import ObjectDoesNotExist
from jiwer import wer
import tensorflow as tf
from tensorflow import keras

import pickle

import os,sys
sys.path.append(os.path.abspath(os.path.join('scripts')))
sys.path.append(os.path.abspath(os.path.join('logs')))
from scripts.clean import AM_ALPHABET, EN_ALPHABET, Clean
from scripts.logger import logger
from scripts.deep_learner import DeepLearn
from scripts.utils import decode_batch_predictions, vocab
# Create your views here.

class AudioLoader(viewsets.ModelViewSet):
    queryset = Audio.objects.all()
    serializer_class = AudioFileSerializer    

class FetchLanguage(APIView):
    def get(self,request,*args,**kwargs):
        alphabets={
            'swahili':EN_ALPHABET,
            'amharic':AM_ALPHABET
        }

        return Response(data=alphabets,
                        status=status.HTTP_200_OK)

class PredictView(APIView):
    
    def post(self,request,*args,**kwargs):
        # saved_model=open("models/model.pickle","rb")
        
        # with saved_model as f:
        #     model=pickle.load(f)

        audio_pk = request.data.get('pk')
        alphabet = request.data.get('alphabet')
        audio = get_object_or_404(Audio,pk=audio_pk)
        char_to_num,num_to_char = vocab(alphabet=alphabet)
        batch_size = 1
        learn = DeepLearn(input_width=0, label_width=0, shift=0,epochs=0)
        fft_length = 384
        model = learn.build_asr_model(
            input_dim=fft_length // 2 + 1,
            output_dim=char_to_num.vocabulary_size(),
            rnn_units=512,
            lr=0.0001
        )

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
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions,alphabet=alphabet)
            predictions.extend(batch_predictions)

        stringed_predictions = ' '.join(map(str,predictions))
        Performance.objects.create(audio=audio,prediction=stringed_predictions)
        return Response({
            'data':predictions
        })