from django.shortcuts import get_object_or_404, render
from rest_framework import status,viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Audio, Performance
from .serializers import AudioFileSerializer
from jiwer import wer
import tensorflow as tf
from pathlib import Path
import pandas as pd

import pickle
import subprocess
import os,sys

sys.path.append(os.path.abspath(os.path.join('scripts')))
sys.path.append(os.path.abspath(os.path.join('logs')))
from scripts.constants import AM_ALPHABET, EN_ALPHABET

from scripts.clean import Clean
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

        audio_pk = request.data.get('pk')
        alphabet = request.data.get('alphabet')
        audio = get_object_or_404(Audio,pk=audio_pk)
        with open('models/model.pkl','rb') as f:
            model = pickle.load(f)
        cleaner = Clean()
        parent_path = Path(__file__).parent.parent.parent
        audio_path = str(parent_path)+audio.audio_file.url
        batch_size = 32
        test_data = pd.DataFrame([{
                        'audio_path':audio_path,
                        'text':'testy testyo'
                    }])
        # Define the dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (list(test_data['audio_path']),(list(test_data['text'])))
        )
        dataset = (
            dataset.map(cleaner.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .padded_batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
    
        predictions=[]
        for batch in dataset:
            X, _ = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions,alphabet=alphabet)
            predictions.extend(batch_predictions)
        stringed_predictions = ' '.join(map(str,predictions))
        Performance.objects.create(audio=audio,prediction=stringed_predictions)
        return Response({
            'data':predictions
        })



