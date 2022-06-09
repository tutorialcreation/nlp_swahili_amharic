from django.shortcuts import get_object_or_404, render
from rest_framework import status,viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Audio, Performance
from .serializers import AudioFileSerializer
from jiwer import wer
#import tensorflow as tf
from pathlib import Path
import pandas as pd

import pickle
import subprocess
import os,sys

sys.path.append(os.path.abspath(os.path.join('scripts')))
sys.path.append(os.path.abspath(os.path.join('logs')))
from scripts.constants import AM_ALPHABET, EN_ALPHABET

# from scripts.deep_learner import DeepLearn
# from scripts.utils import decode_batch_predictions, vocab
# Create your views here.
import speech_recognition as sr
import torch
import zipfile
import torchaudio
from glob import glob

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
        learning_rate = 1e-4
        # char_to_num,num_to_char = vocab(alphabet=alphabet)
        # learn = DeepLearn(input_width=1, label_width=1, shift=1,epochs=5)
        # fft_length = 384
        # model = learn.build_asr_model(
        #     input_dim=fft_length // 2 + 1,
        #     output_dim=char_to_num.vocabulary_size(),
        #     lr=learning_rate
        # )
        # cleaner = Clean()
        parent_path = Path(__file__).parent.parent.parent
        audio_path = str(parent_path)+audio.audio_file.url
        # batch_size = 32
        # test_data = pd.DataFrame([{
        #                 'audio_path':audio_path,
        #                 'text':'testy testyo'
        #             }])
        # # Define the dataset
        # dataset = tf.data.Dataset.from_tensor_slices(
        #     (list(test_data['audio_path']),(list(test_data['text'])))
        # )
        # dataset = (
        #     dataset.map(cleaner.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        #     .padded_batch(batch_size)
        #     .prefetch(buffer_size=tf.data.AUTOTUNE)
        # )
    
        predictions=[]
        # for batch in dataset:
        #     X, _ = batch
        #     batch_predictions = model.predict(X)
        #     batch_predictions = decode_batch_predictions(batch_predictions,alphabet=alphabet)
        #     predictions.extend(batch_predictions)
        # stringed_predictions = ' '.join(map(str,predictions))
        # Performance.objects.create(audio=audio,prediction=stringed_predictions)
        return Response({
            'data':predictions
        })



class SPR(APIView):
    def post(self,request,*args,**kwargs):
        audio_pk = request.data.get('pk')
        alphabet = request.data.get('alphabet')
        audio = get_object_or_404(Audio,pk=audio_pk)
        device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU

        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                            model='silero_stt',
                                            language='en', # also available 'de', 'es'
                                            device=device)
        (read_batch, split_into_batches,
        read_audio, prepare_model_input) = utils  # see function signature for details

        parent_path = Path(__file__).parent.parent.parent
        audio_path = str(parent_path)+audio.audio_file.url

        # download a single file, any format compatible with TorchAudio (soundfile backend)
        torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
                               dst ='speech_orig.wav', progress=True)
        test_files = glob(audio_path)
        batches = split_into_batches(test_files, batch_size=10)
        input = prepare_model_input(read_batch(batches[0]),
                                    device=device)
        predictions=[]
        output = model(input)
        for example in output:
            predictions.append(decoder(example.cpu()))
        Performance.objects.create(audio=audio,prediction=predictions)
        return Response({
            'data':predictions
        })


class Stt(APIView):
    def post(self,request,*args,**kwargs):
        audio_pk = request.data.get('pk')
        audio = get_object_or_404(Audio,pk=audio_pk)
        parent_path = Path(__file__).parent.parent.parent
        audio_path = str(parent_path)+audio.audio_file.url
        result = subprocess.check_output(f"stt --model models/model.tflite --scorer models/swc-general.scorer --audio {audio_path}",shell=True)
        return Response({
            'prediction':result.decode('utf-8')
        })