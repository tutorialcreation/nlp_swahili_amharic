"""
This sript file is responsible of cleaning process of data, we will be calling functions from here!
# of Classes: 1
# of functions: 12
    openfile
    pad_trunc
    char_index
    store_audio_features
    encode_single_sample
    load_audios
    get_labels
    read_text
    read_data
    get_clean_word
    get_duration
    get_durations
    convert_channels
"""

import imp
import librosa
import numpy as np
import pandas as pd
import warnings
import random
import matplotlib.pyplot as plt
from os.path import exists
import seaborn as sns
from functools import reduce
import os,sys
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from logger import logger
from utils import vocab
import torch
import torchaudio
from tensorflow import keras
import tensorflow as tf
import random
from logger import logger
import IPython.display as ipd
import warnings
import wave, array
warnings.filterwarnings("ignore")
AM_ALPHABET='ሀለሐመሠረሰቀበግዕዝተኀነአከወዐዘየደገጠጰጸፀፈፐቈኈጐኰፙፘፚauiāeəo'
EN_ALPHABET='abcdefghijklmnopqrstuvwxyz'


class Clean:
    """
    - this class is responsible for performing 
    Cleaning Tasks
    """
    char_to_num,_=vocab(EN_ALPHABET)

    def __init__(self,df = None):
        """initialize the cleaning class"""
        self.df = df
        logger.info("Successfully initialized clean class")
        
    def openfile(self,audio_file):
        """
        - to open audio file and return the signal and sampling rate
        """
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    def pad_trunc(self,sig,sr, max_ms):
        """
        author:Edenwork
        - Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
        """
        sig, sr = sig,sr
        sig = sig.reshape(-1,1)
        sig = sig.T
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
        # Truncate the signal to the given length
            sig = sig[:,:int(max_len)]

        elif (sig_len < max_len):
        # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, int(max_len) - int(sig_len))
            pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((int(num_rows), int(pad_end_len)))

            sig = torch.cat((pad_begin,  torch.from_numpy(sig), pad_end), 1)

        return (sig, sr)

    def char_index(self,alphabet):
        a_map = {} # map letter to number
        rev_a_map = {} # map number to letter
        for i, a in enumerate(alphabet):
            a_map[a] = i
            rev_a_map[i] = a
        return rev_a_map    

    def store_audio_features(self,y,sr):
        """
        author: Martin Luther
        function: returns different features from the audio
        """
        
        y, sr = y,sr
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        lc = {
            "rmse":np.mean(librosa.feature.rms(y=y)),
            "chroma_stft":np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            "spec_cent":np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "spec_bw":np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            "rolloff":np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            "zcr":np.mean(librosa.feature.zero_crossing_rate(y)),
        }
        for i,e in enumerate(mfcc):
            lc.update({f'mfcc-{i}':f' {np.mean(e)}'})
        
        return lc

    def encode_single_sample(self,wav_file, label,frame_length=2,
                            frame_step=2,fft_length=2,char_to_num=char_to_num):
        """
        this algorithm does the following:
            - Read wav file 
            - Decode the wav file
            - Change type to float
            - Get the spectrogram
            - We only need the magnitude, which can be derived by applying tf.abs
            - normalisation
            - Convert label to Lower case
            - Split the label
            - Map the characters in label to numbers
            -. Return a dict as our model is expecting two inputs
        """
        file = tf.io.read_file(wav_file)
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)
        spectrogram = tf.signal.stft(
            audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
        )
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        label = tf.strings.lower(label)
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")        
        label = char_to_num(label)
        return spectrogram, label

    def load_audios(self,language,wav_type='train',start=0,stop=None,files=None):
        """
        author: Martin Luther
        date: 31/05/2022
        how to use it :
            swahilis = load_audios(prefix,'swahili',0,10)
            amharics = load_audios(prefix,'amharics',0,10)
        expects:
            - prefix - string
            - language - string
            - start - int, stop - int (if you want 10 samples do 
            start = 0, stop = 10)
        returns:
            - samples and audio rates in 44.1khz
        """

        swahili_train_audio_path = f'../data/swahili_{wav_type}_wav/'
        swahili_wav_folders = os.listdir(path=swahili_train_audio_path)
        amharic_train_audio_path = f'../data/amharic_{wav_type}_wav/'
        amharic_wav_folders = os.listdir(path=amharic_train_audio_path)
        file_path = []
        swahili_wavs = []
        transformed_files=[]
        if files:
            transformed_files =  [x+'.wav' for x in files]
        for wav_folder in swahili_wav_folders:
            for wav_file in os.listdir(path=swahili_train_audio_path+wav_folder):
                if len(transformed_files) > 1:
                    if wav_file in transformed_files:
                        swahili_wavs.append(swahili_train_audio_path+wav_folder+'/'+wav_file)
                else:
                    swahili_wavs.append(swahili_train_audio_path+wav_folder+'/'+wav_file)
        loaded_files = []
        
        if language == 'swahili':

            for wav_file in swahili_wavs[start:len(swahili_wavs) if not stop else stop]:
                try:
                    loaded_files.append(librosa.load(wav_file, sr=44100))
                    loaded_files.append(wav_file)
                except Exception as e:
                    logger.error(e)
        else:
            for wav_file in amharic_wav_folders[start:len(amharic_wav_folders) if not stop else stop]:
                try:
                    if len(transformed_files) > 1:
                        if wav_file in transformed_files:
                            loaded_files.append(librosa.load(amharic_train_audio_path+wav_file, sr=44100))
                            loaded_files.append(amharic_train_audio_path+wav_file)
                            
                    else:
                        loaded_files.append(librosa.load(amharic_train_audio_path+wav_file, sr=44100))
                        loaded_files.append(amharic_train_audio_path+wav_file)

                except Exception as e:
                    logger.error(e)
        result = []
        audio,rate,file_path=[],[],[]
        for i,file in enumerate(loaded_files):
            if isinstance(file,tuple):
                audio_,rate_ = file
                audio.append(audio_)
                rate.append(rate_)
            else:
                file_path_ = file
                file_path.append(file_path_)

        for i in range(len(audio)):                
            result.append((audio[i],rate[i],self.get_duration(audio[i],rate[i]),file_path[i]))

        logger.info("successful in operation of loading audios")
        return result

    def get_labels(self,type='swahili',wav_type='train'):
        """
        author: Martin Luther
        date: 31/05/2022
        how it works
            - get_labels('swahili','train')
        expects:
            string
        returns: 
            string
        """
        labels = []
        if type=='swahili':
            swahili_wav_path=f'../data/swahili_{wav_type}_wav/'
            swahili_wav_folder = os.listdir(swahili_wav_path)
            for wav_folder in swahili_wav_folder:
                for wav_file in os.listdir(path=swahili_wav_path+wav_folder):
                    labels.append(wav_file)
        else:
            amharic_wav_path=f'../data/amharic_{wav_type}_wav/'
            amharic_wav_folder= os.listdir(amharic_wav_path)
            for wav_file in amharic_wav_folder:
                labels.append(wav_file)
        labels=[i.strip('.wav') for i in labels]
        return labels

    def read_text(self, text_path):
        '''
        author: Biruk
        date: 30/05/2022
        The function for reading the text
        '''
        text = []
        
        try:
            with open(os.path.join(os.getcwd(),text_path), encoding='utf-8') as fp:
                    line = fp.readline()
                    while line:
                        text.append(line)
                        line = fp.readline()
            # logger.info("successfully read file")
        except FileNotFoundError as e:
            logger.error(e)

        return text

    def read_data(self, train_text_path, test_text_path, train_labels, test_labels):
        '''
        author: Biruk
        date: 30/05/2022
        The function for reading the data from training and testing file paths
        '''
        train_text = self.read_text(train_text_path)
        test_text = self.read_text(test_text_path)

        train_text.extend(test_text)
        train_labels.extend(test_labels)

        new_text = []
        new_labels = []
        for i in train_text:
            result = i.split()

            if result[0] in train_labels:  # if the audio file exists
                new_text.append(' '.join([elem for elem in result[1:]]))
                new_labels.append(result[0])
        
        logger.info("Successfully read the data")          

        return new_text, new_labels 

    def get_clean_word(self, words, position):
        clean_word_dict = self.read_text("../data/amharic_dictionary.txt")
        clean_word_list = []
        for i in clean_word_dict:
            clean_word_list.append(i.split(" ")[0])

        word = temp_word = words[position]
        position = position + 1
        move_with_error = 0

        while((position < len(words) - 1) and (move_with_error < 5)):  # Give it five tries
            new_word = temp_word + words[position]
            if(new_word in clean_word_list):
                temp_word = new_word
                word = temp_word
                position = position + 1
                move_with_error = 0
            else:
                temp_word = new_word
                move_with_error = move_with_error + 1

        return word, position

    def get_duration(self, audio, rate):
        '''
        author: Biruk
        date: 30/05/2022
        The function which computes the duration of the audio files
        '''
        duration_of_recordings=None   
        try:
            duration_of_recordings = float(len(audio)/rate)
        except Exception as e:
            logger.error(e)
        return duration_of_recordings 
        
    def get_durations(self, filenames, label_data):
        """
        author: Biruk
        date: 30/05/2022
        """
        duration_of_recordings=[]
        for k in label_data:
            try:
                if k in filenames:
                    audio, fs = librosa.load(k+'.wav', sr=None)
                    duration_of_recordings.append(float(len(audio)/fs))
            except Exception as e:
                logger.error(f"has obtained an error {e}")                
        logger.info("The audio files duration is successfully computed")          
        return duration_of_recordings 

    def convert_channels(self,file1, output):
        """
        author: Martin Luther
        date: 31/05/2022
        how to use it: convert_channels("Input.wav", "Output.wav")
        expects:
            - wav file
        returns:
            - wav file
        """

        ifile = wave.open(file1)
        print(ifile.getparams())
        (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
        assert comptype == 'NONE'  # Compressed not supported yet
        array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
        left_channel = array.array(array_type, ifile.readframes(nframes))[::nchannels]
        ifile.close()
        stereo = 2 * left_channel
        stereo[0::2] = stereo[1::2] = left_channel
        ofile = wave.open(output, 'w')
        ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
        try:
            ofile.writeframes(stereo)
            logger.info("succesffully converted to stereo")
        except Exception as e:
            logger.error(e)
        ofile.close()

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    store = pd.read_csv('data/store.csv')
    clean_df = Clean(df)
    clean_df.merge_df(store,'Store')
    clean_df.save(name='data/unclean_train.csv')
    clean_df.drop_missing_values()
    clean_df.fix_outliers('Sales',25000)
    clean_df.transfrom_time_series('Date')
    train = clean_df.get_df()
    train['SchoolHoliday'] = train['SchoolHoliday'].astype(int)
    daily_sales = clean_df.aggregations(train,'Store','Sales','Open','sum')
    daily_customers = clean_df.aggregations(train,'Store','Customers','Open','sum')
    avg_sales = clean_df.aggregations(train,'Store','Sales','Open','mean')
    avg_customers = clean_df.aggregations(train,'Store','Customers','Open','mean')
    clean_df.label_encoding(train)
    indexes = ['DayOfWeek','Day', 'Month', 'Year', 'DayOfYear','WeekOfYear','Sales']
    training_data_ = train[train.columns.difference(indexes)]
    train_transformation=clean_df.generate_transformation(training_data_,"numeric","number")
    train_transformed = pd.DataFrame(train_transformation,columns=train.columns.difference(indexes))
    train_index = train[indexes]
    train_index = train_index.reset_index()
    train = pd.concat([train_index,train_transformed],axis=1)
    train.to_csv("data/cleaned_train.csv",index=False)
    train['DailySales'] = train['Store'].map(daily_sales)
    train['DailyCustomers'] = train['Store'].map(daily_customers)
    train['AvgSales'] = train['Store'].map(avg_sales)
    train['AvgCustomers'] = train['Store'].map(avg_customers)
    train.sort_values(["Year","Month","Day"], ascending=False ,ignore_index=True, inplace=True)
    train.to_csv("data/cleaned_aggregated_train.csv",index=False)
