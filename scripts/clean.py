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
import string
import wave, array
warnings.filterwarnings("ignore")
AM_ALPHABET='ሀለሐመሠረሰሸቀቐበቨተቸኀነኘአከኸወዐዘዠየደዸጀገጠጨጰጸፀፈፐፙፘፚauiāeəo'
EN_ALPHABET=string.ascii_lowercase

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
        

    def has_missing_values(self):
        """
        expects:
            -   nothing
        returns:
            -   boolean
        """
        has_missing_values = False
        if True in self.df.isnull().any().to_list():
            has_missing_values = True
        counts = None
        counts = self.df.isnull().sum()
        logger.info("Successfully checked for missing values")
        return counts,has_missing_values

    def store_features(self,type_,value):
        """
        purpose:
            - stores features for the data set
        input:
            - string,int,dataframe
        returns:
            - dataframe
        """
        features = [None]
        if type_ == "numeric":
            features = self.df.select_dtypes(include=value).columns.tolist()
        elif type_ == "categorical":
            features = self.df.select_dtypes(exclude=value).columns.tolist()
        logger.info("Successfully stored the features")
        return features

    def merge_df(self,df_,column):
        """
        expects:
            - string(column)
        returns:
            - merged df
        """
        try:
            column = column
        except Exception as e:
            logger.error(f'please add {e}')
        self.df = pd.merge(self.df, df_, how = 'left', on = column)
        logger.info("Successfully merged the dataframe")
        return self.df

    
    def handle_missing_values_numeric(self, features, df=None):
        """
        this algorithm does the following
        - remove columns with x percentage of missing values
        - fill the missing values with the mean
        returns:
            - df
            - percentage of missing values
        """
        if df:
            self.df=df
        missing_percentage = round((self.df.isnull().sum().sum()/\
                reduce(lambda x, y: x*y, self.df.shape))*100,2)
        for key in features:
            self.df[key] = self.df[key].fillna(self.df[key].mean())
        logger.info("Successfully handled missing values for numerical case")
        return missing_percentage, self.df

    def handle_missing_values_categorical(self,features):
        """
        this algorithm does the following
        - remove columns with x percentage of missing values
        - fill the missing values with the mode
        returns:
            - df
            - percentage of missing values
        """
        missing_percentage = round((self.df.isnull().sum().sum()/\
                reduce(lambda x, y: x*y, self.df.shape))*100,2)
        for key in features:
            self.df[key] = self.df[key].fillna(self.df[key].mode()[0])
        logger.info("Successfully handled missing values for categorical values")            
        return missing_percentage, self.df


    def drop_missing_values(self)->pd.DataFrame:
        """
        remove rows that has column names. This error originated from
        the data collection stage.  
        """
        self.df.dropna(inplace=True)
        logger.info("Successfully dropped the columns with missing values")
    
    def drop_duplicate(self, df:pd.DataFrame,column)->pd.DataFrame:
        """
        - this function drop duplicate rows
        """
        self.df = self.df.drop_duplicates(subset=[column])
        logger.info("Successfully dropped rows with duplicate values")
        return self.df
        
    def convert_to_datetime(self, column)->pd.DataFrame:
        """
        convert column to datetime
        """
        self.df[column] = pd.to_datetime(self.df[column])
        logger.info("Successfully converted the column to datetime")
        return self.df

    def store_audio_features(self,type='amharic'):
        data = []
        amharic_path = '../data/amharic_train_wav/'
        for filename in os.listdir(amharic_path)[0:10]:
            y, sr = librosa.load(amharic_path+filename, mono=False)
            lc = {
                "filename":filename,
                "rmse":np.mean(librosa.feature.rms(y=y)),
                "chroma_stft":np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
                "spec_cent":np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                "spec_bw":np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                "rolloff":np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                "zcr":np.mean(librosa.feature.zero_crossing_rate(y)),
                "mfcc":np.mean(librosa.feature.mfcc(y=y, sr=sr))
            }
            data.append(lc)
            
        
        df = pd.DataFrame(data)
        return df



    def fix_outliers(self,column,threshold):
        """
        - this algorithm fixes outliers
        """
        numerical_columns=self.store_features("numeric","number")
        for i in numerical_columns:
            if i == column:
                self.df = self.df[self.df[column] < threshold]  #Drops samples which have sales more than 25000
                self.df.reset_index(drop=True)
        logger.info("Successfully handled outliers")
        return

    def generate_pipeline(self,type_="numeric",x=1):
        """
        purpose:
            - generate_pipelines for the data
        input:
            - string and int
        returns:
            - pipeline
        """
        pipeline = None
        if type_ == "numeric":
            pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='mean')),
                ('scale', MinMaxScaler())
            ])
        elif type_ == "categorical":
            pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        else:
            pipeline = np.zeros(x)
        return pipeline
    
    def generate_transformation(self,df,type_,value,trim=None,key=None):
        """
        purpose:
            - generates transformations for the data
        input:
            - string,int and df
        returns:
            - transformation
        """
        transformation = None
        pipeline = self.generate_pipeline(type_,value)
        if type_=="numeric":
            transformation=pipeline.fit_transform(df.select_dtypes(include=value))
            logger.info("Successfully transformed numerical data")
        elif type_ == "categorical":
            transformation=pipeline.fit_transform(df.select_dtypes(exclude=value))
            logger.info("Successfully transformed categorical data")
        return transformation

    
    def remove_unnamed_cols(self):
        """
        - this algorithm removes columns with unnamed
        """
        self.df.drop(self.df.columns[self.df.columns.str.contains('unnamed',
        case = False)],axis = 1, inplace = True)
        logger.info("Successfully removed columns with head unnamed")
    
    def label_encoding(self,train,test=None):
        """
        - label encode the data
        """
        categorical_features=self.store_features("categorical","number")
        train[categorical_features] = train[categorical_features].apply(lambda x: pd.factorize(x)[0])
        logger.info("Successfully encoded your categorical data")


    def transfrom_time_series(self,date_column):
        """
        - transform the data into a 
        time series dataset
        """
        self.df[date_column] = pd.to_datetime(self.df[date_column],errors='coerce')
        self.df['Day'] = self.df[date_column].dt.day
        self.df['Month'] = self.df[date_column].dt.month
        self.df['Year'] = self.df[date_column].dt.year
        self.df['DayOfYear'] = self.df[date_column].dt.dayofyear
        self.df['WeekOfYear'] = self.df[date_column].dt.weekofyear
        self.df.set_index(date_column, inplace=True)
        logger.info("Successfully transformed data to time series data")

    def save(self,name):
        """
        - returns the dataframes
        """
        self.df.to_csv(name,index=False)
        logger.info("Successfully saved the dataframe")

    def get_df(self):
        """
        - returns the dataframe
        """
        return self.df


    def aggregations(self,df,column=None,second_column=None,
                    third_column=None,according_to="sum"):
        """
        - this is for adding features based on aggregations
        """
        if according_to=="sum":
            grouped_x =  df.groupby([df[column]])[second_column].sum()
        elif according_to=="mean":
            grouped_x =  df.groupby([df[column]])[second_column].mean()
        grouped_y = df.groupby([df[column]])[third_column].count()
        per_x = grouped_x / grouped_y
        logger.info("successful aggregation")
        return dict(per_x)



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

    
    

    def encode_single_sample(self,wav_file, label,frame_length=256,
                            frame_step=160,fft_length=384,char_to_num=char_to_num):
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
                    logger.info("successfully read file")
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
    no_observations = int(sys.argv[1])
    NO_OBSERVATIOINS=no_observations
    cleaning_audios = Clean()
    swahili_train_labels = cleaning_audios.get_labels('swahili','train')
    swahili_test_labels = cleaning_audios.get_labels('swahili','test')
    amharic_train_labels = cleaning_audios.get_labels('amharic','train')
    amharic_test_labels = cleaning_audios.get_labels('amharic','test')
    swahili_text_data, swahili_label_data = cleaning_audios.read_data('../data/swahili_train_text.txt', '../data/swahili_test_text.txt',
                                                  swahili_train_labels, swahili_test_labels)
    amharic_text_data, amharic_label_data = cleaning_audios.read_data('../data/amharic_train_text.txt', '../data/amharic_test_text.txt',
                                                  amharic_train_labels, amharic_test_labels)
    swahili_data = pd.DataFrame({'key': swahili_label_data, 'text': swahili_text_data})
    amharic_data = pd.DataFrame({'key': amharic_label_data, 'text': amharic_text_data})                                                  
    swahili_recordings = cleaning_audios.load_audios('swahili',files=swahili_data.key.to_list()[0:NO_OBSERVATIOINS])
    amharic_recordings = cleaning_audios.load_audios('amharic',files=amharic_data.key.to_list()[0:NO_OBSERVATIOINS])
    swahili_data_df = swahili_data.head(NO_OBSERVATIOINS)
    amharic_data_df = amharic_data.head(NO_OBSERVATIOINS)
    durations = []
    for recording in swahili_recordings:
        _,_,duration,_ = recording
        durations.append(duration)
    swahili_data_df['duration'] = durations
    durations = []
    for recording in amharic_recordings:
        _,_,duration,_ = recording
        durations.append(duration)
    amharic_data_df['duration'] = durations
    y = [x in swahili_test_labels for x in swahili_data.key]
    swahili_data["category"] = ["Test" if i else "Train" for i in y]
    y = [x in amharic_test_labels for x in amharic_data.key]
    amharic_data["category"] = ["Test" if i else "Train" for i in y]
    rates = []
    for recording in swahili_recordings:
        _,rate,_,_ = recording
        rates.append(rate)
    swahili_data_df['rate'] = rates
    rates = []
    for recording in amharic_recordings:
        _,rate,_,_ = recording
        rates.append(rate)
    amharic_data_df['rate'] = rates
    file_path = []
    for recording in swahili_recordings:
        _,_,_,file_path_ = recording
        file_path.append(file_path_)
    swahili_data_df['file'] = file_path
    file_path = []
    for recording in amharic_recordings:
        _,_,_,file_path_ = recording
        file_path.append(file_path_)
    amharic_data_df['file'] = file_path
    audio_rates=[]
    for recording in amharic_recordings:
        audio,rate,duration,_ = recording
        audio_rates.append((audio,rate,duration))

    max_ms = max([i[2] for i in audio_rates[0:len(audio_rates)]])*1000
    max_ms
    audios = [i[0].tolist() for i in audio_rates[0:len(audio_rates)]]
    new_audio_rates = []
    for i,x in enumerate(audio_rates):
        audio,rate,_, = x
        new_audio_rates.append(cleaning_audios.pad_trunc(audio,rate,max_ms))
    amharic_data_df['duration'] = amharic_data_df['duration'][3] 
    audio_rates=[]
    for recording in swahili_recordings:
        audio,rate,duration,_ = recording
        audio_rates.append((audio,rate,duration))

    max_ms = max([i[2] for i in audio_rates[0:len(audio_rates)]])*1000
    max_ms
    audios = [i[0].tolist() for i in audio_rates[0:len(audio_rates)]]
    new_audio_rates = []
    for i,x in enumerate(audio_rates):
        audio,rate,_ = x
        new_audio_rates.append(cleaning_audios.pad_trunc(audio,rate,max_ms))
    swahili_data_df['duration']=swahili_data_df['duration'][4]
    audio_rates=[]
    for recording in amharic_recordings:
        audio,rate,_,_ = recording
        audio_rates.append((audio,rate))
    data = []
    for audio,rate in audio_rates:
        data.append(cleaning_audios.store_audio_features(audio,rate))
    amharic = pd.DataFrame(data)
    amharic_df = pd.concat([amharic_data_df,amharic],axis=1)
    amharic_df['type'] = 'amharic'
    audio_rates=[]
    for recording in swahili_recordings:
        audio,rate,_,_ = recording
        audio_rates.append((audio,rate))
    data = []
    for audio,rate in audio_rates:
        data.append(cleaning_audios.store_audio_features(audio,rate))
    swahili = pd.DataFrame(data)
    swahili_df = pd.concat([swahili_data_df,swahili],axis=1)
    swahili_df['type']='swahili'
    amharic_df.to_csv("../data/amharic.csv",index=False)
    swahili_df.to_csv("../data/swahili.csv",index=False)