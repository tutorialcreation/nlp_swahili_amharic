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
import torch
import torchaudio
from tensorflow import keras
import random
from logger import logger
import IPython.display as ipd
import warnings
import wave, array
warnings.filterwarnings("ignore")

class Clean:
    """
    - this class is responsible for performing 
    Cleaning Tasks
    """

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


    def clean_text(self,df,column):
        """
        todo: Biruk / amharic (nltk) ... Amal / swahili
        """
        df['text'] = 0
        return df

    
    def char_index(self,alphabet):
        a_map = {} # map letter to number
        rev_a_map = {} # map number to letter
        for i, a in enumerate(alphabet):
            a_map[a] = i
            rev_a_map[i] = a
        return rev_a_map

    def vocab(self,alphabet):

        # Mapping characters to integers
        characters = [x for x in alphabet]
        char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
        # Mapping integers back to original characters
        num_to_char = keras.layers.StringLookup(
            vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
        )

        print(
            f"The vocabulary is: {char_to_num.get_vocabulary()} "
            f"(size ={char_to_num.vocabulary_size()})"
        )
        return (char_to_num,num_to_char)


    def store_audio_features(self,y,sr):
        """
        author: Martin Luther
        function: returns different features from the audio
        """
        
        y, sr = y,sr
        lc = {
            "rmse":np.mean(librosa.feature.rms(y=y)),
            "chroma_stft":np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            "spec_cent":np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "spec_bw":np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            "rolloff":np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            "zcr":np.mean(librosa.feature.zero_crossing_rate(y)),
            "mfcc":np.mean(librosa.feature.mfcc(y=y, sr=sr))
        }
        
        return lc

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
                except Exception as e:
                    logger.error(e)
        else:
            for wav_file in amharic_wav_folders[start:len(amharic_wav_folders) if not stop else stop]:
                try:
                    if len(transformed_files) > 1:
                        if wav_file in transformed_files:
                            loaded_files.append(librosa.load(amharic_train_audio_path+wav_file, sr=44100))
                    else:
                        loaded_files.append(librosa.load(amharic_train_audio_path+wav_file, sr=44100))

                except Exception as e:
                    logger.error(e)
        result = []
        for file in loaded_files:
            audio,rate = file
            result.append((audio,rate,self.get_duration(audio,rate)))
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
