from pyexpat import features
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pandas.plotting import scatter_matrix
import librosa.display
from logger import logger
from numpy.lib.stride_tricks import as_strided
from mpl_toolkits.axes_grid1 import make_axes_locatable

class EDA:
    """
    - this class is responsible for performing 
    Exploratory Data Analysis
    """

    def __init__(self, df=None):
        """initialize the eda class"""
        self.df = df
        logger.info("Successfully initialized eda class")

    def descriptive_stats(self, describe=False, info=False, size=False):
        """
        expects: 
            - boolean
        returns:
            - summary
        """
        summary = None
        if describe:
            summary = self.df.describe()
            logger.info("Successfully performed describe")
        elif info:
            summary = self.df.info
            logger.info("Successfully performed info")
        elif size:
            summary = self.df.shape
            logger.info("Successfully performed shape")
        return summary

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
        logger.info("Successfully removed missing values from data")
        return has_missing_values

    def plot_counts(self, column, second_column=None, type=None):
        """
        expects:
            -   string
        returns:
            -   plot
        """
        if type == "univariate":
            plt.figure(figsize=(12, 6))
            sns.countplot(data=self.df, x=column)
            plt.title(f"Unique value counts of the {column} columns")
            plt.show()
            logger.info("Successfully plotted univariate countplot")
        elif type == "bivariate":
            plt.figure(figsize=(12, 6))
            sns.countplot(data=self.df, x=second_column, hue=column)
            plt.title(f"{column} vs {second_column}")
            plt.show()
            logger.info("Successfully plotted bivariate countplot")
        elif type == "bivariate_line":
            plt.figure(figsize=(28,10))
            plt.ylim(-25000,25000)
            sns.lineplot(self.df.index, y= self.df[column], hue=self.df[second_column]);
            plt.show()
            logger.info("Successfully plotted bivariate line")
        elif type == "bivariate_scatter":
            plt.figure(figsize=(12, 8))
            plt.scatter(self.df[column],self.df[second_column], alpha=0.1)
            plt.xlabel(column)
            plt.ylabel(second_column)
            plt.plot()
            logger.info("Successfully plotted bivariate scatter")
        elif type == "bivariate_hist":
            cols = ['purple','green','red','yellow']
            labels = ['With','Without']
            column_values = np.unique(self.df[column])
            if isinstance(column_values,int):
                for i in reversed(range(0,2)):
                    promos = self.df[self.df[column] == i][second_column]
                    plt.hist(promos, 
                    color=cols[i], alpha=0.3, label =labels[i])
            else:
                for i,x in enumerate(column_values):
                    plt.hist(self.df[self.df[column] == x][second_column], 
                    color=cols[i], alpha=0.3, label = x)

            plt.ylabel(column)
            plt.xlabel(second_column)
            plt.legend()
            plt.plot()
            logger.info("Successfully plotted bivariate histograms")
        elif type == "bivariate_count":
            sns.countplot( x=column, data=self.df, hue=second_column, palette="Set1")
            plt.show()
            logger.info("Successfully plotted bivariate countplot")
        return


    def between(self,column,start,stop):
        """
        - get values in between certain range
        """
        return self.df[self.df[column].between(start,stop)]

    
    def correlation_analysis(self,column=None,second_column=None,type="all"):
        """
        expects:
            - nothing
        returns:
            - dataframe
        """
        if type=="all":
            corr = self.df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True)
            plt.title('Heatmap of correlation for the numerical columns')
            plt.show()
            logger.info("Successfully plotted heatmap")
        elif type=="bivariate":
            attributes = [column,second_column]
            scatter_matrix(self.df[attributes], alpha=0.1);
            plt.show()
            logger.info("Successfully plotted correlation scatter matrix")
        return 

    def plot_distributions(self):
        """
        - this algorithm is responsible for plotting distributions
        """
        num_feats = list(self.df.select_dtypes(include=['int64', 'float64', 'int32']).columns)
        self.df[num_feats].hist(figsize=(20,15))
        logger.info("Successfully plotted distributions in the histogram formats")

    def get_df(self):
        """
        - returns the dataframes
        """
        return self.df


    def sound_plots(self,audio,rate,type=None,features=None):
        """
        author: Martin Luther Bironga
        date: 31/05/2022
        how it works:
            sound_plots(type='waveshow')
        """
        if type=='waveshow':
            plt.figure(figsize=(20, 5))
            librosa.display.waveshow(audio,sr=rate)
            plt.show()
        elif type=='specshow':
            X = librosa.stft(audio)
            if isinstance(features,np.ndarray):
                if features.size != 0:
                    features = features
            else:
                features = librosa.amplitude_to_db(abs(X))
            plt.figure(figsize=(14, 5))
            librosa.display.specshow(features, sr=rate, x_axis='time', y_axis='hz')
            plt.colorbar()
            plt.show()

    def spectrogram(self,samples, fft_length=256, sample_rate=2, hop_length=128):
        """
        Compute the spectrogram for a real signal.
        The parameters follow the naming convention of
        matplotlib.mlab.specgram

        Args:
            samples (1D array): input audio signal
            fft_length (int): number of elements in fft window
            sample_rate (scalar): sample rate
            hop_length (int): hop length (relative offset between neighboring
                fft windows).

        Returns:
            x (2D array): spectrogram [frequency x time]
            freq (1D array): frequency of each row in x

        Note:
            This is a truncating computation e.g. if fft_length=10,
            hop_length=5 and the signal has 23 elements, then the
            last 3 elements will be truncated.
        """
        assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

        window = np.hanning(fft_length)[:, None]
        window_norm = np.sum(window**2)

        # The scaling below follows the convention of
        # matplotlib.mlab.specgram which is the same as
        # matlabs specgram.
        scale = window_norm * sample_rate

        trunc = (len(samples) - fft_length) % hop_length
        x = samples[:len(samples) - trunc]

        # "stride trick" reshape to include overlap
        nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
        nstrides = (x.strides[0], x.strides[0] * hop_length)
        x = as_strided(x, shape=nshape, strides=nstrides)

        # window stride sanity check
        assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

        # broadcast window, compute fft over columns and square mod
        x = np.fft.rfft(x * window, axis=0)
        x = np.absolute(x)**2

        # scale, 2.0 for everything except dc and fft_length/2
        x[1:-1, :] *= (2.0 / scale)
        x[(0, -1), :] /= scale

        freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

        return x, freqs

    def plot_spectrogram_feature(self,vis_spectrogram_feature):
        # plot the normalized spectrogram
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(111)
        im = ax.imshow(vis_spectrogram_feature, cmap=plt.cm.jet, aspect='auto')
        plt.title('Spectrogram')
        plt.ylabel('Time')
        plt.xlabel('Frequency')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()


    def plot_spec(self,data:np.array,sr:int) -> None:
        '''
        Function for plotting spectrogram along with amplitude wave graph
        '''
    
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].title.set_text(f'Shfiting the wave by Times {sr/10}')
        ax[0].specgram(data,Fs=2)
        ax[1].set_ylabel('Amplitude')
        ax[1].plot(np.linspace(0,1,len(data)), data)


    def features(self,audio,rate,type='mfcc'):
        if type=='mfcc':
            features = librosa.feature.mfcc(audio,sr=rate)
        
        return features

if __name__ == '__main__':
    """testing eda"""
    path_1 = sys.argv[1]
    df = pd.read_csv(path_1)
    eda = EDA(df)
    eda_df = eda.get_df()
    eda_df.to_csv("data/eda.csv", index=False)
