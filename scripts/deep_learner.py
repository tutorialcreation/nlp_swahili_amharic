# importing of libraries
from bleach import Cleaner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os,sys

from scripts.modeling import Modeler
print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join('../scripts')))
from logger import logger
from tensorflow import keras
from tensorflow.keras import layers
from model_serializer import ModelSerializer
from clean import Clean
from utils import vocab
import mlflow
import csv
import seaborn as sns
sns.set()

AM_ALPHABET='ሀለሐመሠረሰቀበግዕዝተኀነአከወዐዘየደገጠጰጸፀፈፐቈኈጐኰፙፘፚauiāeəo'
EN_ALPHABET='abcdefghijklmnopqrstuvwxyz'
class DeepLearn:
    """
    - this class is responsible for deep learning
    """

    def __init__(self,input_width, label_width, shift,epochs=5,
                train_df=None, val_df=None, test_df=None,
                label_columns=None):
        """initialize the Deep Learn class"""
        self.epochs = epochs
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        """this function splits the window"""
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """
        this function is responsible
        for making the dataset
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property        # Work out the label column indices.
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def get_input_labels(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_res', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._res = result
        return result

    def compile_and_fit(self,model, window, patience=2):
        """this function fits the data set for prediction"""
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=self.epochs,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history

    def model(self,model_,serialize=False):
        """model the lstm class"""
        mlflow.tensorflow.autolog()
        model = model_
        with mlflow.start_run(run_name='deep-learner'):
            mlflow.set_tag("mlflow.runName", "deep-learner")
            learn = DeepLearn(input_width=self.input_width, label_width=self.label_width, shift=self.shift,
                     train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
                     label_columns=self.label_columns)
            inputs_, _ = learn.get_input_labels
            _ = self.compile_and_fit(model, learn)
            logger.info("Successfully executed the model")
            
            """forecast the data"""
            forecast=model(inputs_).numpy().tolist()
        if serialize:
            serializer = ModelSerializer(model)
            serializer.pickle_serialize()
        return forecast

    def plot(self, model=None, plot_col=None, max_subplots=3):
        """this algorithm checks the performance of the model"""
        plot_col = self.label_columns
        inputs, labels = self.get_input_labels
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time')


    def CTCLoss(self,y_true, y_pred):
        # Compute the training-time loss value
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss

        

    def build_asr_model(self,input_dim, output_dim, rnn_layers=5, rnn_units=12,
                        serialize=True):
        """
        this functions works like this
        - input spectogram
        - Expand the dimension to use 2D CNN.
        - Convolution layer 1
        - Convolution layer 2
        - Reshape the resulted volume to feed the RNNs layers
        - Dense layer                                                                        
        - Classification layer
        - Model
        - Optimizer
        - Compile the model and log it
        - Serialize and save the model 
        """
        input_spectrogram = layers.Input((None, input_dim), name="input")
        x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
        x = layers.Conv2D(
            filters=3,
            kernel_size=[11, 41],
            strides=[2, 2],
            padding="same",
            use_bias=False,
            name="conv_1",
        )(x)
        x = layers.BatchNormalization(name="conv_1_bn")(x)
        x = layers.ReLU(name="conv_1_relu")(x)
        x = layers.Conv2D(
            filters=2,
            kernel_size=[11, 21],
            strides=[1, 2],
            padding="same",
            use_bias=False,
            name="conv_2",
        )(x)
        x = layers.BatchNormalization(name="conv_2_bn")(x)
        x = layers.ReLU(name="conv_2_relu")(x)
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        for i in range(1, rnn_layers + 1):
            recurrent = layers.GRU(
                units=rnn_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=f"gru_{i}",
            )
            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if i < rnn_layers:
                x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        x = layers.ReLU(name="dense_1_relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
        mlflow.tensorflow.autolog()
        model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
        with mlflow.start_run(run_name='audio-deep-learner'):
            mlflow.set_tag("mlflow.runName", "audio-deep-learner")
            opt = keras.optimizers.Adam(learning_rate=1e-4)
            model.compile(optimizer=opt, loss=self.CTCLoss)
            logger.info("Successfully run the deep learing model")
        if serialize:
            serializer = ModelSerializer(model)
            serializer.pickle_serialize()
        return model

    

    
if __name__=='__main__':
    cleaner = Clean()
    char_to_num,num_to_char=vocab(EN_ALPHABET)
    swahili_df = pd.read_csv('../data/swahili.csv')
    lang = pd.read_csv("../data/swahili.csv")
    lang['type']='swahili'
    amharic_df = pd.read_csv("../data/amharic.csv")
    amharic_df['type']='amharic'
    language_df = lang.append(amharic_df, ignore_index=True)
    pre_model = Modeler()
    swahili_preprocessed = pre_model.preprocessing_learn(swahili_df,'key','file')

    