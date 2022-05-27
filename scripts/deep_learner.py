# importing of libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os,sys
print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join('../scripts')))
from logger import logger
from model_serializer import ModelSerializer
import mlflow
import csv
import seaborn as sns
sns.set()


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

    def model(self,model_,serialize=True):
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
            data = [[i] for i in forecast]
 
            # opening the csv file in 'w+' mode
            file = open('data/forecast_deep.csv', 'w+', newline ='')
            
            # writing the data into the file
            with file:   
                write = csv.writer(file)
                write.writerows(data)
            mlflow.log_artifact("data/forecast_deep.csv")
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
    
if __name__=='__main__':
    train_ = pd.read_csv("data/cleaned_train.csv")
    train_.set_index('Date',inplace=True)
    """make sales to be last column"""
    train=train_.loc[:,train_.columns!='Sales']
    train['Sales']=train_['Sales']
    n = len(train)
    train_df = train[0:int(n*0.7)]
    val_df = train[int(n*0.7):int(n*0.9)]
    test_df = train[int(n*0.9):]
    num_features = train.shape[1]
    learn = DeepLearn(input_width=1, label_width=1, shift=1,epochs=5,
                     train_df=train_df, val_df=val_df, test_df=test_df,
                     label_columns=['Sales'])
    forecast = learn.model(
        model_=tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
        ])
    )
    
    