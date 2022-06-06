from tensorflow import keras
from utils import decode_batch_predictions,vocab
from jiwer import wer
import tensorflow as tf
import numpy as np
import mlflow

AM_ALPHABET='ሀለሐመሠረሰቀበግዕዝተኀነአከወዐዘየደገጠጰጸፀፈፐቈኈጐኰፙፘፚauiāeəo'
EN_ALPHABET='abcdefghijklmnopqrstuvwxyz'


class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""
    
    def __init__(self, model,dataset):
        super().__init__()
        self.dataset = dataset
        self.model = model


    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        _,num_to_char = vocab(alphabet=AM_ALPHABET)
        for batch in self.dataset:
            X, y = batch
            batch_predictions = self.model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions,alphabet=AM_ALPHABET)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        mlflow.log_metric('wer-rate',wer_score)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)