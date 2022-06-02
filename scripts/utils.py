from mlflow import get_experiment_by_name
import os
from tensorflow import keras
import numpy as np
import tensorflow as tf

def get_experiment_id(file):
    current_experiment=dict(get_experiment_by_name("abtest"))
    experiment_id=current_experiment['experiment_id']
    file_path = os.path.abspath(os.path.join(f'mlruns/0/{experiment_id}/artifacts/{file}')) 
    return file_path

def vocab(alphabet):

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


def decode_batch_predictions(pred,alphabet=None):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    _,num_to_char = vocab(alphabet=alphabet)
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text
