import keras
import numpy as np
import pandas as pd
import seaborn as sns
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Conv1D

# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length),
                           range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

# function to generate labels
def gen_labels(id_df, seq_length, label): 

    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]

    return data_matrix[seq_length:num_elements, :]

# Metrics using keras backend
def r2_keras(y_true, y_pred): 
    """Coefficient of Determination 
    """
    res =  K.sum(K.square( y_true - y_pred ))
    tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - res/(tot + K.epsilon()) )

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Model
def DCNN(sequence_length, nb_features, nb_out):
    print(sequence_length, nb_features, nb_out)
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(128, 8, padding='same', activation='relu',
                                  input_shape=(sequence_length, nb_features)))
    model.add(keras.layers.Conv1D(256, 8, padding='same', activation='relu'))
    model.add(keras.layers.Conv1D(256, 8, padding='same', activation='relu'))
    model.add(keras.layers.Conv1D(128, 8, padding='same', activation='relu'))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(100, activation='linear'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=nb_out))
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss=keras.losses.Huber(), optimizer=optimizer, metrics=([rmse, r2_keras]))
    
    return model
