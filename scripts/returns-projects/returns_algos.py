import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from scipy.signal import spectrogram
from scipy.fft import fft, ifft
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from data_prep import DataPrep, gen_seq, series_to_supervised, plotting
from config import Config

config = Config()

"""
Models to train:
- Transformer Encoder
- LSTM
- Linear models: Linear Regression, Elastic Net, Bayesian Ridge
- Gaussian Process Regressor
- Decision Tree Regressor
- Ensemble models: Random Forest Regressor, AdaBoost Regressor, Gradient Boosting Regressor
"""

def TransformerEncoder(input_dim, 
                       num_transformer_blocks, 
                       head_size, num_heads, 
                       ff_dim, 
                       dropout, 
                       mlp_dropout,
                       mlp_units):
    
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, activation="relu")(x)
        return x + res
    
    inputs = tf.keras.Input(shape=(config.sequence_length, input_dim), name="input_layer")
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
     
    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for i, dim in enumerate(mlp_units):
        x = tf.keras.layers.Dense(dim, activation="relu", name=f"dense_layer_{i}")(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)
    outputs = tf.keras.layers.Dense(1, name="transformer_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Transformer")

    return model

def trainTransformer(normalized_input_data,
                     input_dim,
                     num_transformer_blocks,
                     head_size,
                     num_heads,
                     ff_dim,
                     dropout,
                     mlp_units,
                     mlp_dropout,
                     sheet="One-Month"
    ):

    def custom_learning_rate(epoch):
        return (np.sqrt(input_dim) * 
                np.min([np.power(epoch+1, -0.5), 
                np.power(epoch+1, -0.5) * np.power(50, -1.5)])
                )

    reframed = series_to_supervised(normalized_input_data, config.sequence_length, 1)
        
    # Drop the Returns column (1st)
    reframed.drop(reframed.columns[input_dim:], axis=1, inplace=True)

    
    # Split into train and test sets (80-20)
    train_split = config.train_split
    val_split = config.val_split
    values = reframed.values
    
    train = values[:int(len(values)*train_split), :]
    validation = values[int(len(values)*train_split):int(len(values)*val_split), :] 
    
    # Split into input and outputs    
    train_input, train_output = train[:, 1:], train[:, 0]
    val_input, val_output = validation[:, 1:], validation[:, 0]
    
    # Reshape input to be 3D [samples, timesteps, features]
    train_input = train_input.reshape((train_input.shape[0], config.sequence_length, train_input.shape[1]))
    val_input = val_input.reshape((val_input.shape[0], config.sequence_length, val_input.shape[1]))
    #print(train_input.shape, train_output.shape, val_input.shape, val_output.shape)
    
    model = TransformerEncoder(
        input_dim-1, 
        num_transformer_blocks, 
        head_size, 
        num_heads, 
        ff_dim, 
        dropout, 
        mlp_dropout,
        mlp_units)

    # Stop training when a monitored quantity has stopped improving for 3 consecutive epochs.
    callbacks = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        start_from_epoch=100)
    
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.legacy.Adam(
                    learning_rate=0.002,
                    beta_1=0.9,
                    beta_2=0.98,
                    epsilon=1e-9),
                  metrics=['mae'])
    #model.summary()

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(custom_learning_rate)

    history = model.fit(train_input, train_output,
                        validation_data=(val_input, val_output),
                        verbose=1,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        shuffle=False,
                        callbacks=[callbacks, lr_schedule]
                    )
    
    model.save(f"../saved_models/Transformer_returns_{sheet}.keras")

def LSTM(input_dim):
    # Design an LSTM to predict the next value of the time series
    inputs = tf.keras.layers.Input(shape=(config.sequence_length, input_dim), name="input_layer")
    x = tf.keras.layers.LSTM(
        config.filters[0], return_sequences=True, name=f"LSTM_layer_{config.filters[0]}")(inputs)
    x = tf.keras.layers.LSTM(
        config.filters[1], return_sequences=True, name=f"LSTM_layer_{config.filters[1]}")(x)
    x = tf.keras.layers.LSTM(
        config.filters[2], name=f"LSTM_layer_{config.filters[2]}", recurrent_dropout=.1)(x)
    x = tf.keras.layers.Dense(config.lstm_mlp_units[0], name="dense_layer")(x)
    x = tf.keras.layers.Dense(config.lstm_mlp_units[1], name="dense_layer_2")(x)
    outputs = tf.keras.layers.Dense(1, name="output_layer")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="LSTM")

    return model

def train_LSTM(normalized_input_data, input_dim, sheet="One-Month", plot=False):
    
    reframed = series_to_supervised(normalized_input_data, config.sequence_length, 1)
        
    # Drop the Returns column (1st)
    reframed.drop(reframed.columns[input_dim:], axis=1, inplace=True)

    
    # Split into train and test sets (80-20)
    train_split = config.train_split
    val_split = config.val_split
    values = reframed.values
    
    train = values[:int(len(values)*train_split), :]
    validation = values[int(len(values)*train_split):int(len(values)*val_split), :] 
    
    # Split into input and outputs    
    train_input, train_output = train[:, 1:], train[:, 0]
    val_input, val_output = validation[:, 1:], validation[:, 0]
    
    # Reshape input to be 3D [samples, timesteps, features]
    train_input = train_input.reshape((train_input.shape[0], config.sequence_length, train_input.shape[1]))
    val_input = val_input.reshape((val_input.shape[0], config.sequence_length, val_input.shape[1]))
    #print(train_input.shape, train_output.shape, val_input.shape, val_output.shape)
    
    # Read model
    model = LSTM(input_dim-1)
    
    # Stop training when a monitored quantity has stopped improving for 3 consecutive epochs.
    callbacks = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, start_from_epoch=100)
    
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.002),
                  metrics=['mae'])
    
    history = model.fit(train_input, train_output,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        validation_data=(val_input, val_output),
                        verbose=1,
                        shuffle=False,
                        callbacks=[callbacks])
    
    model.save(f"../saved_models/LSTM_returns_{sheet}.keras")

    # plot history
    if plot:
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

def linearModels(normalized_input_data, algorithm="LR", sheet="One-Month"):
    # Load model
    if algorithm == "LR":
        model = LinearRegression()
    elif algorithm == "EN":
        model = ElasticNet()
    elif algorithm == "BR":
        model = BayesianRidge()
    else:
        raise ValueError("Algorithm not supported")
    
    # Split into train and test sets (80-20)
    train_split = config.train_split

    values = normalized_input_data
    train = values[:int(len(values)*train_split), :]

    # Split into input and outputs
    train_input, train_output = train[:, 1:], train[:, 0]

    # Fit model
    model.fit(train_input, train_output)

    # Save model
    # Save the trained model using pickle
    with open(f'../saved_models/{algorithm}_returns_{sheet}.pkl', 'wb') as file:
        pickle.dump(model, file)

def gaussianProcess(normalized_input_data, sheet="One-Month"):
    # Load model
    model = GaussianProcessRegressor()

    # Split into train and test sets (80-20)
    train_split = train_split

    values = normalized_input_data
    train = values[:int(len(values)*train_split), :]

    # Split into input and outputs
    train_input, train_output = train[:, 1:], train[:, 0]

    # Fit model
    model.fit(train_input, train_output)

    # Save model
    # Save the trained model using pickle
    with open(f'../saved_models/GP_returns_{sheet}.pkl', 'wb') as file:
        pickle.dump(model, file)

def decisionTree(normalized_input_data, sheet="One-Month"):
    # Load model
    model = DecisionTreeRegressor()

    # Split into train and test sets (80-20)
    train_split = train_split

    values = normalized_input_data
    train = values[:int(len(values)*train_split), :]

    # Split into input and outputs
    train_input, train_output = train[:, 1:], train[:, 0]

    # Fit model
    model.fit(train_input, train_output)

    # Save model
    # Save the trained model using pickle
    with open(f'../saved_models/DT_returns_{sheet}.pkl', 'wb') as file:
        pickle.dump(model, file)    

def ensembleModels(normalized_input_data, algorithm="RF", sheet="One-Month"):
    # Load model
    if algorithm == "RF":
        model = RandomForestRegressor()
    elif algorithm == "AB":
        model = AdaBoostRegressor()
    elif algorithm == "GB":
        model = GradientBoostingRegressor()
    else:
        raise ValueError("Algorithm not supported")

    # Split into train and test sets (80-20)
    train_split = train_split

    values = normalized_input_data
    train = values[:int(len(values)*train_split), :]

    # Split into input and outputs
    train_input, train_output = train[:, 1:], train[:, 0]

    # Fit model
    model.fit(train_input, train_output)

    # Save model
    # Save the trained model using pickle
    with open(f'../saved_models/{algorithm}_returns_{sheet}.pkl', 'wb') as file:
        pickle.dump(model, file)