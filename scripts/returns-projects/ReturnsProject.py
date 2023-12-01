import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from scipy.signal import spectrogram
from scipy.fft import fft, ifft
from sklearn.preprocessing import MinMaxScaler

from config import Config
from data_prep import DataPrep, gen_seq, series_to_supervised, plotting

from returns_algos import (trainTransformer, train_LSTM, linearModels, 
                           gaussianProcess, decisionTree, ensembleModels)


# Load configuration file with necessary parameters
config = Config()

# Load data, choose period via sheet name
sheets = ["One-Month", "Two-Month", "Three-Month"]
sheet_cntr = int(input('Choose xlsx sheet (0: One-Month, 1: Two-Month, 2: Three-Month): '))
sheet = sheets[sheet_cntr]

# Choose whether to use FFT or not; whether to scale data or not
use_fft = True
scale_data = False

# Load data
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+"/Data_for_Frequency_Domain.xlsx"

df = pd.read_excel(data_path,
                sheet_name=sheet,
                names=["Returns","Price/Dividedn","Lag Dividednt Growth","Risk-free Rate","Lag Returns"],
                usecols="F:J",
                #skipfooter=4
                )


# Input dimension; accommodates the four main columns plus the Returns column (only for LSTM)
input_dim = df.shape[1] + 1

input_data = df.values

# Scale the data {0,1}
if scale_data:
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_input_data = scaler.fit_transform(input_data)

normalized_input_data = input_data

## Perform Fast Fourier Transform (FFT) on each column
if use_fft:   
    normalized_input_data = np.apply_along_axis(lambda x: fft(x), axis=1, arr=normalized_input_data)
    normalized_input_data = normalized_input_data.real

"""
Train the models:
- Transformer Encoder
- LSTM
- Linear models: Linear Regression, Elastic Net, Bayesian Ridge
- Gaussian Process Regressor
- Decision Tree Regressor
- Ensemble models: Random Forest Regressor, AdaBoost Regressor, Gradient Boosting Regressor
"""

model_type = input("Type of model (Transformer, LSTM, linear, GP, DT, ensemble): ")

if model_type == "Transformer":
    num_transformer_blocks = config.num_transformer_blocks
    head_size = config.head_size
    num_heads = config.num_heads
    ff_dim = config.ff_dim
    dropout = config.dropout
    mlp_units = config.mlp_units
    mlp_dropout = config.mlp_dropout
    trainTransformer(normalized_input_data, 
                     input_dim, 
                     num_transformer_blocks, 
                     head_size, 
                     num_heads, 
                     ff_dim, 
                     dropout,
                     mlp_units,
                     mlp_dropout, 
                     sheet=sheet)
elif model_type == "LSTM":
    train_LSTM(normalized_input_data, input_dim, sheet=sheet)
elif model_type == "linear":
    model = input("Type of linear model (LR, EN, BR): ")
    if model == "LR":
        linearModels(normalized_input_data, model, sheet=sheet)
    elif model == "EN":
        linearModels(normalized_input_data, model, sheet=sheet)
    elif model == "BR":
        linearModels(normalized_input_data, model, sheet=sheet)
    else:
        print("Invalid model")
elif model_type == "GP":
    gaussianProcess(normalized_input_data, sheet=sheet)
elif model_type == "DT":
    decisionTree(normalized_input_data, sheet=sheet)
elif model_type == "ensemble":
    model = input("Type of ensemble model (RF, AB, GB): ")
    if model == "RF":
        ensembleModels(normalized_input_data, model, sheet=sheet)
    elif model == "AB":
        ensembleModels(normalized_input_data, model, sheet=sheet)
    elif model == "GB":
        ensembleModels(normalized_input_data, model, sheet=sheet)
else:
    raise Exception("Model type not valid")

