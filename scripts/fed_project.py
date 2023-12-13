import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.insert(0, "/Users/erotokritosskordilis/git-repos/Autonomous_Fed/")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from config import Config
from data_prep import DataPrep, gen_seq, series_to_supervised, plotting

from forecasting_models import (   
    trainTransformer,
    train_LSTM, 
    linearModels, 
    gaussianProcess, 
    decisionTree, 
    ensembleModels
)


# Load configuration file with necessary parameters
config = Config()
data_prep = DataPrep()

# Load data
specifications_set = int(input("Choose specifications set (0, 1, 2): "))
df = data_prep.read_data(specifications_set=specifications_set)

# Input dimension (only for LSTM & Transformer)
input_dim = df.shape[1]

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
    trainTransformer(df, 
                     input_dim, 
                     num_transformer_blocks, 
                     head_size, 
                     num_heads, 
                     ff_dim, 
                     dropout,
                     mlp_units,
                     mlp_dropout,
                     specifications_set)
elif model_type == "LSTM":
    train_LSTM(df, input_dim, specifications_set)
elif model_type == "linear":
    model = input("Type of linear model (LR, EN, BR): ")
    if model == "LR":
        linearModels(df, specifications_set, model)
    elif model == "EN":
        linearModels(df, model)
    elif model == "BR":
        linearModels(df, model)
    else:
        print("Invalid model")
elif model_type == "GP":
    gaussianProcess(df)
elif model_type == "DT":
    decisionTree(df.values, specifications_set)
elif model_type == "ensemble":
    model = input("Type of ensemble model (RF, AB, GB): ")
    if model == "RF":
        ensembleModels(df.values, specifications_set, model)
    elif model == "AB":
        ensembleModels(df.values, model)
    elif model == "GB":
        ensembleModels(df.values, model)
else:
    raise Exception("Model type not valid")

