import copy
import math
import os
import pickle
import sys

import pandas as pd
import numpy as np
import tensorflow as tf

from scipy.fft import fft, ifft
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from config import Config
from data_prep import DataPrep, series_to_supervised

config = Config()

use_fft = True

sheets = ["One-Month", "Two-Month", "Three-Month"]
sheet_cntr = int(input('Choose xlsx sheet (0: One-Month, 1: Two-Month, 2: Three-Month): '))
sheet = sheets[sheet_cntr]

data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+"/Data_for_Frequency_Domain.xlsx"
model_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+"/saved_models/"

df = pd.read_excel(data_path,
               sheet_name=sheet,
               names=["Returns","Price/Dividedn","Lag Dividednt Growth","Risk-free Rate","Lag Returns"],
               usecols="F:J",
               #skipfooter=4
                  )

# Input dimension (accommodates the four main columns plus the Returns column)
input_dim = df.shape[1] + 1

# Scale the data {0,1}
input_data = df.values
#scaler = MinMaxScaler(feature_range=(0, 1))
#normalized_input_data = scaler.fit_transform(input_data)
normalized_input_data = input_data

if use_fft:
    ## Perform Fast Fourier Transformation (FFT) on each column
    normalized_input_data = np.apply_along_axis(lambda x: fft(x), axis=1, arr=normalized_input_data)
    normalized_input_data = normalized_input_data.real

def predModel(algorithm, inputs):
    # Load the model
    try:
        with open(model_path+f'{algorithm}_returns_{sheet}.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        preds = loaded_model.predict(inputs)  
        return preds
    except:
        print(f"{algorithm} does not exist")
        return []        
        

#model_type = input("Type of model (transformer, lstm, lr, en, br, gp, dt, rf, ab, gb): ")
rmses = []
r2s = []

df_preds = pd.DataFrame({"Bayesian Ridge":[],
                         "Linear Regression":[]
                        })

plot_results = False

models = ['transformer', 'lstm', 'lr', 'en', 'br', 'gp', 'dt', 'rf', 'ab', 'gb']

for model_type in models:
    if model_type == "lstm":
        # Load encoder and decoder
        try:
            lstm_model = tf.keras.models.load_model(
                model_path+f"/LSTM_returns_{sheet}.keras", 
                compile=False,
                safe_mode=False
            )
        except:
            print("LSTM does not exist")
            rmses.append(float('NaN'))
            r2s.append(float('NaN'))
        else:
            reframed = series_to_supervised(normalized_input_data, config.sequence_length, 1)
        
            # Drop the Returns column (1st)
            reframed.drop(reframed.columns[input_dim:], axis=1, inplace=True)
            
            # Split into train and test sets (80-20)
            test_split = config.test_split
            values = reframed.values
            test = values[int(len(values)*test_split):, :]
            
            # Split into input and outputs, and reshape
            test_input, test_output = test[:, 1:], test[:, 0]
            test_input = test_input.reshape((test_input.shape[0], config.sequence_length, test_input.shape[1]))
            print(test_input.shape, test_output.shape)
            
            preds = lstm_model.predict(test_input)
            
            # plot results
            if plot_results:
                plt.plot(preds, label="predicted")
                plt.plot(test_output, label="true")
                plt.legend()
                plt.show()
        
            # calculate RMSE, R2 score
            rmse = math.sqrt(mean_squared_error(test_output, preds))
            r2 = r2_score(test_output, preds)
            print("LSTM results")
            print(f'RMSE: {rmse:.3f}; R2 score: {r2:.3f}')
            rmses.append(rmse)
            r2s.append(r2)
    elif model_type == "transformer":
        # Load encoder and decoder
        try:
            lstm_model = tf.keras.models.load_model(
                model_path+f"/Transformer_returns_{sheet}.keras", 
                compile=False,
                safe_mode=False
            )
        except:
            print("Transformer does not exist")
            rmses.append(float('NaN'))
            r2s.append(float('NaN'))
        else: 
            reframed = series_to_supervised(normalized_input_data, config.sequence_length, 1)
        
            # Drop the Returns column (1st)
            reframed.drop(reframed.columns[input_dim:], axis=1, inplace=True)
            values = reframed.values
            # Split into train and test sets (80-20)
            test_split = config.test_split
            values = reframed.values
            test = values[int(len(values)*test_split):, :]
            
            # Split into input and outputs, and reshape
            test_input, test_output = test[:, 1:], test[:, 0]
            test_input = test_input.reshape((test_input.shape[0], config.sequence_length, test_input.shape[1]))
            print(test_input.shape, test_output.shape)
            
            preds = lstm_model.predict(test_input)
            
            # plot results
            if plot_results:
                plt.plot(preds, label="predicted")
                plt.plot(test_output, label="true")
                plt.legend()
                plt.show()
        
            # calculate RMSE, R2 score
            rmse = math.sqrt(mean_squared_error(test_output, preds))
            r2 = r2_score(test_output, preds)
            print(f"{model_type.upper()} results")
            print(f'RMSE: {rmse:.3f}; R2 score: {r2:.3f}')
            rmses.append(rmse)
            r2s.append(r2)
    else:
        test_split = config.test_split
        test = normalized_input_data[int(len(normalized_input_data)*test_split):, :]
        
        # Split into input and outputs
        test_input, test_output = test[:, 1:], test[:, 0]
        
        # Perform predictions, return them to 'preds'
        preds = predModel(model_type.upper(), test_input)

        if len(preds) > 0:
            if model_type == 'lr':
                preds_time_domain = ifft(preds)
                df_preds['Linear Regression'] = preds
            elif model_type == 'br':
                preds_time_domain = ifft(preds)
                df_preds['Bayesian Ridge'] = preds
        
            # plot results
            if plot_results:
                plt.plot(preds, label="predicted")
                plt.plot(test_output, label="true")
                plt.legend()
                plt.show()
            
            # calculate RMSE, R2 score
            rmse = math.sqrt(mean_squared_error(test_output, preds))
            r2 = r2_score(test_output, preds)
            print(f"{model_type.upper()} results")
            print(f'RMSE: {rmse:.3f}; R-squared: {r2:.3f}')
            rmses.append(rmse)
            r2s.append(r2)
        else:
            rmses.append(float('NaN'))
            r2s.append(float('NaN'))

index_values = ['Transformer Encoder', 'LSTM', 'Linear Regression', 'Elastic Net', 
                'Bayesian Ridge Regression', 'Gaussian Process', 'Decision Tree', 
                'Random Forest', 'AdaBoost', 'GradientBoost']

results = pd.DataFrame({"RMSE":rmses, "R2":r2s}, index=index_values)
results.sort_values(by=['RMSE','R2'],ascending=True).round(3)

######### Bayesian Ridge Regression #########

normalized_input_data_copy = copy.deepcopy(normalized_input_data)
normalized_input_data_copy[int(len(df)*config.test_split):,0] = df_preds['Bayesian Ridge'].values
reconstructed_data = np.apply_along_axis(lambda x: ifft(x), axis=1, arr=normalized_input_data)
reconstructed_data_copy = np.apply_along_axis(lambda x: ifft(x), axis=1, arr=normalized_input_data_copy)

df_reconstructed = pd.DataFrame({"Returns":reconstructed_data[:,0].real,
                                 "Price/Dividedn":reconstructed_data[:,1].real,
                                 "Lag Dividednt Growth":reconstructed_data[:,2].real,
                                 "Risk-free Rate":reconstructed_data[:,3].real,
                                 "Lag Returns":reconstructed_data[:,4].real
                                })
#df_reconstructed.plot()

df_reconstructed_copy = pd.DataFrame({"Returns":reconstructed_data_copy[:,0].real,
                                      "Price/Dividedn":reconstructed_data_copy[:,1].real,
                                      "Lag Dividednt Growth":reconstructed_data_copy[:,2].real,
                                      "Risk-free Rate":reconstructed_data_copy[:,3].real,
                                      "Lag Returns":reconstructed_data_copy[:,4].real
                                     })	
#df_reconstructed_copy.plot()

preds = df_reconstructed_copy['Returns'].iloc[int(len(df_reconstructed_copy)*config.test_split):]
trues = df_reconstructed['Returns'].iloc[int(len(df_reconstructed)*config.test_split):]

plt.plot(preds, label="Predicted")
plt.plot(trues, label="True")
plt.legend()
plt.title("Bayesian Ridge Regression estimates vs. True values")
plt.plot()
plt.show()

######## Linear Regression #########

normalized_input_data_copy = copy.deepcopy(normalized_input_data)
normalized_input_data_copy[int(len(df)*config.test_split):,0] = df_preds['Linear Regression'].values
reconstructed_data = np.apply_along_axis(lambda x: ifft(x), axis=1, arr=normalized_input_data)
reconstructed_data_copy = np.apply_along_axis(lambda x: ifft(x), axis=1, arr=normalized_input_data_copy)

df_reconstructed = pd.DataFrame({"Returns":reconstructed_data[:,0].real,
                                 "Price/Dividedn":reconstructed_data[:,1].real,
                                 "Lag Dividednt Growth":reconstructed_data[:,2].real,
                                 "Risk-free Rate":reconstructed_data[:,3].real,
                                 "Lag Returns":reconstructed_data[:,4].real
                                })
#df_reconstructed.plot()

df_reconstructed_copy = pd.DataFrame({"Returns":reconstructed_data_copy[:,0].real,
                                      "Price/Dividedn":reconstructed_data_copy[:,1].real,
                                      "Lag Dividednt Growth":reconstructed_data_copy[:,2].real,
                                      "Risk-free Rate":reconstructed_data_copy[:,3].real,
                                      "Lag Returns":reconstructed_data_copy[:,4].real
                                     })	
#df_reconstructed_copy.plot()

preds = df_reconstructed_copy['Returns'].iloc[int(len(df_reconstructed_copy)*config.test_split):]
trues = df_reconstructed['Returns'].iloc[int(len(df_reconstructed)*config.test_split):]

plt.plot(preds, label="Predicted")
plt.plot(trues, label="True")
plt.legend()
plt.title("Linear Regression estimates vs. True values")
plt.plot()
plt.show()