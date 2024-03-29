import os
import numpy as np
import pandas as pd
import ray

from copy import deepcopy
from config import Config

import matplotlib.pyplot as plt

def plotting(data):
    norm_data, freq_data = data
    # Plot the original inflation rate time series
    plt.figure(figsize=(12, 4))
    plt.plot(norm_data)
    plt.title('Inflation Rate Time Series')
    plt.xlabel('Time')
    plt.ylabel('Inflation Rate')
    plt.show()

    # Plot the spectrogram
    plt.figure(figsize=(12, 4))
    plt.plot(freq_data)  # Using log scale for better visualization
    plt.title('STFT - Spectrogram of Inflation Rate')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

# Generator function for creating sequences
def gen_seq(id_df, seq_length, seq_cols):

    data_matrix =  id_df[seq_cols]
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length, 1), range(seq_length, num_elements, 1)):
        
        yield data_matrix[stop-seq_length:stop].values.reshape((-1,len(seq_cols)))

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frames a dataframe as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.

    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    n_vars = data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def return_to_domain(df, epsilon):
    df['FEDFUNDS'] = df['FEDFUNDS'].map(lambda p: (np.where(p==1, epsilon, np.log((p + epsilon)/(1-p + epsilon)) / 100)))
    df['Inflation_1'] = df['Inflation_1'].map(lambda p: np.log((p)/(1-p + epsilon)))
    df['Output_GAP'] = df['Output_GAP'].map(lambda p: np.log((p)/(1-p + epsilon)))
    return df

class DataPrep:
    def __init__(self) -> None:
        self.config = Config()
        self.e = 1e-7
    
    def col_names(self):
        return {
            #'Inflation 1 Implicit GDP price deflator (%)': "inflation_1"+suffix, 
            #"Inflation 2 PCE Index (%)": "inflation_2"+suffix,
            "FEDFUNDS (%)": "fedfunds",
            "Output GAP (%)": "output_gap",
            "Inflation 2 PCE Index (%) residuals": "inflation_residuals"
            }
    
    def read_data(self, specifications_set, read_only: bool = False):
        # Read data
        scaler = None
        specifications_set = specifications_set.upper()

        if specifications_set == 'A':
            column_names = ['Inflation_1',
                            'FEDFUNDS',
                            'Output_GAP',
                            #'Natural_Rate_of_Interest',
                        ]

            df = pd.read_excel(self.config.path_to_data,
                            sheet_name="FRED Graph",
                            skiprows=range(9),
                            names=column_names,
                            usecols='B,C,D', #,N
                            #skipfooter=21
                            )
            col_order = ['FEDFUNDS', 'Inflation_1', 'Output_GAP']
            df = df[col_order]
            df.dropna(inplace=True)
            #print(df)
            df_interest_rate = df['FEDFUNDS']
            df.drop(columns=['FEDFUNDS'], inplace=True)
            """
            plot = df.plot()
            fig = plot.get_figure()
            fig.savefig('../../Autonomous_Fed/results/input_data.png')
            """
            #df["FEDFUNDS"] = df["FEDFUNDS"].map(lambda x: (1/(1 + np.exp(-x))))
            #df["Output_GAP"] = df["Output_GAP"].map(lambda x: (1/(1 + np.exp(-x))))
            #df["Inflation_1"] = df["Inflation_1"].map(lambda x: (1/(1 + np.exp(-x))))                                                                   
            #df["Natural_Rate_of_Interest"] = df["Natural_Rate_of_Interest"].map(lambda x: (1+x/100))
            
        elif specifications_set == 'B':
            column_names = ['FEDFUNDS',
                            'CPI_Inflation',
                            'Output_GAP'
                        ]

            df = pd.read_excel(self.config.path_to_data,
                            sheet_name="FRED Graph",
                            skiprows=range(9),
                            names=column_names,
                            usecols='D,K,E',
                            skipfooter=21
                            )

            df.dropna(inplace=True)
            df["FEDFUNDS"] = df["FEDFUNDS"].map(lambda x: (1+x)/100)
            df["Output_GAP"] = df["Output_GAP"].map(lambda x: (1+x)/100)
            df["CPI_Inflation"] = df["CPI_Inflation"].map(lambda x: (1+x)/100)
        elif specifications_set == 'C':
            column_names = ['FEDFUNDS',
                            'log_GDP',
                            'log_potential_GDP',
                            'log_CPI',
                            #'Natural_Rate_of_Interest',
                        ]

            df = pd.read_excel(self.config.path_to_data,
                            sheet_name="FRED Graph",
                            skiprows=range(9),
                            names=column_names,
                            usecols='C,I:K', #,N
                            skipfooter=21
                            )

            df.dropna(inplace=True)
        
        if specifications_set in ['A', 'C']:
            from sklearn.preprocessing import MinMaxScaler

            if not os.path.exists('../results'):
                os.makedirs('../../Autonomous_Fed/results')

            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_input_data = scaler.fit_transform(df)
            df = pd.DataFrame(normalized_input_data, columns=['Inflation_1', 'Output_GAP'])
            df.reset_index(drop=True, inplace=True)
            df_interest_rate.reset_index(drop=True, inplace=True)
            
            plot = df.plot()
            fig = plot.get_figure()
            fig.savefig('../../Autonomous_Fed/results/normalized_input_data.png')
            plt.clf()
            plot_ir = df_interest_rate.plot()
            fig_ir = plot_ir.get_figure()
            fig_ir.savefig('../../Autonomous_Fed/results/interest_rate.png')
            #return df
            """"""
        return df, df_interest_rate, scaler
    
    def inverse_transform(self, data):
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.inverse_transform(data)

if __name__ == "__main__":
    epsilon = 1e-10
    data_prep = DataPrep()
    specifications_set = input("Choose specifications set: {A, B, C}: ")
    df, df_interest_rate, scaler = data_prep.read_data(specifications_set=specifications_set)
    #df.reset_index(drop=True, inplace=True)
    print(df)
    print(df_interest_rate)
    #print(df.shape)
    """
    df_copy = df.copy()
    df_copy.reset_index(drop=True, inplace=True)
    df_copy['FEDFUNDS'] = df_copy['FEDFUNDS'].map(lambda p: np.log((p + epsilon)/(1-p + epsilon)))
    df_copy['Inflation_1'] = df_copy['Inflation_1'].map(lambda p: np.log((p + epsilon)/(1-p + epsilon)))
    df_copy['Output_GAP'] = df_copy['Output_GAP'].map(lambda p: np.log((p + epsilon)/(1-p + epsilon)))
    print(df_copy)  
    plt.plot(df)
    plt.show()
    """
