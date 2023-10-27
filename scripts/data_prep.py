import pandas as pd
import ray
from ray.data.preprocessors import MinMaxScaler, Concatenator

from copy import deepcopy
from config import Config

class DataPrep:
    def __init__(self) -> None:
        self.config = Config()
    
    def col_names(self, suffix: str):
        return {
            #'Inflation 1 Implicit GDP price deflator (%)': "inflation_1"+suffix, 
            #"Inflation 2 PCE Index (%)": "inflation_2"+suffix,
            "FEDFUNDS (%)": "fedfunds",
            "Output GAP (%)": "output_gap",
            "Inflation 2 PCE Index (%) residuals": "inflation_residuals"
            }

    def scaling(self, columns: list):
        return MinMaxScaler(
            columns=columns
        )
    
    def read_data(self):
        # Read data
        df = pd.read_csv(self.config.path_to_data)
        df = df.drop(columns=['Inflation 1 Implicit GDP price deflator (%)',
                              'Inflation 2 PCE Index (%)',
                              'Frequency: Quarterly observation_date',
                              'GDP (Billion $)',
                              'Potential GDP (Billion $)'])
        df["FEDFUNDS (%)"] = df["FEDFUNDS (%)"].map(lambda x: (1+x)/100)
        df["Output GAP (%)"] = df["Output GAP (%)"].map(lambda x: (1+x)/100)
        df["Inflation 2 PCE Index (%) residuals"] = df["Inflation 2 PCE Index (%) residuals"].map(lambda x: (1+x)/100)

        df.rename(columns=self.col_names(self.config.suffix_prior), inplace=True)
        
        train_dataset = ray.data.from_pandas(df)

        preprocessor = self.scaling(columns=self.col_names(self.config.suffix_prior).values())
        
        df = preprocessor.fit_transform(train_dataset).to_pandas()

        # Split df to train and validation sets
        #split_point = int(len(df)*0.8)
       # train_dataset = df[:split_point]
        #validation_dataset = df[split_point:]

        #df = ray.data.from_pandas(train_dataset)
        #df = Concatenator(output_column_name="features").fit_transform(df)#.to_pandas()

        return df#train_dataset, validation_dataset

if __name__ == "__main__":
    
    data_prep = DataPrep()
    df = data_prep.read_data()
