import pandas as pd
import tensorflow as tf

from config import Config
from data_prep import DataPrep
from model_training import VAE

if __name__ == "__main__":
    # Load the data
    config = Config()
    data_prep = DataPrep()
    df = data_prep.read_data()

    # Load models
    vae = VAE(input_dim=df.shape[1])

    # Load the VAE and produce simulated future steps
    sim_vae = vae.load_vae("/Users/erotokritosskordilis/git-repos/Autonomous_Fed/saved_models")

    # Simulate the future
    current_state = df[-1:].values
    new_df = pd.DataFrame(columns=['fedfunds', 'output_gap', 'inflation_residuals'])
    
    while True:
        new_state = sim_vae.predict(current_state)
        print({"current_state": current_state, 
               "next_state": new_state
               })
        
        # Make next state the current state, and append to new_df dictionary
        current_state = new_state
        new_df = new_df._append({"fedfunds": current_state[0][0], 
                                "output_gap": current_state[0][1], 
                                "inflation_residuals": current_state[0][2]}, 
                               ignore_index=True)

        if len(new_df) == 10:
            break

    print(new_df)
