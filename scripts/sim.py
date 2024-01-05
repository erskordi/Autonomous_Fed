import os
import requests
import sys
import tempfile

path_to_model = "/Users/erotokritosskordilis/git-repos/Autonomous_Fed/saved_models/LSTM.keras"

sys.path.insert(0, "/Users/erotokritosskordilis/git-repos/Autonomous_Fed/saved_models")

import numpy as np
import pandas as pd
import ray

from ray import serve

from config import Config

@serve.deployment
class TF_LSTM_Model:
    """
    Use Ray serve to deploy the LSTM model
    """
    def __init__(self, model_path):
        import tensorflow as tf
        
        self.model_path = model_path
        self.lstm_model = tf.keras.models.load_model(self.model_path)
    
    async def __call__(self, starlette_request):
        
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        input_array = np.array((await starlette_request.json())["array"])
        #print(f'Input array: {input_array}')

        # Step 2: tensorflow input -> tensorflow output
        prediction = self.lstm_model(input_array)
       

        # Step 3: tensorflow output -> web output
        return {"prediction": prediction.numpy().tolist(), "file": self.model_path}

@serve.deployment
class TF_VAE_Model:
    """
    Use Ray serve to deploy the LSTM model
    """
    def __init__(self, model_path):
        import tensorflow as tf
        
        self.model_path = model_path
        self.encoder_model = tf.keras.models.load_model(self.model_path[0], safe_mode=False)
        self.decoder_model = tf.keras.models.load_model(self.model_path[1], safe_mode=False)
    
    async def __call__(self, starlette_request):
        
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        input_array = np.array((await starlette_request.json())["array"])
        #print(f'Input array: {input_array}')

        # Step 2: tensorflow input -> tensorflow output
        prediction = self.decoder_model(self.encoder_model(input_array))

        # Step 3: tensorflow output -> web output
        return {"prediction": prediction.numpy().tolist(), "file": self.model_path}

if __name__ == "__main__":

    config = Config()
    # Start the Ray Serve instance
    ray.init()
    serve.start()

    specifications_set = int(input("Choose specifications set: {0, 1, 2, 3} (3 is only for VAE): "))
    model_type = input("Choose model type (LSTM, VAE): ")

    if model_type == 'LSTM':
        path = os.path.join('/Users/erotokritosskordilis/git-repos/Autonomous_Fed/saved_models/',f'LSTM_FedModel_{specifications_set}.keras')
        

        # Deploy the model
        TF_LSTM_Model.deploy(path)

        serve.run(target=TF_LSTM_Model.bind(path))

        inputs_size = [4, 3, 4] 

        resp = requests.get(
            "http://localhost:8000/saved_models", 
            json={"array": np.random.uniform(
                0,1,inputs_size[specifications_set]
                ).reshape(1,config.sequence_length,inputs_size[specifications_set]).tolist()}
        )

        print(resp.json()['prediction'][0])
    else:
        encoder_path = os.path.join('/Users/erotokritosskordilis/git-repos/Autonomous_Fed/saved_models/',f'encoder_FedModel_{specifications_set}.keras')
        decoder_path = os.path.join('/Users/erotokritosskordilis/git-repos/Autonomous_Fed/saved_models/',f'decoder_FedModel_{specifications_set}.keras')
        path = [encoder_path, decoder_path]

        # Deploy the model
        TF_VAE_Model.deploy(path)

        serve.run(target=TF_VAE_Model.bind(path))

        inputs_size = [4, 3, 4, 4] 

        resp = requests.get(
            "http://localhost:8000/saved_models", 
            json={"array": np.random.uniform(
                0,1,inputs_size[specifications_set]
                ).reshape(1,inputs_size[specifications_set]).tolist()}
        )

        print(resp.json()['prediction'][0])
