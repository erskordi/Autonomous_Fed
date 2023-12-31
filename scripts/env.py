import logging
import numpy as np
import os
import pandas as pd
import requests
import tensorflow as tf

import ray
from ray import tune, serve
from ray.rllib.env import MultiAgentEnv

import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding

from config import Config
from data_prep import gen_seq, series_to_supervised, plotting, DataPrep
from sim import TF_VAE_Model

logger = logging.getLogger(__name__)

class AutonomousFed(gymnasium.Env):
    cntr = 0
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model_config = config['model_config']
        self.df = config['df']
        self.original_df = self.df.copy()

        self.model_type = config['model_type']

        # Define start and end dates (this is arbitrary)
        self.start_date = config['start_date']
        self.end_date = config['end_date']

        # Create quarterly datetime range
        self.quarterly_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='QS-JAN')

        self.a_pi = config['a_pi']
        self.a_psi = config['a_psi']

        logger.info(f"Specifications set {config['specifications_set']},a_pi = {self.a_pi}, a_psi = {self.a_psi}")

        self.specifications_set = config['specifications_set']

        self.prev_states = []

        self.action_space = spaces.Dict({
            "interest_rate":spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
            "a_pi":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "a_psi":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.df.shape[1]-1,), dtype=np.float32)
    
    def dates(self):
        for date in self.quarterly_dates:
            yield date 

    def reset(self, *, seed=None, options=None):
        """
        reset() as described in the OpenAI Gymnasium API (RLlib uses this API)
        
        Requires the following:
        - self
        - * 
        - seed
        - options

        # Initialize the state of the environment
        # and return the initial observation
        """
        super().reset(seed=seed, options=options)
        self.df = self.original_df.copy()
        obs = np.array(self.df.iloc[len(self.df)-1, 1:], dtype=np.float32)

        return obs, {'2021-04-01': obs}#

    def step(self, action):
        """
        'done' has been deprecated in the OpenAI Gymnasium API (RLlib uses this API)
        Instead, step() returns the following:
        - obs: observation
        - reward
        - terminated
        - truncated
        - info
        """
        # Create quarterly datetime range using generator method
        date_gen = self.dates()

        # Step 1: Action to list
        action_ir = action['interest_rate'].tolist()

        # Step 2: Get previous state 
        prev_state = self.df.iloc[len(self.df)-1, 1:].values.tolist()
       
        # Step 3: transform HTTP request -> tensorflow input
        obs = requests.get(
                "http://localhost:8000/saved_models", 
                 json={"array": 
                         np.array(action_ir+prev_state).reshape(1,self.df.shape[1]).tolist()
                    }
            )
        
        # Step 4: tensorflow input -> tensorflow output
        obs = obs.json()['prediction'][0]
        #print("action+obs:", np.array(action+obs))
        # Add the new observation for t+1 to the dataframe
        self.df.loc[len(self.df)] = np.array(action_ir+obs)
        AutonomousFed.cntr += 1

        # Step 5: Reward, terminated, truncated, info
        reward = self._reward(
            obs, self.df.iloc[len(self.df)-4, 3], 
            action['a_pi'][0], action['a_psi'][0])
        terminated = False
        truncated = {}
        info = {next(date_gen): obs}

        # Step 6: Set terminated to True if cntr == len(self.quarterly_dates)
        if AutonomousFed.cntr == len(self.quarterly_dates):
            terminated = True
            AutonomousFed.cntr = 0
            self.prev_states = []

        return obs, reward, terminated, truncated, info
    
    def _reward(self, obs, cpi_t_minus_three, a_pi, a_psi):
        """
        Reward defined using Taylor Rule for monetary policy:

        fedfunds_t = inflation1_t + a_pi * (inflation_1 - desired_inflation) + a_psi * (log_GDP_gap ** 2)
        - Desired inflation = 0.02
        """
        if self.specifications_set == 'A':
            output_gap = obs[1]
            inflation_diff = obs[0] - 0.02
            reward = - (a_pi * inflation_diff + a_psi * (output_gap ** 2) + obs[2])
        elif self.specifications_set == 'C':
            output_gap = abs(np.exp(obs[0] - obs[1]) - 1)
            cpi_inflation_diff = abs(np.exp(obs[2] - cpi_t_minus_three) - 1)
            reward = - (a_pi * cpi_inflation_diff + a_psi * (output_gap** 2) + obs[3])

        return reward
    
    def render(self, mode="human"):
        """
        render() as described in the OpenAI Gymnasium API (RLlib uses this API)
        """
        super().render(mode=mode)


if __name__ == "__main__":

    specifications_set = input("Choose specifications set: {A, B, C}: ")
    # Initialize Ray Serve
    serve.start()

    # Load the models based on the specifications set
    encoder_path = os.path.join('/Users/erotokritosskordilis/git-repos/Autonomous_Fed/saved_models/',f'encoder_FedModel_{specifications_set}.keras')
    decoder_path = os.path.join('/Users/erotokritosskordilis/git-repos/Autonomous_Fed/saved_models/',f'decoder_FedModel_{specifications_set}.keras')
    path = [encoder_path, decoder_path]

    # Deploy the models
    serve.run(target=TF_VAE_Model.bind(path),logging_config={"log_level": "ERROR"})
    

    config = {'start_date': '2021-07-01', 
              'end_date': '2050-12-31', 
              'model_type': 'VAE',
              'a_pi': 0.5,
              'a_psi': 0.5,
              'specifications_set': specifications_set,
              'df': DataPrep().read_data(specifications_set=specifications_set),
              'model_config': Config()}

    episode_length = len(pd.date_range(start=config['start_date'], end=config['end_date'], freq='QS-JAN'))

    env = AutonomousFed(config)

    # Environment sanity test
    obs = env.reset()
    print(f'Observation: {obs}')

    # Test step
    for _ in range(episode_length):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f'Observation: {obs}')
        print(f'Reward: {reward}')
        print(f'Info:\n {info}')

        if terminated or truncated:
            obs, info = env.reset()

