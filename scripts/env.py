import logging
import math
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

os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"

logger = logging.getLogger(__name__)

class AutonomousFed(gymnasium.Env):
    cntr = 0
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model_config = config['model_config']
        self.df = config['df']
        self.scaler = config['scaler']
        self.columns = self.df.columns
        #self.original_df = self.df.copy()

        self.model_type = config['model_type']

        # Define start and end dates (this is arbitrary)
        self.start_date = config['start_date']
        self.end_date = config['end_date']

        # Create quarterly datetime range
        self.quarterly_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='QS-JAN')

        self.omega_pi = config['omega_pi']
        self.omega_psi = config['omega_psi']

        logger.info(
            f"Specifications set {config['specifications_set']},omega_pi = {self.omega_pi},\
            omega_psi = {self.omega_psi}"
        )

        self.specifications_set = config['specifications_set']

        self.prev_states = []

        self.epsilon = 1e-10

        low = 0
        high = 100

        if self.config['action_specifications'] == 'ir_omega_equals':
            self.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float16)
        elif self.config['action_specifications'] == 'ir_omega_not_equals':
            self.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float16)
        elif self.config['action_specifications'] == 'ir_omega_pi_action':
            self.action_space = spaces.Dict({
                "interest_rate":spaces.Box(low=low, high=high, shape=(1,), dtype=np.float16),
                "omega_psi":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float16),
            })
        else:
            self.action_space = spaces.Dict({
                "interest_rate":spaces.Box(low=low, high=high, shape=(1,), dtype=np.float16),
                "omega_pi":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float16),
                "omega_psi":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float16),
            })
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float16)

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
        df_copy = self.df.copy()
        #df_copy['FEDFUNDS'] = df_copy['FEDFUNDS'].map(lambda p: (np.where(p==1, self.epsilon, np.log((p + self.epsilon)/(1-p + self.epsilon)) / 100)))
        df_copy['Inflation_1'] = df_copy['Inflation_1'].map(lambda p: np.log((p + self.epsilon)/(1-p + self.epsilon)))
        df_copy['Output_GAP'] = df_copy['Output_GAP'].map(lambda p: np.log((p + self.epsilon)/(1-p + self.epsilon)))
        obs = np.array(df_copy.iloc[len(df_copy)-1, 1:], dtype=np.float16)
        

        return obs, {'2021-07-01': obs}#

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

        # Step 1: Action to list
        # If action is a dict, then it is a dict of the form {'interest_rate': 0.5, 'omega_pi': 0.5, 'omega_psi': 0.5}
        # Otherwise, it's only about the interest rate
        if self.config['action_specifications'] == 'ir_omega_equals' or \
           self.config['action_specifications'] == 'ir_omega_not_equals':
            action_ir = action.tolist()
        else:
            action_ir = action['interest_rate'].tolist()
        
        # Step 2: Get previous state 
        prev_state = self.df.iloc[len(self.df)-1,1:].values.tolist()
       
        # Step 3: transform HTTP request -> tensorflow input
        obs = requests.get(
                "http://localhost:8000/saved_models", 
                 json={"array": 
                         np.array(action_ir+prev_state).reshape(1,self.df.shape[1]).tolist()
                    }
            )
        
        # Step 4: tensorflow input -> tensorflow output
        obs = obs.json()['prediction'][0]
        
        # Add the new observation for t+1 to the dataframe
        new_row = pd.DataFrame(np.array(action_ir+obs).reshape(1, self.df.shape[1]), columns=self.df.columns)
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        AutonomousFed.cntr += 1

        # Step 5: Reward, terminated, truncated, info
        if self.config['specifications_set'] == 'A':
            df_copy = self.df.copy()
            df_copy['Inflation_1'] = df_copy['Inflation_1'].map(lambda p: np.log((p + self.epsilon)/(1-p + self.epsilon)))
            df_copy['Output_GAP'] = df_copy['Output_GAP'].map(lambda p: np.log((p + self.epsilon)/(1-p + self.epsilon)))
            obs = np.array(df_copy.iloc[len(df_copy)-1, 1:], dtype=np.float16)
            if self.config['action_specifications'] == 'ir_omega_equals' or \
            self.config['action_specifications'] == 'ir_omega_not_equals':
                reward = self._reward(
                    obs,
                    self.omega_pi, 
                    self.omega_psi
                    )
            elif self.config['action_specifications'] == 'ir_omega_pi_action':
                reward = self._reward(
                    obs, 
                    self.omega_pi, 
                    action['omega_psi'][0]
                    )
            else:
                reward = self._reward(
                    obs, 
                    action['omega_pi'][0], 
                    action['omega_psi'][0]
                    )
        elif self.config['specifications_set'] == 'C':
            if self.config['action_specifications'] == 'ir_omega_equals' or \
            self.config['action_specifications'] == 'ir_omega_not_equals':
                reward = self._reward(
                    obs, 
                    self.omega_pi, 
                    self.omega_psi, 
                    self.df.iloc[len(self.df)-4, 3]
                    )
            elif self.config['action_specifications'] == 'ir_omega_pi_action':
                reward = self._reward(
                    obs, 
                    self.omega_pi, 
                    action['omega_psi'][0], 
                    self.df.iloc[len(self.df)-4, 3]
                    )
            else:
                reward = self._reward(
                    obs, 
                    action['omega_pi'][0], 
                    action['omega_psi'][0], 
                    self.df.iloc[len(self.df)-4, 3]
                    )
        terminated = False
        truncated = {}
        info = {self.quarterly_dates[AutonomousFed.cntr-1]: obs}

        # Step 6: Set terminated to True if cntr == len(self.quarterly_dates)
        if AutonomousFed.cntr == len(self.quarterly_dates):
            terminated = True
            AutonomousFed.cntr = 0
            self.prev_states = []

        return obs, reward, terminated, truncated, info
    
    def _reward(self, obs, a_pi, a_psi, cpi_t_minus_four=None):
        """
        Reward defined using Taylor Rule for monetary policy:

        fedfunds_t = inflation1_t + a_pi * (inflation_1 - desired_inflation) + a_psi * (log_GDP_gap ** 2)

        Penalty for deviation from desired inflation: 10 * (inflation_1 - desired_inflation) ** 2 if abs(inflation_1 - desired_inflation) > 0.02
        Penalty for deviation from output gap: 10 * (log_GDP_gap ** 2) if abs(log_GDP_gap) > 0.02
        
        - Desired inflation = 0.02
        """
         
        desired_inflation = 2 # 2% desired inflation
        if self.specifications_set == 'A':
            output_gap = obs[1]
            inflation_diff = abs(obs[0] - desired_inflation)
            r_pi = (inflation_diff ** 2)
            r_psi = (output_gap ** 2)
            r_pi_penalty = 10 * r_pi if r_pi > 0.02 ** 2 else 0
            r_psi_penalty = 10 * r_psi if r_psi > 0.02 ** 2 else 0
            reward = - (a_pi * r_pi + a_psi * r_psi + r_pi_penalty + r_psi_penalty)
        elif self.specifications_set == 'C':
            output_gap = np.exp(abs(obs[0] - obs[1]))
            cpi_inflation_diff = np.exp(abs(obs[2] - cpi_t_minus_four))
            reward = - (a_pi * (cpi_inflation_diff ** 2) + a_psi * (output_gap** 2))# + obs[3]

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
    encoder_path = os.path.join('/home/erskordi/projects/Autonomous_Fed/saved_models/',f'encoder_FedModel_{specifications_set}.keras')
    decoder_path = os.path.join('/home/erskordi/projects/Autonomous_Fed/saved_models/',f'decoder_FedModel_{specifications_set}.keras')
    path = [encoder_path, decoder_path]

    # Deploy the models
    serve.run(target=TF_VAE_Model.bind(path),logging_config={"log_level": "ERROR"})
    

    config = {'start_date': '2021-07-01', 
              'end_date': '2050-12-31', 
              'model_type': 'VAE',
              'action_specifications': 'ir_omega_equals',
              'omega_pi': 0.5,
              'omega_psi': 0.5,
              'specifications_set': specifications_set,
              'scaler': None,
              'df': DataPrep().read_data(specifications_set=specifications_set)[0],
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
        print(f'Action: {action}, Observation: {obs}, Reward: {reward}')
        #print(f'Info:\n {info}')
        print()

        if terminated or truncated:
            obs, info = env.reset()

