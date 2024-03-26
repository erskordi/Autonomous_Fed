import logging
import math
import numpy as np
import os
import pandas as pd
import pickle
import requests
import tensorflow as tf

import ray
from ray import tune, serve
from ray.rllib.env import MultiAgentEnv

import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding

from config import Config
from data_prep import gen_seq, series_to_supervised, plotting, DataPrep, return_to_domain
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
        self.simulator = config['simulator']
        self.omega_pi = config['omega_pi']
        self.omega_psi = config['omega_psi']

        logger.info(
            f"Specifications set {config['specifications_set']},omega_pi = {self.omega_pi},\
            omega_psi = {self.omega_psi}"
        )

        with open('../../Autonomous_Fed/saved_models/rf_regressor.pkl', 'rb') as f:
            self.regr = pickle.load(f)

        self.specifications_set = config['specifications_set']

        self.use_penalty = config['use_penalty']
        self.normalization_scheme = config['normalization_scheme']

        self.prev_states = []
        self.prev_actions = []

        self.epsilon = 1e-10

        self.initial_timestep = None

        low = 0.0
        high = 1.0

        if self.config['action_specifications'] == 'ir_omega_equals':
            self.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)
        elif self.config['action_specifications'] == 'ir_omega_not_equals':
            self.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)
        elif self.config['action_specifications'] == 'ir_omega_pi_action':
            self.action_space = spaces.Dict({
                "interest_rate":spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32),
                "omega_psi":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            })
        else:
            self.action_space = spaces.Dict({
                "interest_rate":spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32),
                "omega_pi":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "omega_psi":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            })
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32)

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

        # Reset the counter to a random initial state, clear the list of previous interest rate actions
        AutonomousFed.cntr = np.random.randint(0, len(self.quarterly_dates)-2)
        self.initial_timestep = AutonomousFed.cntr
        self.prev_actions.clear()

        if self.normalization_scheme == 'minmax':
            inv_data = self.scaler.inverse_transform(self.df)
            obs = inv_data[AutonomousFed.cntr,1:]
        else:
            self.df = return_to_domain(self.df, self.epsilon)
            obs = self.df.iloc[AutonomousFed.cntr,1:].values.tolist()

        return obs, {self.quarterly_dates[AutonomousFed.cntr]: obs}#

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
        if self.config['action_specifications'] in ('ir_omega_equals', 'ir_omega_not_equals'):
            action_ir = action.tolist()
        else:
            action_ir = action['interest_rate'].tolist()
        
        self.prev_actions.append(action_ir[0])

        # Step 2: Get previous state 
        prev_state = self.df.iloc[AutonomousFed.cntr,1:].values.tolist()

        predictor_input = np.array([*action_ir, *prev_state]).reshape(1,self.df.shape[1])
       
        # Choose simulator (RF or VAE)
        if self.simulator == 'RF':
            obs = self.regr.predict(predictor_input)[0]
            # Add the new observation for t+1 to the dataframe
            inv_data = self.scaler.inverse_transform(np.array([*action_ir, *obs]).reshape(1, self.df.shape[1]))
        else:
            # Step 3: transform HTTP request -> tensorflow input
            obs = self._vae_output(action_ir, prev_state)
            # Add the new observation for t+1 to the dataframe
            inv_data = self.scaler.inverse_transform(np.array(action_ir+obs).reshape(1, self.df.shape[1]))
            df_preds = pd.DataFrame(inv_data, columns=self.df.columns)

        # Recall true state (only used if we want to penalize deviation from true state in reward function)
        true_state = self.df.iloc[AutonomousFed.cntr+1,1:].values.tolist()

        # Step 5: Reward, terminated, truncated, info
        if self.config['specifications_set'] == 'A':
            if self.normalization_scheme == 'minmax':
                obs = inv_data[0,1:]
            else:
                df_preds = return_to_domain(df_preds, self.epsilon)
                obs = df_preds.iloc[0,1:].values.tolist()
            if self.config['action_specifications'] in ('ir_omega_equals', 'ir_omega_not_equals'):
                reward = self._reward(
                    obs,
                    true_state,
                    action_ir[0],
                    self.omega_pi, 
                    self.omega_psi
                    )
            elif self.config['action_specifications'] == 'ir_omega_pi_action':
                reward = self._reward(
                    obs, 
                    true_state,
                    action_ir[0],
                    self.omega_pi, 
                    action['omega_psi'][0]
                    )
            else:
                reward = self._reward(
                    obs, 
                    true_state,
                    action_ir[0],
                    action['omega_pi'][0], 
                    action['omega_psi'][0]
                    )
        elif self.config['specifications_set'] == 'C':
            if self.config['action_specifications'] in ('ir_omega_equals', 'ir_omega_not_equals'):
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
        info = {self.quarterly_dates[AutonomousFed.cntr]: obs}

        AutonomousFed.cntr += 1
        # Step 6: Set terminated to True if cntr == len(self.quarterly_dates)
        if AutonomousFed.cntr == len(self.quarterly_dates)-1:
            terminated = True
            AutonomousFed.cntr = 0
            self.prev_states = []

        return obs, reward, terminated, truncated, info
    
    def _reward(self, obs, true_state, action_ir, a_pi, a_psi, cpi_t_minus_four=None):
        """
        Reward defined using Taylor Rule for monetary policy:

        fedfunds_t = inflation1_t + a_pi * (inflation_1 - desired_inflation) + a_psi * (log_GDP_gap ** 2)

        Penalty for deviation from desired inflation: 10 * (inflation_1 - desired_inflation) ** 2 if abs(inflation_1 - desired_inflation) > 0.02
        Penalty for deviation from output gap: 10 * (log_GDP_gap ** 2) if abs(log_GDP_gap) > 0.02
        
        - Desired inflation = 0.02
        """
        use_extra_penalty = False
        desired_inflation = 2 # 2% desired inflation
        if self.specifications_set == 'A':
            output_gap = obs[1]
            inflation_diff = abs(obs[0] - desired_inflation)
            r_pi = (inflation_diff ** 2)
            r_psi = (output_gap ** 2)
            extra_penalty = (obs[0] - true_state[0]) ** 2 + (obs[1] - true_state[1]) ** 2 if use_extra_penalty else 0
            previous_action_penalty = 100 * float(~np.isclose(
                                                    self.prev_actions[-2], 
                                                    action_ir, 
                                                    rtol=1e-01, 
                                                    atol=1e-01, 
                                                    equal_nan=False)
                                                   ) if len(self.prev_actions) > 1 else 0
            if self.use_penalty:
                r_pi_penalty = 10 * r_pi if r_pi > desired_inflation ** 2 else 0
                r_psi_penalty = 10 * r_psi if r_psi > desired_inflation ** 2 else 0
                reward = - (a_pi * r_pi + a_psi * r_psi + r_pi_penalty + r_psi_penalty)\
                 - extra_penalty - previous_action_penalty
            else:
                reward = - (a_pi * r_pi + a_psi * r_psi) - extra_penalty - previous_action_penalty
        elif self.specifications_set == 'C':
            output_gap = np.exp(abs(obs[0] - obs[1]))
            cpi_inflation_diff = np.exp(abs(obs[2] - cpi_t_minus_four))
            reward = - (a_pi * (cpi_inflation_diff ** 2) + a_psi * (output_gap** 2))# + obs[3]

        return reward
    
    def _vae_output(self, action_ir, prev_state):
        obs = requests.get(
                    "http://localhost:8000/saved_models", 
                    json={"array": 
                            np.array(action_ir+prev_state).reshape(1,self.df.shape[1]).tolist()
                        }
                )
        
        # Step 4: tensorflow input -> tensorflow output
        return obs.json()['prediction'][0]

    def render(self, mode="human"):
        """
        render() as described in the OpenAI Gymnasium API (RLlib uses this API)
        """
        super().render(mode=mode)


if __name__ == "__main__":

    specifications_set = input("Choose specifications set: {A, B, C}: ").upper()

    simulator = 'RF'
    if simulator == 'VAE':
        # Initialize Ray Serve
        serve.start()

        # Load the models based on the specifications set
        encoder_path = os.path.join('../../Autonomous_Fed/saved_models/',f'encoder_FedModel_{specifications_set}.keras')
        decoder_path = os.path.join('../../Autonomous_Fed/saved_models/',f'decoder_FedModel_{specifications_set}.keras')
        path = [encoder_path, decoder_path]

        # Deploy the models
        serve.run(target=TF_VAE_Model.bind(path),logging_config={"log_level": "ERROR"})

    df, scaler = DataPrep().read_data(specifications_set=specifications_set)

    config = {'start_date': '1954-07-01', 
              'end_date': '2023-07-01', 
              'model_type': 'VAE',
              'action_specifications': 'ir_omega_equals',
              'simulator': simulator,
              'omega_pi': 0.5,
              'omega_psi': 0.5,
              'specifications_set': specifications_set,
              'use_penalty': False,
              'normalization_scheme': 'minmax',
              'df': df,
              'scaler': scaler,
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
        #print(f'Action: {action}, Observation: {obs}, Reward: {reward}')
        #print(f'Info:\n {info}')
        #print()
        if terminated or truncated:
            obs, info = env.reset()

