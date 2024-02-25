import logging
import numpy as np
import os
import pandas as pd
import requests
import tensorflow as tf

import ray
import ray.rllib.algorithms.ppo as ppo

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.framework import try_import_tf

from config import Config
from env import AutonomousFed
from data_prep import gen_seq, series_to_supervised, plotting, DataPrep
from sim import TF_VAE_Model

tf1, tf, tfv = try_import_tf()

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
        self.original_df = self.df.copy()

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

        low = 0
        high = 30

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
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.df.shape[1]-1,), dtype=np.float32)

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
        obs = np.array(self.df.iloc[0, 1:], dtype=np.float32)

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

        # Step 1: Action to list
        # If action is a dict, then it is a dict of the form {'interest_rate': 0.5, 'omega_pi': 0.5, 'omega_psi': 0.5}
        # Otherwise, it's only about the interest rate
        if self.config['action_specifications'] == 'ir_omega_equals' or \
           self.config['action_specifications'] == 'ir_omega_not_equals':
            action_ir = action.tolist()
        else:
            action_ir = action['interest_rate'].tolist()
        

        AutonomousFed.cntr += 1
       
        # Step 3: transform HTTP request -> tensorflow input
        obs = np.array(self.df.iloc[AutonomousFed.cntr, 1:], dtype=np.float32)
            

        # Step 5: Reward, terminated, truncated, info
        if self.config['specifications_set'] == 'A':
            #df_copy = pd.DataFrame(self.scaler.inverse_transform(self.df), columns=self.columns)
            #obs = np.array(df_copy.iloc[len(df_copy)-1, 1:], dtype=np.float32)
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
        
        - Desired inflation = 0.02
        """
        # 
        desired_inflation = 0.17#1.02
        if self.specifications_set == 'A':
            output_gap = obs[1]
            inflation_diff = abs(obs[0] - desired_inflation)
            reward = - (a_pi * (inflation_diff ** 2) + a_psi * (output_gap ** 2))# + obs[2]
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
    if ray.is_initialized():
        ray.shutdown()

    ray.init(num_cpus=4)

    df, scaler = DataPrep().read_data(specifications_set=specifications_set)

    env_config = {'start_date': '2021-07-01', 
                'end_date': '2050-12-31', 
                'model_type': 'VAE',
                'action_specifications': 'ir_omega_all_actions',
                'omega_pi': 0.5,
                'omega_psi': 0.5,
                'specifications_set': specifications_set,
                'df': df,
                'scaler': scaler,
                'model_config': Config()}

    env_name = "AutonomousFed"
    
    register_env(env_name, lambda config: AutonomousFed(env_config))
    config = ppo.PPOConfig().framework('tf2').\
        environment(env_name, disable_env_checking=True).\
            training(kl_coeff=0.0).resources(num_gpus=1)
    config = config.to_dict()
    

    # Manual Eval
    config["num_workers"] = 0
    eval_algo = ppo.PPO(config=config)
    eval_algo.restore('/home/erskordi/ray_results/experiment_2024-02-22_20-23-43/experiment_AutonomousFed_2f1fa_00000_0_2024-02-22_20-23-43/policies/default_policy/policy_state.pkl')
    env = eval_algo.workers.local_worker().env
    inflation_list = []
    gdp_gap_list = []
    interest_list = []
    omega_pi_list = []
    omega_psi_list = []

    obs, info = env.reset()
    done = False
    eval_results = {"eval_reward": 0, "eval_eps_length": 0}
    while not done:
        action = eval_algo.compute_single_action(obs)
        if env_config['action_specifications'] == 'ir_omega_equals':
            interest_list.append(action[0])
        elif env_config['action_specifications'] == 'ir_omega_not_equals':
            interest_list.append(action[0])
        elif env_config['action_specifications'] == 'ir_omega_pi_action':
            interest_list.append(action["interest_rate"][0])
            omega_psi_list.append(action["omega_psi"][0])
        else:
            interest_list.append(action["interest_rate"][0])
            omega_pi_list.append(action["omega_pi"][0])
            omega_psi_list.append(action["omega_psi"][0])
        next_obs, reward, done, truncated, info = env.step(action)
        inflation_list.append(next_obs[0])
        gdp_gap_list.append(next_obs[1])
        #print(f"Next obs: {next_obs}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")
        eval_results["eval_reward"] += reward
        eval_results["eval_eps_length"] += 1
    
    if env_config['action_specifications'] == 'ir_omega_equals':
        results_for_plotting = {
            "date": pd.date_range(start='2021-07-01', end='2050-12-31', freq='QS-JAN'),
            "Inflation": inflation_list,
            "Output_Gap": gdp_gap_list,
            "Interest_Rate": interest_list,
        }
    elif env_config['action_specifications'] == 'ir_omega_not_equals':
        results_for_plotting = {
            "date": pd.date_range(start='2021-07-01', end='2050-12-31', freq='QS-JAN'),
            "Inflation": inflation_list,
            "Output_Gap": gdp_gap_list,
            "Interest_Rate": interest_list,
        }
    elif env_config['action_specifications'] == 'ir_omega_pi_action':
        results_for_plotting = {
            "date": pd.date_range(start='2021-07-01', end='2050-12-31', freq='QS-JAN'),
            "Inflation": inflation_list,
            "Output_Gap": gdp_gap_list,
            "Interest_Rate": interest_list,
            "Omega_Psi": omega_psi_list,
        }
    else:
        results_for_plotting = {
            "date": pd.date_range(start='2021-07-01', end='2050-12-31', freq='QS-JAN'),
            "Inflation": inflation_list,
            "Output_Gap": gdp_gap_list,
            "Interest_Rate": interest_list,
            "Omega_Psi": omega_psi_list,
            "Omega_Pi": omega_pi_list,
        }
    
    # Create a dataframe from the results
    results_df = pd.DataFrame(results_for_plotting)
    
    # Save the results to a csv file
    spec = env_config['action_specifications']
    results_df.to_csv(f'/home/erskordi/projects/Autonomous_Fed/scripts/results/results_trueData_{spec}.csv')