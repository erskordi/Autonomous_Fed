import argparse
import numpy as np
import os
import pandas as pd
import pprint
import yaml
from datetime import timedelta

import ray
import ray.rllib.algorithms.ppo as ppo

from ray import tune, serve, train
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.utils.framework import try_import_tf

from config import Config
from env import AutonomousFed
from data_prep import gen_seq, series_to_supervised, plotting, DataPrep
from sim import TF_VAE_Model

tf1, tf, tfv = try_import_tf()

RAY_PICKLE_VERBOSE_DEBUG=1
os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"
os.environ['RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S'] = '3'

parser = argparse.ArgumentParser()

parser.add_argument("--n_cpus", type=int, default=4)
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="tf2",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=1000, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=0.0, help="Reward at which we stop training."
)
parser.add_argument(
    "--no-tune",
    type=bool,
    default=False,
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--omega-pi",
    type=float,
    default=0.5,
    help="Weight for the inflation loss.",
)
parser.add_argument(
    "--omega-psi",
    type=float,
    default=0.5,
    help="Weight for the output gap loss.",
)
parser.add_argument(
    "--action-specifications",
    type=str,
    default='ir_omega_equals',
    help="Action specifications for the environment; list of values: 1. ir_omega_equals, 2. ir_omega_not_equals, 3. ir_omega_pi_action, 4. ir_omega_all.",
)


def experiment(config):
    iterations = config.pop("train-iterations")

    algo = ppo.PPO(config=config)
    checkpoint = None
    train_results = {}

    # Train
    for i in range(iterations):
        train_results = algo.train()
        if i % 10 == 0 or i == iterations - 1:
            checkpoint = algo.save(train.get_context().get_trial_dir())
        train.report(train_results)
    algo.stop()

    # Manual Eval
    config["num_workers"] = 0
    eval_algo = ppo.PPO(config=config)
    eval_algo.restore(checkpoint)
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
        if args.action_specifications == 'ir_omega_equals':
            interest_list.append(action[0])
        elif args.action_specifications == 'ir_omega_not_equals':
            interest_list.append(action[0])
        elif args.action_specifications == 'ir_omega_pi_action':
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
    results = {**train_results, **eval_results}
    train.report(results)
    
    if args.action_specifications == 'ir_omega_equals':
        results_for_plotting = {
            "date": pd.date_range(start='2021-07-01', end='2050-12-31', freq='QS-JAN'),
            "Inflation": inflation_list,
            "Output_Gap": gdp_gap_list,
            "Interest_Rate": interest_list,
        }
    elif args.action_specifications == 'ir_omega_not_equals':
        results_for_plotting = {
            "date": pd.date_range(start='2021-07-01', end='2050-12-31', freq='QS-JAN'),
            "Inflation": inflation_list,
            "Output_Gap": gdp_gap_list,
            "Interest_Rate": interest_list,
        }
    elif args.action_specifications == 'ir_omega_pi_action':
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
    results_df.to_csv(f'/home/erskordi/projects/Autonomous_Fed/scripts/results/results_{args.action_specifications}.csv')
    """"""

if __name__ == "__main__":
    args = parser.parse_args()
    if ray.is_initialized():
        ray.shutdown()

    ray.init(num_cpus=args.n_cpus)

    specifications_set = input("Choose specifications set: {A, B, C}: ")
    # Initialize Ray Serve
    serve.start()

    # Load the models based on the specifications set
    encoder_path = os.path.join('../saved_models/',f'encoder_FedModel_{specifications_set}.keras')
    decoder_path = os.path.join('../saved_models/',f'decoder_FedModel_{specifications_set}.keras')
    path = [encoder_path, decoder_path]

    # Deploy the models
    #TF_VAE_Model.deploy(path)
    serve.run(target=TF_VAE_Model.bind(path),logging_config={"log_level": "ERROR"})

    df, scaler = DataPrep().read_data(specifications_set=specifications_set)

    env_config = {'start_date': '2021-07-01', 
                'end_date': '2050-12-31', 
                'model_type': 'VAE',
                'action_specifications': args.action_specifications,
                'omega_pi': args.omega_pi,
                'omega_psi': args.omega_psi,
                'specifications_set': specifications_set,
                'df': df,
                'scaler': scaler,
                'model_config': Config()}

    env_name = "AutonomousFed"
    
    register_env(env_name, lambda config: AutonomousFed(env_config))
    config = ppo.PPOConfig().framework(args.framework).\
        environment(env_name, disable_env_checking=True).\
            training(kl_coeff=0.0).resources(num_gpus=1)
    config = config.to_dict()
    config["train-iterations"] = args.stop_iters

    tune.Tuner(
        tune.with_resources(experiment, ppo.PPO.default_resource_request(config)),
        param_space=config,
    ).fit()

