import argparse
import numpy as np
import os
import pandas as pd
import pprint
import yaml
from datetime import timedelta

import ray
from ray import tune, serve, air
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
    "--stop-iters", type=int, default=50, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=0.1, help="Reward at which we stop training."
)
parser.add_argument(
    "--no-tune",
    type=bool,
    default=False,
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--omega_pi",
    type=float,
    default=0.5,
    help="Weight for the inflation loss.",
)
parser.add_argument(
    "--omega_psi",
    type=float,
    default=0.5,
    help="Weight for the output gap loss.",
)
parser.add_argument(
    "--action_specifications",
    type=str,
    default='ir_omega_equals',
    help="Action specifications for the environment; list of values: 1. ir_omega_equals, 2. ir_omega_not_equals, 3. ir_omega_pi_action, 4. ir_omega_all.",
)

args = parser.parse_args()

if ray.is_initialized():
    ray.shutdown()

ray.init(num_cpus=args.n_cpus)

specifications_set = input("Choose specifications set: {A, B, C}: ")
# Initialize Ray Serve
serve.start()

# Load the models based on the specifications set
encoder_path = os.path.join('/Users/erotokritosskordilis/git-repos/Autonomous_Fed/saved_models/',f'encoder_FedModel_{specifications_set}.keras')
decoder_path = os.path.join('/Users/erotokritosskordilis/git-repos/Autonomous_Fed/saved_models/',f'decoder_FedModel_{specifications_set}.keras')
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

config = (
    PPOConfig().framework(args.framework).\
    environment(env_name, disable_env_checking=True).\
    training(kl_coeff=0.0)
)

stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

if args.no_tune:
    # manual training with train loop using PPO and fixed learning rate
    if args.run != "PPO":
        raise ValueError("Only support --run PPO with --no-tune.")
    print("Running manual train loop without Ray Tune.")
    # use fixed learning rate instead of grid search (needs tune)
    config.lr = 1e-3
    algo = config.build()
    # run manual training loop and print results after each iteration
    for _ in range(args.stop_iters):
        result = algo.train()
        # stop training of the target train steps or reward are reached
        if (
            result["timesteps_total"] >= args.stop_timesteps
            or result["episode_reward_mean"] >= args.stop_reward
        ):
            break
    algo.stop()
else:
    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    tuner = tune.tuner.Tuner(
        args.run,
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop),
    )
    results = tuner.fit()