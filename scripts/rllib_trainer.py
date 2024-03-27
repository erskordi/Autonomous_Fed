import argparse
import numpy as np
import os
import pandas as pd
import pprint
import torch
import yaml
from datetime import timedelta

import ray
from ray import tune, serve, air
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from config import Config, print_colored_text
from env import AutonomousFed
from data_prep import gen_seq, series_to_supervised, plotting, DataPrep
from sim import TF_VAE_Model

from policy_models import LinearPolicy, MLPModel, LSTMModel

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

RAY_PICKLE_VERBOSE_DEBUG=1
os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"
os.environ['RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S'] = '3'

torch.cuda.empty_cache()

cpu_count = os.cpu_count()

parser = argparse.ArgumentParser()

parser.add_argument("--n-cpus", type=int, default=cpu_count, help="Number of CPUs to use.")
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf1", "torch"],
    default="torch",
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
    "--simulator",
    type=str,
    default="VAE",
    help="Simulator type for the environment; list of values: 1. VAE, 2. Random Forest Regressor.",
)
parser.add_argument(
    "--omega-psi",
    type=float,
    default=0.5,
    help="Weight for the output gap loss.",
)
parser.add_argument(
    "--normalization-scheme",
    type=str,
    default='minmax',
    help="Normalization scheme for the data; list of values: 1. minmax, 2. sigmoid.",
)
parser.add_argument(
    "--action-specifications",
    type=str,
    default='ir_omega_equals',
    help="Action specifications for the environment; list of values: 1. ir_omega_equals, 2. ir_omega_not_equals, 3. ir_omega_pi_action, 4. ir_omega_all.",
)
parser.add_argument(
    "--use-penalty",
    type=bool,
    default=False,
    help="Use penalty for regularizing reward function.",
)
parser.add_argument(
    "--n-gpus", type=float, default=1.0, help="Number of GPUs to use."
)

args = parser.parse_args()


if ray.is_initialized():
    ray.shutdown(shutdown_at_exit=True)

ray.init(
    #address = cluster.address,
    num_cpus=args.n_cpus,
    num_gpus=args.n_gpus,
    resources = {'special_hardware': 1}
)

ModelCatalog.register_custom_model("linear_model", LinearPolicy)
ModelCatalog.register_custom_model("mlp_model", MLPModel)
ModelCatalog.register_custom_model("lstm_model", LSTMModel)

specifications_set = input("Choose specifications set: {A, B, C}: ").upper()

if args.simulator == 'VAE':
    # Initialize Ray Serve
    serve.start()

    # Load the models based on the specifications set
    encoder_path = os.path.join('../saved_models/',f'encoder_FedModel_{specifications_set}.keras')
    decoder_path = os.path.join('../saved_models/',f'decoder_FedModel_{specifications_set}.keras')
    path = [encoder_path, decoder_path]

    # Deploy the models
    #TF_VAE_Model.deploy(path)
    serve.run(target=TF_VAE_Model.bind(path),logging_config={"log_level": "ERROR"})

df, df_interest_rate, scaler = DataPrep().read_data(specifications_set=specifications_set)

env_config = {'start_date': '1954-07-01', 
              'end_date': '2023-07-01', 
              'model_type': args.simulator,
              'action_specifications': args.action_specifications,
              'simulator': args.simulator, 
              'omega_pi': args.omega_pi,
              'omega_psi': args.omega_psi,
              'specifications_set': specifications_set,
              'use_penalty': args.use_penalty,
              'normalization_scheme': args.normalization_scheme,
              'df': [df, df_interest_rate],
              'scaler': scaler,
              'model_config': Config()}

env_name = "AutonomousFed"
register_env(env_name, lambda config: AutonomousFed(env_config))

config = PPOConfig()
config.training(
    kl_coeff=0.2,
    lr=0.01,
    grad_clip=0.01,
    sgd_minibatch_size=16,
    #model={
    #    "custom_model": "linear_model",
    #    "custom_model_config": {},
    #},
)
config = config.framework(args.framework)
config = config.environment(env_name, disable_env_checking=True)
config = config.resources(num_gpus=args.n_gpus)
config = config.rollouts(num_rollout_workers=int(.75*(args.n_cpus)))

stop = {
        "training_iteration": args.stop_iters,
        #"timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10)

if args.no_tune:
    # manual training with train loop using PPO and fixed learning rate
    if args.run != "PPO":
        raise ValueError("Only support --run PPO with --no-tune.")
    print("Running manual train loop without Ray Tune.")
    # use fixed learning rate instead of grid search (needs tune)
    config.lr = 1e-3
    algo = config.build()
    # run manual training loop and print results after each iteration
    for itr in range(args.stop_iters):
        result = algo.train()
        if itr % 10 == 0:
            save_result = algo.save()
            print(save_result)
            path_to_checkpoint = save_result.checkpoint.path
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )
    algo.stop()
else:
    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    tuner = tune.Tuner(
        args.run,
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, checkpoint_config=checkpoint_config),
        tune_config=tune.TuneConfig(reuse_actors=True),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="episode_reward_mean")
    best_checkpoint = best_result.checkpoint 
    print_colored_text(f"Best checkpoint path: {best_checkpoint}", color='green')