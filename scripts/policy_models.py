import os
import gymnasium as gym
import numpy as np

import ray
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer

from ray.rllib.utils.annotations import OldAPIStack, override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from tensorflow.keras.utils import plot_model

from torchviz import make_dot

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

@OldAPIStack
class TfLinearPolicy(TFModelV2):
    """
    This custom model is a simple linear model without any hidden layers or non-linear activation functions.
    It represents a simple linear policy that maps observations to actions akin to linear regression.

    In this model, we only have a single layer.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(TfLinearPolicy, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        
        hiddens = list([])
        activation = None
        activation = get_activation_fn(activation)
        no_final_linear = False
        vf_share_layers = True
        free_log_std = False

        # We are using obs_flat, so take the flattened shape as input.
        inputs = tf.keras.layers.Input(
            shape=(int(np.product(obs_space.shape)),), name="observations"
        )
        # Last hidden layer output (before logits outputs).
        last_layer = inputs
        # The action distribution outputs.
        logits_out = None
        i = 1

        if num_outputs:
            logits_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=None,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="uniform"),
                #kernel_initializer=normc_initializer(0.01),
            )(last_layer)
        # Adjust num_outputs to be the number of nodes in the last layer.
        else:
            self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
                -1
            ]

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.01, mode="fan_in", distribution="uniform"),
            #kernel_initializer=normc_initializer(0.01),
        )(last_layer)

        self.base_model = tf.keras.Model(
            inputs, [(logits_out if logits_out is not None else last_layer), value_out]
        )
        #print(self.base_model.summary())
        #plot_model(self.base_model, to_file="base_model.png", show_shapes=True)

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        model_out, self._value_out = self.base_model(input_dict["obs_flat"])
        return model_out, state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])
    
    def metrics(self):
        return {"foo": tf.constant(42.0)}

@OldAPIStack
class TorchLinearPolicy(TorchModelV2, torch.nn.Module):
    """
    This custom model is a simple linear model without any hidden layers or non-linear activation functions.
    It represents a simple linear policy that maps observations to actions akin to linear regression.

    In this model, we only have a single layer.
    """
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        hiddens = list([])
        activation = None
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = False
        self.vf_share_layers = True
        self.free_log_std = False

        
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None
        layers.append(
            SlimFC(
                in_size=prev_layer_size,
                out_size=num_outputs,
                initializer=normc_initializer(1.0),
                activation_fn=activation,
            )
        )
        if no_final_linear and num_outputs:
            self._logits = SlimFC(
                in_size=int(np.product(obs_space.shape)),
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None,
            )
        else:
            self.num_outputs = ([int(np.product(obs_space.shape))])[-1]
            

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            out = self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            out = self._value_branch(self._features).squeeze(1)
        return out

if __name__ == "__main__":
    # Test the custom model
    obs_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(2,))
    action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
    model_config = {}
    name = "test_model"
    num_outputs = 2

    input_dict = {"obs_flat": np.array([[1.0, 2.0]]), "obs": np.array([[1.0, 2.0]])}

    # Test the Torch model
    torch_model = TorchLinearPolicy(obs_space, action_space, num_outputs, model_config, name)
    print(torch_model)