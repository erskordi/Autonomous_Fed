import os

import ray
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

class TfLinearPolicy(TFModelV2):
    """
    This custom model is a simple linear model without any hidden layers or non-linear activation functions.
    It represents a simple linear policy that maps observations to actions akin to linear regression.

    In this model, we only have a single layer.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(TfLinearPolicy, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        layer_out = tf.keras.layers.Dense(
            num_outputs, 
            activation=None, 
            name='linear_layer', 
            kernel_initializer=normc_initializer(0.01))(self.inputs)
        value_out = tf.keras.layers.Dense(
            1, 
            activation=None, 
            name='value_layer', 
            kernel_initializer=normc_initializer(0.01))(layer_out)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        #print(self.base_model.summary())

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs_flat"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
    
    def metrics(self):
        return {"foo": tf.constant(42.0)}

class TorchLinearPolicy(TorchModelV2, torch.nn.Module):
    """
    This custom model is a simple linear model without any hidden layers or non-linear activation functions.
    It represents a simple linear policy that maps observations to actions akin to linear regression.

    In this model, we only have a single layer.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)

        self.inputs = torch.nn.Linear(obs_space.shape[0], num_outputs)
        self.value_out = torch.nn.Linear(num_outputs, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.inputs(input_dict["obs_flat"])
        self._value_out = self.value_out(model_out)
        return model_out, state

    def value_function(self):
        return torch.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": torch.tensor(42.0)}