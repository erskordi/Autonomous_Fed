import os

import ray
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

class LinearPolicy(TFModelV2):
    """
    This custom model is a simple linear model without any hidden layers or non-linear activation functions.
    It represents a simple linear policy that maps observations to actions akin to linear regression.

    In this model, we only have a single layer.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(LinearPolicy, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        
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

class MLPModel(TFModelV2):
    """
    This custom model is a simple MLP model with a single hidden layer and a single dense layer.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(MLPModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        layer_out = self.inputs
        for layer in config['dense_neurons']:
            layer_out = tf.keras.layers.Dense(
                layer, 
                activation='relu', 
                name='dense_layer')(layer_out)
        model_out = tf.keras.layers.Dense(
            num_outputs, 
            activation=None, 
            name='model_out')(layer_out)
        value_out = tf.keras.layers.Dense(
            1, 
            activation=None, 
            name='value_layer', 
            kernel_initializer=normc_initializer(0.01))(layer_out)
        self.base_model = tf.keras.Model(self.inputs, model_out)
        print(self.base_model.summary())

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
    
    def metrics(self):
        return {"foo": tf.constant(42.0)}

class LSTMModel(TFModelV2):
    """
    This custom model is a simple LSTM model with a single LSTM layer and a single dense layer.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(LSTMModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        layer_out = tf.keras.layers.LSTM(
            config['lstm_cell_size'], 
            activation='tanh', 
            return_sequences=False, 
            name='lstm_layer')(self.inputs)
        for layer in model_config['dense_neurons']:
            layer_out = tf.keras.layers.Dense(
                layer, 
                activation='relu', 
                name='dense_layer')(layer_out)
        model_out = tf.keras.layers.Dense(
            num_outputs, 
            activation=None, 
            name='model_out')(layer_out)
        value_out = tf.keras.layers.Dense(
            1, 
            activation=None, 
            name='value_layer', 
            kernel_initializer=normc_initializer(0.01))(layer_out)
        self.base_model = tf.keras.Model(self.inputs, model_out)
        print(self.base_model.summary())

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        print(model_out)
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
    
    def metrics(self):
        return {"foo": tf.constant(42.0)}


if __name__ == "__main__":
    ray.init()
    model = LinearPolicy()
    
    
    ray.shutdown()