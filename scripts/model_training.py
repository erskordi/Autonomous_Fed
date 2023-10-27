import pandas as pd
import tensorflow as tf

import ray
from ray import tune

from config import Config
from data_prep import DataPrep

import matplotlib.pyplot as plt

class VAE:
    def __init__(self, input_dim):
        self.config = Config()
        self.input_dim = self.config.input_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.model = self.build_vae()
        self.optimizer = tf.keras.optimizers.Adam()

    def build_encoder(self):
        inputs = tf.keras.layers.Input(shape=(self.input_dim,), name='encoder_input')
        x = tf.keras.layers.Dense(self.config.intermediate_dim, activation='relu')(inputs)
        z_mean = tf.keras.layers.Dense(self.config.latent_dim, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(self.config.latent_dim, name='z_log_var')(x)

        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = tf.keras.layers.Lambda(sampling, output_shape=(self.config.latent_dim,), name='z')([z_mean, z_log_var])
        return tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def build_decoder(self):
        latent_inputs = tf.keras.layers.Input(shape=(self.config.latent_dim,), name='z_sampling')
        x = tf.keras.layers.Dense(self.config.intermediate_dim, activation='relu')(latent_inputs)
        outputs = tf.keras.layers.Dense(self.input_dim)(x)
        return tf.keras.models.Model(latent_inputs, outputs, name='decoder')

    def build_vae(self):
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)
        return tf.keras.models.Model(inputs, outputs, name='vae')

    def compute_loss(self, data, reconstruction, z_mean, z_log_var):
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(data, reconstruction))
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        return reconstruction_loss + kl_loss

    @tf.function
    def train_step(self, data, target):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            loss = self.compute_loss(target, reconstruction, z_mean, z_log_var)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def train(self, data, target, epochs=50, batch_size=32):
        for epoch in range(epochs):
            print("\nEpoch %d" % (epoch,))
            losses = []
            for idx in range(0, len(data), batch_size):
                data_batch = data[idx:idx+batch_size]
                target_batch = target[idx:idx+batch_size]
                loss = self.train_step(data_batch, target_batch)
                losses.append(float(loss))
                #print("Loss:", float(loss))
            print("Mean loss:", sum(losses) / len(losses))
        
    
    def save_models(self, path):
        """Save the encoder, decoder, and vae models."""
        self.encoder.save(f"{path}/encoder")
        self.decoder.save(f"{path}/decoder")
        self.model.save(f"{path}/vae")

    @classmethod
    def load_encoder(cls, path):
        """Load the encoder model."""
        return tf.keras.models.load_model(f"{path}/encoder")

    @classmethod
    def load_decoder(cls, path):
        """Load the decoder model."""
        return tf.keras.models.load_model(f"{path}/decoder")

    @classmethod
    def load_vae(cls, path):
        """Load the VAE model."""
        return tf.keras.models.load_model(f"{path}/vae")


if __name__ == "__main__":
    config = Config()
    data_prep = DataPrep()
    df = data_prep.read_data()
    vae = VAE(input_dim=df.shape[1])
    data = df[:-1].values
    target = df[1:].values
    vae.train(data, target, epochs=config.epochs, batch_size=config.batch_size)

    # Save the trained models
    path = "/Users/erotokritosskordilis/git-repos/Autonomous_Fed/saved_models"
    vae.save_models(path)

