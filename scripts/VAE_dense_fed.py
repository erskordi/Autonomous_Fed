from gc import callbacks
from tabnanny import verbose
import os
import numpy as np
import pickle
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

from itertools import chain

from config import Config
from data_prep import gen_seq, series_to_supervised, plotting, DataPrep

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(y_true, y_predict), axis=-1)
                )
        return reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -.5 * (1 + encoder_log_variance - tf.square(encoder_mu) - tf.exp(encoder_log_variance))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss

def sampling(mu_log_variance):
    x_mean, x_logvar = mu_log_variance
    batch = tf.shape(x_mean)[0]
    dim = tf.shape(x_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

    return x_mean + tf.exp(0.5 * x_logvar) * epsilon


if __name__ == "__main__":

    checkpoint_path = 'saved_models/training/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,verbose=1)

    config = Config()

    specifications_set = int(input("Choose specifications set (0, 1, 2): "))
    df = DataPrep().read_data(specifications_set=specifications_set)

    # Build encoder
    latent_dim = config.latent_dim

    encoder_inputs = tf.keras.Input(shape=(df.shape[1],))
    x = encoder_inputs

    for i in range(len(config.dense_neurons)):
        if i == 0:
            x = tf.keras.layers.Dense(config.dense_neurons[i], activation='relu', name="dense_layer_" + str(i))(x)
        elif (i > 0) & (i <= len(config.dense_neurons)):
            x = tf.keras.layers.Dense(config.dense_neurons[i], activation='relu', name="dense_layer_" + str(i))(x)
        else:
            x = tf.keras.layers.Dense(config.dense_neurons[i], activation='relu', name="dense_layer_final")(x)

    x_mean = tf.keras.layers.Dense(config.latent_dim, name="x_mean")(x)#, activation='sigmoid'
    x_logvar = tf.keras.layers.Dense(config.latent_dim, name="x_logvar")(x)
    
    mean_logvar_model = tf.keras.models.Model(x, (x_mean, x_logvar), name="mean_logvar_model")

    encoder_output = tf.keras.layers.Lambda(sampling, name="encoder_output")([x_mean, x_logvar])

    encoder = tf.keras.models.Model(encoder_inputs, encoder_output, name="encoder")
    encoder.summary()
    
    # Build decoder
    latent_inputs = tf.keras.Input(shape=(config.latent_dim,))
    x = latent_inputs

    for i in range(len(config.dense_neurons)-1,-1,-1):
        if i == len(config.dense_neurons) - 1:
            x = tf.keras.layers.Dense(config.dense_neurons[i], activation='relu', name="latent_layer")(x)
        else:
            x = tf.keras.layers.Dense(config.dense_neurons[i], activation='relu', name="dense_layer_" + str(i))(x)
    decoder_outputs = tf.keras.layers.Dense(df.shape[1]-1, activation="sigmoid", name="decoder_output")(x)

    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    
    # Build VAE
    vae_input = tf.keras.layers.Input(shape=(df.shape[1],), name="VAE_input")
    vae_encoder_output = encoder(vae_input)
    vae_decoder_output = decoder(vae_encoder_output)
    vae = tf.keras.models.Model(vae_input, vae_decoder_output, name="VAE")
    vae.summary()

    # Stop training when a monitored quantity has stopped improving for 3 consecutive epochs.
    callbacks = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, start_from_epoch=100)
    
    # Train VAE
    x_train = df.values
    vae.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005), loss=loss_func(x_mean, x_logvar))
    vae.fit(x_train[:-1,:], x_train[1:,1:],\
        epochs=config.epochs,
        batch_size=config.batch_size//2, 
        callbacks=[cp_callback, callbacks]
    )
    

    # Save models (decoder and/or encoder)

    encoder.save(f'../saved_models/encoder_FedModel_{specifications_set}.keras')
    decoder.save(f'../saved_models/decoder_FedModel_{specifications_set}.keras')

    vae.save(f'../saved_models/vae_FedModel_{specifications_set}.keras')
    