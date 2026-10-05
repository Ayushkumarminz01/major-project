"""Denoising Convolutional Autoencoder (DCAE) — matches the base-paper architecture:
Gaussian-noise input corruption, Conv -> latent -> Deconv, MSE reconstruction loss.
"""
import math

from tensorflow.keras import layers, models


def build_dcae(side, latent_dim):
    s2 = math.ceil(side / 2)
    crop = s2 * 2 - side

    inp = layers.Input((side, side, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    d = layers.Dense(s2 * s2 * 64, activation="relu")(z)
    d = layers.Reshape((s2, s2, 64))(d)
    d = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(d)
    d = layers.BatchNormalization()(d)
    if crop:
        d = layers.Cropping2D(((0, crop), (0, crop)))(d)
    d = layers.Conv2D(32, 3, padding="same", activation="relu")(d)
    out = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(d)

    autoencoder = models.Model(inp, out, name="DCAE")
    encoder = models.Model(inp, z, name="DCAE_encoder")
    return autoencoder, encoder
