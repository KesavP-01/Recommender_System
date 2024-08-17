import tensorflow as tf
from keras import layers
import keras as ks

def model_build(input):
    model = ks.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model