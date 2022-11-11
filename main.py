"""
ECE472 Midterm
ODE Neural Network Spiral Generation
Bonny (Yue) Wang
Bob (Sangjoon) Lee
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers
import matplotlib.pyplot as plt
from tqdm import trange
from tfdiffeq import odeint
import os

# Parameters
EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 24
N_SAMPLES = 150
EXTRAPOLATION_RATIO = 0.65

script_path = os.path.dirname(os.path.realpath(__file__))


def generate_Data(n_samples, extrapolation_ratio):
    theta = np.linspace(2, 6.1 * np.pi, n_samples)

    # Scale factor
    a = 1.5

    r = theta**a

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Normalize x and y
    x = x / x.max()
    y = y / y.max()

    # Seperate Train samples
    n_trainSample = int(n_samples * extrapolation_ratio)

    # Add Noise to data
    noise = np.random.normal(0, 0.03, size=(n_samples - n_trainSample))
    noise_Theta = np.random.normal(0, 0.2, size=(n_samples))

    theta_Noisy = theta + noise_Theta
    theta_Noisy.sort()
    x_Iregular = r * np.cos(theta_Noisy)
    y_Iregular = r * np.sin(theta_Noisy)
    x_Iregular = x_Iregular / x_Iregular.max()
    y_Iregular = y_Iregular / y_Iregular.max()

    # Reverse for the direction and add noise to x y
    x_Noisy = np.flip(x_Iregular[n_trainSample:] + noise)
    y_Noisy = np.flip(y_Iregular[n_trainSample:] + noise)
    plt.plot(x_Noisy, y_Noisy, "go", x, y, "-")

    # corresponding t from the end
    t = np.flip(theta_Noisy.max() + (-1 * theta_Noisy[n_trainSample:]))
    return x, y, x_Noisy, y_Noisy, t


# The model for approximatint the f'
class ODE_Function(tf.keras.Model):

    # Activation function selection based on
    # https://github.com/titu1994/tfdiffeq/blob/master/examples/circular_ode_demo.py
    # This works the best for the convergence

    def __init__(self, **kwargs):
        super(ODE_Function, self).__init__(**kwargs)

        self.layer0 = layers.Dense(
            32,
            activation="tanh",
            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
            kernel_regularizer=tf.keras.regularizers.L2(0.01),
        )
        self.layer1 = layers.Dense(
            64,
            activation="tanh",
            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
            kernel_regularizer=tf.keras.regularizers.L2(0.01),
        )
        self.layer_Output = layers.Dense(
            2,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
            kernel_regularizer=tf.keras.regularizers.L2(0.01),
        )

    def call(self, t, input):
        h0 = self.layer0(input)
        h1 = self.layer1(h0)
        h_Output = self.layer_Output(h1)

        return h_Output


def get_Batch(t, labels):
    # Prepare the sequential data batches
    # Include the intial start position, following 16 points location as the label, and time stamps for those following points
    # For simplicity and assume there are noise in times, we assume all time step followed the pattern in the first 16 time stamp recorded

    choices = np.random.choice(
        np.arange(t.shape[0] - BATCH_SIZE, dtype=np.int64), BATCH_SIZE, replace=False
    )

    initial_State = tf.convert_to_tensor(labels[choices], dtype="float32")
    batch_T = tf.cast(t[:BATCH_SIZE], dtype="float32")
    batch_Y = tf.cast(
        tf.stack([labels[choices + i] for i in range(BATCH_SIZE)], axis=0),
        dtype="float32",
    )
    return batch_T, batch_Y, initial_State


def trainAndShow():
    x, y, x_Noisy, y_Noisy, t = generate_Data(N_SAMPLES, EXTRAPOLATION_RATIO)

    # Reshape the data for join
    x_Noisy = x_Noisy[:, np.newaxis]
    y_Noisy = y_Noisy[:, np.newaxis]

    lables = np.append(x_Noisy, y_Noisy, axis=1)

    ode_Fn = ODE_Function()
    optimizer = tf.keras.optimizers.Adam(LR)

    bar = trange(EPOCHS)

    for i in bar:
        with tf.GradientTape() as tape:
            batch_T, batch_Labels, initial_State = get_Batch(t, lables)

            predictions = odeint(ode_Fn, initial_State, batch_T)

            loss = tf.reduce_mean((predictions - batch_Labels) ** 2)

        grads = tape.gradient(loss, ode_Fn.trainable_variables)
        optimizer.apply_gradients(zip(grads, ode_Fn.trainable_variables))
        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    # Generate the spiral to show the ODE NN trajectory
    initial_State = tf.constant([[lables[0]]], dtype=tf.float32)
    results = odeint(ode_Fn, initial_State, tf.cast(np.linspace(0, 16, 300), "float"))
    temp = np.transpose(results).squeeze()

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Reconstructed Spiral Series by ODENN")
    plt.plot(temp[0], temp[1], "r-", label="ODENN")
    plt.plot(x, y, "g-", label="Truth")
    plt.plot(x_Noisy, y_Noisy, "bo", label="Observation")
    # plt.savefig(f"{script_path}/plot.pdf")
    plt.legend()
    plt.savefig(f"{script_path}/plot.pdf")
    plt.show()


if __name__ == "__main__":
    trainAndShow()
