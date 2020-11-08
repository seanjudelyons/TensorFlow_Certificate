#this trains on the MNSit with above 99 percent accuracy

import tensorflow as tf
from os import path, getcwd, chdir


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location

mnist = tf.keras.datasets.mnist


# GRADED FUNCTION: train_mnist

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


def train_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

    return history.epoch, history.history['accuracy'][-1]

train_mnist()