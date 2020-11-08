
#this trains till 90 percent accuracy on the mnsit fashion and then calls back

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import fashion_mnist

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

(training_images, training_labels), (test_images,test_labels)=fashion_mnist.load_data()

callbacks = myCallback()

import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

training_images= training_images.reshape(60000, 28, 28, 1) #reshaping array
training_images= training_images /255.0
test_images= test_images.reshape(10000, 28, 28, 1)
test_images= test_images / 255.0
print(training_images.shape)
print(test_images.shape)


model=tf.keras.models.Sequential([
tf.keras.layers.Conv2D(64, (3,3),activation='relu', input_shape=(28, 28, 1)),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3),activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])


test_loss= model.evaluate(test_images, test_labels)


print(test_labels[:100])




