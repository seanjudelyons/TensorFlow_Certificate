import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

import os

train_happy_dir = os.path.join('/Users/seanjudelyons/Downloads/happy-or-sad/happy/')  # the zip file had the folders called horses and humans

train_sad_dir = os.path.join('/Users/seanjudelyons/Downloads/happy-or-sad/sad/')

train_happy_names = os.listdir(train_happy_dir)
print(train_happy_names[:10])

train_sad_names = os.listdir(train_sad_dir)
print(train_sad_names[:10])

print('total training happy images:', len(os.listdir(train_happy_dir)))
print('total training sad images:', len(os.listdir(train_happy_dir)))


# GRADED FUNCTION: train_happy_sad_model
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.999):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

def train_happy_sad_model():
    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.summary()

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory('/Users/seanjudelyons/Downloads/happy-or-sad/', target_size=(150, 150), batch_size=20,
                                                        class_mode='binary')

    history = model.fit(train_generator, steps_per_epoch=4, epochs=20, verbose=1, callbacks=[callbacks])

    from keras.preprocessing import image
    import numpy as np

    path = os.path.join('/Users/seanjudelyons/Downloads/happyorsadtest.png')
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)

    print(classes[0])
    if classes[0] > 0.5:
        print("The picture is Sad")
    else:
        print("The picture is Happy")

    return history.history['acc'][-1]


train_happy_sad_model()

