#We will also try image augmentation in this
import os
import zipfile
import random
import tensorflow as tf
import shutil
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd



CAT_SOURCE_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/PetImages/Cat/')
TRAINING_CATS_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/cats-v-dogs/training/dogs/')
TESTING_CATS_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/cats-v-dogs/testing/dogs/')
DOG_SOURCE_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/PetImages/Dog/')
TRAINING_DOGS_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/cats-v-dogs/training/dogs/')
TESTING_DOGS_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/cats-v-dogs/testing/dogs/')


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_data = random.sample(files, len(files))
    training_data = shuffled_data[0:training_length]
    testing_data = shuffled_data[-testing_length:]

    for fn in training_data:
        datasource = SOURCE + fn
        destination = os.path.join(TRAINING + fn)
        copyfile(datasource, destination)

    for fn1 in testing_data:
        datasourece1 = SOURCE + fn1
        destination1 = os.path.join(TESTING + fn1)
        copyfile(datasourece1, destination1)


split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print(len(os.listdir(TRAINING_CATS_DIR)))
print(len(os.listdir(TRAINING_DOGS_DIR)))
print(len(os.listdir(TESTING_CATS_DIR)))
print(len(os.listdir(TESTING_DOGS_DIR)))


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = tf.keras.models.Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR = "/Users/seanjudelyons/Downloads/Find/cats-v-dogs/training/"

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE
# TRAIN GENERATOR.
train_generator = train_datagen.flow_from_directory(TRAINING_DIR, target_size=(150, 150), batch_size=10, class_mode='binary')


VALIDATION_DIR = "/Users/seanjudelyons/Downloads/Find/cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1/255.)


# NOTE: YOU MUST USE A BACTH SIZE OF 10 (batch_size=10) FOR THE
# VALIDATION GENERATOR.
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, target_size=(150, 150), batch_size=10,class_mode='binary')



# Expected Output:
# Found 2700 images belonging to 2 classes.
# Found 300 images belonging to 2 classes.


history = model.fit_generator(train_generator,
                              epochs=2,
                              verbose=1,
                              validation_data=validation_generator)



import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)











