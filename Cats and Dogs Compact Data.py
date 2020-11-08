
#works better with smaller data set
import os

CAT_SOURCE_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/PetImages/Cat/')
TRAINING_CATS_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/cats-v-dogs/training/dogs/')
TESTING_CATS_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/cats-v-dogs/testing/dogs/')
DOG_SOURCE_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/PetImages/Dog/')
TRAINING_DOGS_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/cats-v-dogs/training/dogs/')
TESTING_DOGS_DIR = os.path.join('/Users/seanjudelyons/Downloads/Find/cats-v-dogs/testing/dogs/')

train_dog_names = TRAINING_DOGS_DIR
train_cat_names = TRAINING_CATS_DIR

validation_dog_names = TESTING_DOGS_DIR
validation_cat_names = TESTING_CATS_DIR


import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()


from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/Users/seanjudelyons/Downloads/Find/cats-v-dogs/training/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/Users/seanjudelyons/Downloads/Find/cats-v-dogs/testing/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')


history = model.fit(
      train_generator,
      steps_per_epoch=22,
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=7)

from keras.preprocessing import image
import numpy as np

path = os.path.join('/Users/seanjudelyons/Downloads/CatTest.jpg')
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])

#predicting image
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0]>0.5:
    print("The picture is similar to a cat")
else:
    print("The picture is similar to a dog")


