import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import shutil
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = os.path.join('/Users/seanjudelyons/Downloads/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')


pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.97):
      print("\nReached 97.0% accuracy so cancelling training!")
      self.model.stop_training = True



from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_output)# Flattening mixed 7 (Flatten)

x = layers.Dense(1024, activation='relu')(x)#the output from mixed 7 is becomiing the input (Dense)

x = layers.Dropout(0.2)(x)#dropping 20 percent of neurons to improve accuracy

x = layers.Dense  (1, activation='sigmoid')(x)#output neuron 1layer, activation sigmoid

model = Model( pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])


train_horses_dir =os.path.join('/Users/seanjudelyons/Downloads/horse-or-human/horses')
train_humans_dir =os.path.join('/Users/seanjudelyons/Downloads/horse-or-human/humans')
validation_horses_dir =os.path.join('/Users/seanjudelyons/Downloads/validation-horse-or-human/humans')
validation_humans_dir =os.path.join('/Users/seanjudelyons/Downloads/validation-horse-or-human/humans')

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))

# Expected Output:
# 500
# 527
# 128
# 128

train_dir = os.path.join('/Users/seanjudelyons/Downloads/horse-or-human/')
validation_dir = os.path.join('/Users/seanjudelyons/Downloads/validation-horse-or-human/')

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary',
                                                    target_size = (150, 150))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = 20,
                                                          class_mode  = 'binary',
                                                          target_size = (150, 150))

# Expected Output:
# Found 1027 images belonging to 2 classes.
# Found 256 images belonging to 2 classes.

callbacks = myCallback()
history = model.fit(
            train_generator,
            validation_data = validation_generator,
            epochs = 1,
            verbose = 1, callbacks=[callbacks])

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()