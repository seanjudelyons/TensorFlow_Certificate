import tensorflow as tf
import numpy as np
from tensorflow import keras

# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([100000.0, 150000.0, 200000.0, 250000.0, 300000.0, 350000.0, 400000.0, 450000.0, 500000.0, 550000.0], dtype=float)
    ys = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape= [1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(ys, xs, epochs=5000)
    return (model.predict(y_new)[0]) //100000

prediction = house_model([7.0])

print(prediction)