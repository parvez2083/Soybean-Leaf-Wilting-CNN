from __future__ import absolute_import, division, print_function

import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import matplotlib.pyplot as plt
#from tensorflow.keras.utils import plot_model
#from sklearn import preprocessing
import numpy as np
import data4projC as dt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# projC dataset parameters.
#dt.rows = 28
#dt.chan = 1
#dt.classes = 5 # total classes (0-9 digits).

# Training parameters.
learning_rate = 0.002
batch_size = 64
num_epoch = 10
lmbda = 0.001
dropout = 0.3 # Dropout, probability to drop a unit

# Network parameters.
conv1_filters = 8 # number of filters for 1st conv layer.
conv1_size = [21, 11] # size of filters for 1st conv layer.
conv2_filters = 16 # number of filters for 2nd conv layer.
conv2_size = 7 # size of filters for 2nd conv layer.
conv3_filters = 32 # number of filters for 2nd conv layer.
conv3_size = 3 # size of filters for 2nd conv layer.
fc1_units = 128 # number of neurons for 1st fully-connected layer.
#fc2_units = 64 # number of neurons for 2nd fully-connected layer.

# Prepare projC data.
#from tensorflow.keras.datasets import projC
#(x_train, y_train), (x_test, y_test) = projC.load_data()
# Import projC data
#from input import read_data_sets
(x_train, y_train), (x_val, y_val) = \
                    dt.read_data_sets("../", one_hot=False)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(conv1_filters, conv1_size, activation='relu',
                           input_shape=(dt.rows, dt.cols, dt.chan),
                            kernel_regularizer=tf.keras.regularizers.l2(lmbda)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 3)),
    tf.keras.layers.Conv2D(conv2_filters, conv2_size, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(lmbda)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(conv3_filters, conv3_size, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(lmbda)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(fc1_units, activation='relu'),
    #                        kernel_regularizer=tf.keras.regularizers.l2(lmbda)),
    tf.keras.layers.Dropout(dropout),
    #tf.keras.layers.Dense(fc2_units, activation='relu'),
    #                        kernel_regularizer=tf.keras.regularizers.l2(lmbda)),
    #tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(dt.classes, activation='sigmoid')
    #                        kernel_regularizer=tf.keras.regularizers.l2(lmbda))
])
model.summary()
#plot_model(model, to_file='model.png')

# Specify the training configuration (optimizer, loss, metrics)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),  # Optimizer
              # Loss function to minimize
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              # List of metrics to monitor
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Train the model by slicing the data into "batches"
# of size "batch_size", and repeatedly iterating over
# the entire dataset for a given number of "epochs"
print('# Fit model on training data')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epoch,
                    verbose=2,
                    validation_data=(x_val, y_val))

# The returned "history" object holds a record
# of the loss values and metric values during training
#print('\nhistory dict:', history.history)
for K in history.history.keys():
    plt.plot(history.history[K])
    plt.ylabel('loss, acc')
    plt.xlabel('epoch')
plt.legend(history.history.keys())
plt.show()


