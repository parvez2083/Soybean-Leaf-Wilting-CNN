# -*- coding: utf-8 -*-

import os
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
import numpy as np
import tensorflow as tf


root_dir = os.path.abspath('./')
# Set the directory of the training data
train_dir = os.path.join(root_dir, 'Training')

train_files = pd.read_csv(os.path.join(root_dir, 'train_files.csv'))

temp = []
n=0
for img_name in train_files.file_name:
    image_path = os.path.join(train_dir, img_name)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype('float32')
    temp.append(img)
    n+=1

X_train = np.stack(temp)

# change this for different shapes
X_train = X_train.reshape(n, 480, 640, 1)


# X_train /= 255.0
# train_x = train_x.reshape(-1, 307200).astype('float32')

y_train = train_files.annotation.values

# Use first 700 examples or training, 100 for testing, 96 for validation
train_size = 700
test_size = 96
val_size = 100


X_test, X_train = X_train[:test_size], X_train[test_size:]
y_test, y_train = y_train[:test_size], y_train[test_size:]

X_val, X_train = X_train[:val_size], X_train[val_size:]
y_val, y_train = y_train[:val_size], y_train[val_size:]

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)



# initialize the VGG16 model from the keras library
vgg16_model = tf.keras.applications.VGG16(include_top=False,
                                        weights=None,
                                        input_tensor=None,
                                        input_shape=(480, 640, 1),
                                        pooling=None,
                                        classes=5)

vgg16_model.summary()


# get the predictions from the model
X_train_vgg = vgg16_model.predict(X_train)
X_val_vgg = vgg16_model.predict(X_val)

print(X_train_vgg.shape)


# Do k-means clustering on the VGG_16 output
km = KMeans(n_jobs=-1, n_clusters=5, n_init=20)

"""
We would pass the output of the VGG16 model to the k-means clustering algorithm, check its performance.
Here, n is 5, which is the number of class labels.
"""

# this is our input placeholder
input_img = tf.keras.Input(shape=(15, 20, 512, ))

# "encoded" is the encoded representation of the input
encoded = tf.keras.layers.Dense(500, activation='relu')(input_img)
encoded = tf.keras.layers.Dense(500, activation='relu')(encoded)
encoded = tf.keras.layers.Dense(2000, activation='relu')(encoded)
encoded = tf.keras.layers.Dense(10, activation='sigmoid')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = tf.keras.layers.Dense(2000, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(500, activation='relu')(decoded)
decoded = tf.keras.layers.Dense(500, activation='relu')(decoded)
decoded = tf.keras.layers.Dense(512)(decoded)

# this model maps an input to its reconstruction
autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.summary()

#  this model maps an input to its encoded representation
encoder = tf.keras.Model(input_img, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# train the autoencoder
train_history = autoencoder.fit(X_train_vgg, X_train_vgg, epochs=10, batch_size=16)

pred_auto_train = encoder.predict(X_train_vgg)
pred_auto = encoder.predict(X_val_vgg)

# print(pred_auto_train.shape)
# print(pred_auto.shape)

pred_auto_train = pred_auto_train.reshape(-1, 3000).astype('float32')
pred_auto = pred_auto.reshape(-1, 3000).astype('float32')


# fit the k means clustering with the output from the autoencoder
kmeans = km.fit(pred_auto_train)
pred = km.predict(pred_auto)

# find the score from the k-means output
score = normalized_mutual_info_score(y_val, pred)
print(score)

print(kmeans.labels_)
print(y_val)
