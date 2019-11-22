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
    # .flatten()
    # print(n)
    # print(type(img))
    # print(img.shape)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = img.astype('float32')
    temp.append(img)
    n+=1

X_train = np.stack(temp)

# change this for different shapes
X_train = X_train.reshape(n, 480, 640, 1)


# train_x /= 255.0
# train_x = train_x.reshape(-1, 307200).astype('float32')

y_train = train_files.annotation.values

# Use first 700 examples or training, 100 for testing, 96 for validation
train_size = 700
test_size = 100

X_train, X_test = X_train[:train_size], X_train[train_size:]
y_train, y_test = y_train[:train_size], y_train[train_size:]

# X_test, X_val = X_test[:test_size], X_test[test_size:]
# y_test, y_val = y_test[:test_size], y_test[test_size:]

print(X_train.shape)
print(X_test.shape)
# print(X_val.shape)



with tf.device('/device:GPU:0'):

    vgg16_model = tf.keras.applications.VGG16(include_top=False, 
                                            weights=None, 
                                            input_tensor=None, 
                                            input_shape=(480, 640, 1), 
                                            pooling=None, 
                                            classes=5)

    vgg16_model.summary()          



    # print(vgg16_model.layers[-1].output_shape)                             

    # model = tf.keras.Sequential()

    # # Remove the prediction layer and add to new model
    # for layer in vgg16_model.layers[:-1]: 
    #     model.add(layer)    

    # # Freeze the layers 
    # for layer in model.layers:
    #     layer.trainable = False

    # # Add 'softmax' instead of earlier 'prediction' layer.
    # model.add(tf.keras.layers.Dense(5, activation='softmax'))

    y_pred = vgg16_model.predict(X_train)





    # model.summary()

    # model.compile(optimizer=tf.keras.optimizers.Adam(), 
    #               loss=tf.keras.losses.sparse_categorical_crossentropy,
    #               metrics=["accuracy"])




    # # prevents changing the weights and freezes them
    # vgg16_model.trainable=False
    # global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    # prediction_layer = tf.keras.layers.Dense(5,activation='softmax')

    # model = tf.keras.Sequential([
    #   vgg16_model,
    #   global_average_layer,
    #   prediction_layer
    # ])



    print(y_pred.shape)





    # Do k-means clustering on the VGG_16 output
    km = KMeans(n_jobs=-1, n_clusters=5, n_init=20)



    # this is our input placeholder
    input_img = tf.keras.Input(shape=(y_pred.shape))

    # "encoded" is the encoded representation of the input
    encoded = tf.keras.layers.Dense(500, activation='relu')(input_img)
    encoded = tf.keras.layers.Dense(500, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(2000, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(10, activation='sigmoid')(encoded)

    # "decoded" is the lossy reconstruction of the input
    decoded = tf.keras.layers.Dense(2000, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(500, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(500, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(307200)(decoded)

    # this model maps an input to its reconstruction
    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.summary()

    #  this model maps an input to its encoded representation
    encoder = tf.keras.Model(input_img, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    train_history = autoencoder.fit(y_pred, y_pred, epochs=10, batch_size=16)

    pred_auto_train = encoder.predict(X_train)
    pred_auto = encoder.predict(X_test)

    km.fit(pred_auto_train)
    pred = km.predict(pred_auto)

    score = normalized_mutual_info_score(y_test, pred)
    print(score)