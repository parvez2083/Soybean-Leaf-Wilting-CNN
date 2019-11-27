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

# set constants
COLS, ROWS, CHANNEL = 120, 160, 1
ENCODED_LAYER_SIZE = 10

temp = []
n=0
for img_name in train_files.file_name:
    image_path = os.path.join(train_dir, img_name)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(cv2.resize(img, dsize=(COLS, ROWS), interpolation=cv2.INTER_CUBIC), axis=2)

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
X_train = X_train.reshape(n, COLS, ROWS, 1)

# X_train /= 255.0
# train_x = train_x.reshape(-1, 307200).astype('float32')

y_train = train_files.annotation.values

# Use first 700 examples or training, 100 for testing, 96 for validation
train_size = 749
test_size = 146
# val_size = 100


X_train, X_test = X_train[:train_size], X_train[train_size:]
y_train, y_test = y_train[:train_size], y_train[train_size:]

# X_val, X_train = X_train[:val_size], X_train[val_size:]
# y_val, y_train = y_train[:val_size], y_train[val_size:]

# print(X_train.shape)
# print(X_test.shape)
# print(X_val.shape)



with tf.device('/device:GPU:0'):

    # initialize the VGG16 model from the keras library
    densenet_model = tf.keras.applications.DenseNet121(include_top=False, 
                                            weights='imagenet', 
                                            input_tensor=None, 
                                            input_shape=(COLS, ROWS, CHANNEL), 
                                            pooling=None, 
                                            classes=5)

    densenet_model.summary()          



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


    # get the predictions from the model
    y_pred = densenet_model.predict(X_test)

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    acc = accuracy_score(y_test, y_pred)



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



    print(X_train_vgg.shape)


    


