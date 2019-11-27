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
X_train = X_train.reshape(n, COLS, ROWS, CHANNEL)


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



with tf.device('/device:GPU:0'):

    # initialize the VGG16 model from the keras library
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




    # get the predictions from the model
    # X_train_vgg = vgg16_model.predict(X_train)
    # X_val_vgg = vgg16_model.predict(X_val)
    # print(X_train_vgg.shape)



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


    # Do k-means clustering on the VGG_16 output
    km = KMeans(n_jobs=-1, n_clusters=5, n_init=20)

    """
    We would pass the output of the VGG16 model to the k-means clustering algorithm, check its performance.
    Here, n is 5, which is the number of class labels.
    """

    # this is our input placeholder
    input_img = tf.keras.Input(shape=(COLS, ROWS, CHANNEL, ))

    # "encoded" is the encoded representation of the input
    encoded = tf.keras.layers.Dense(500, activation='relu')(input_img)
    encoded = tf.keras.layers.Dense(500, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(2000, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(ENCODED_LAYER_SIZE, activation='sigmoid')(encoded)

    # "decoded" is the lossy reconstruction of the input
    decoded = tf.keras.layers.Dense(2000, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(500, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(500, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(CHANNEL)(decoded)

    # this model maps an input to its reconstruction
    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.summary()

    #  this model maps an input to its encoded representation
    encoder = tf.keras.Model(input_img, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    # train the autoencoder
    train_history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=32)

    pred_auto_train = encoder.predict(X_train)
    pred_auto = encoder.predict(X_val)

    print(pred_auto_train.shape)
    print(pred_auto.shape)

    pred_auto_train = pred_auto_train.reshape(-1, train_size*COLS*ROWS*ENCODED_LAYER_SIZE).astype('float32')
    pred_auto = pred_auto.reshape(-1, val_size*COLS*ROWS*ENCODED_LAYER_SIZE).astype('float32')


    # fit the k means clustering with the output from the autoencoder
    kmeans = km.fit(pred_auto_train)
    y_pred = km.predict(pred_auto)

    # find the score from the k-means output
    score = normalized_mutual_info_score(y_val, y_pred)
    print(score)

    count_matrix = np.zeros((5, 5), dtype='uint16')
    for ytr,ypr in zip(y_train, y_pred):
        count_matrix[ypr][ytr] = count_matrix[ypr][ytr]+1

    dict_class = np.zeros(5, dtype='uint8')
    for i in range(5):
        #print(count_matrix)
        idx = np.argmax(count_matrix)
        r = int(idx/5)
        c = idx - r*5
        dict_class[np.argmax(count_matrix[:,c])] = c
        #print((count_matrix[r,c]))

        for j in range(5):
            count_matrix[j,c] = 0
            count_matrix[r,j] = 0
        #max_count = np.amax(count_matrix, axis=1)

    print(dict_class)

    print(dict_class[y_pred],y_train)
    print(np.sum(dict_class[y_pred]==y_train)/n)