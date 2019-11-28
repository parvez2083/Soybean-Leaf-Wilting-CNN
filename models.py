# -*- coding: utf-8 -*-

import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

root_dir = os.path.abspath('./')
# root_dir = os.path.abspath('./drive/My Drive/C1/Project_C1/')
# Set the directory of the training data
train_dir = os.path.join(root_dir, 'Training')

train_files = pd.read_csv(os.path.join(root_dir, 'train_files.csv'))

# set constants
COLS, ROWS, CHANNEL = 480, 640, 3
EPOCHS = 100
BATCH_SIZE = 16

temp = []
n=0
for img_name in train_files.file_name:
    image_path = os.path.join(train_dir, img_name)
    img = cv2.imread(image_path)
    # img = np.expand_dims(cv2.resize(img, dsize=(COLS, ROWS), interpolation=cv2.INTER_CUBIC), axis=2)
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
# X_train = X_train.reshape(n, COLS, ROWS, CHANNEL)

y_train = train_files.annotation.values

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1, shuffle=True, stratify=None)


models = [tf.keras.applications.DenseNet121(include_top=True,
                                            weights=None,
                                            input_tensor=None,
                                            input_shape=(COLS, ROWS, CHANNEL),
                                            pooling=None,
                                            classes=5),

          tf.keras.applications.ResNet50(include_top=True,
                                         weights=None,
                                         input_tensor=None,
                                         input_shape=(COLS, ROWS, CHANNEL),
                                         pooling=None,
                                         classes=5),

          tf.keras.applications.MobileNetV2(include_top=True,
                                            weights=None,
                                            input_tensor=None,
                                            input_shape=(COLS, ROWS, CHANNEL),
                                            pooling=None,
                                            classes=5)  ]

model_names = ["Densenet", "Resnet", "Mobilenetv2"]

for i, mdl in enumerate(models):

    print("-------------------")
    print("Compiling {}".format(model_names[i]))
    print("-------------------")

    # initialize the model
    model = mdl

    model.summary()
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])


    print("-------------------")
    print("Training {}".format(model_names[i]))
    print("-------------------")


    with tf.device('/device:GPU:0'):

        history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

        # acc = accuracy_score(y_test, y_pred)
        # pre = precision_score(y_test, y_pred, average='micro')
        # rec = recall_score(y_test, y_pred, average='micro')
        # f1 = f1_score(y_test, y_pred, average='micro')
        #
        # print('Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}'.format(acc, pre, rec, f1))

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']


    plt.figure(i)
    plt.suptitle('Peformance', fontsize=20)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('performance', fontsize=14)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.xticks(np.arange(0, EPOCHS, EPOCHS/10))
    plt.savefig('./{}_performance.png'.format(model))
