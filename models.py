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

y_train = train_files.annotation.values

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1, shuffle=True, stratify=y_train)


# +
# models = [tf.keras.applications.DenseNet121(include_top=True,
#                                             weights=None,
#                                             input_tensor=None,
#                                             input_shape=(COLS, ROWS, CHANNEL),
#                                             pooling=None,
#                                             classes=5),

#           tf.keras.applications.ResNet50(include_top=True,
#                                          weights=None,
#                                          input_tensor=None,
#                                          input_shape=(COLS, ROWS, CHANNEL),
#                                          pooling=None,
#                                          classes=5),

#           tf.keras.applications.MobileNetV2(include_top=True,
#                                             weights=None,
#                                             input_tensor=None,
#                                             input_shape=(COLS, ROWS, CHANNEL),
#                                             pooling=None,
#                                             classes=5)  ]


def custom_vgg16(layer1_size, layer2_size, layer3_size, dropout, activation):
    
    # initialize the VGG16 model from the keras library
    vgg16_model = tf.keras.applications.VGG16(include_top=False,
                                              weights='imagenet',
                                              input_tensor=None,
                                              input_shape=(COLS, ROWS, CHANNEL),
                                              pooling=None,
                                              classes=5)

    model = tf.keras.Sequential()

    for layer in vgg16_model.layers[:-1]: 
        model.add(layer)


    # Freeze the layers
    for layer in model.layers:
        layer.trainable = False

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(layer1_size, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(layer2_size, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(layer3_size, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    # model.add(tf.keras.layers.Dense(layer4_size, activation=activation))
    # model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    # Add 'softmax' instead of earlier 'prediction' layer.


    return model

# +
# Hyperparams

layer1 = [1024]
layer2 = [512]
layer3 = [512]
# layer3 = [256, 512]
# layer4 = [256, 512] 

dropout = [0.4]
activation = ["relu"]

EPOCHS = 30
BATCH_SIZE = 16

learning_rate = [0.0005]

# plot directory for accuracy and loss
validation = "./vgg/validation/results_better.txt"
plot_acc = "./vgg/acc/"
plot_loss = "./vgg/loss/"

os.makedirs(plot_acc, exist_ok=True)
os.makedirs(plot_loss, exist_ok=True)

# +
f = open(validation, "w+")

n=0

for l1 in layer1:
    for l2 in layer2:
        for l3 in layer3:
            for dr in dropout:
                for act in activation:
                    for lr in learning_rate:
                        print("-------------------")
                        print("Compiling model with {}, {}, {}, {}, {}, {}".format(l1, l2, l3, dr, act, lr))
                        print("-------------------")

                        # initialize the model
                        model = custom_vgg16(l1, l2, l3, dr, act)

                        model.summary()
                        model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                                            optimizer=tf.keras.optimizers.Adam(lr),
                                            metrics=['accuracy'])


                        print("-------------------")
                        print("Training model with {}, {}, {}, {}, {}, {}".format(l1, l2, l3, dr, act, lr))
                        print("-------------------")


                        with tf.device('/device:GPU:0'):

                            history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

                            acc = history.history['acc']
                            val_acc = history.history['val_acc']
                            loss = history.history['loss']
                            val_loss = history.history['val_loss']


                            plt.figure(n)
                            plt.suptitle('Accuracy learning curve', fontsize=20)
                            plt.xlabel('epochs', fontsize=14)
                            plt.ylabel('accuracy', fontsize=14)
                            plt.plot(acc, label='training accuracy')
                            plt.plot(val_acc, label='validation accuracy')
                            plt.xticks(np.arange(0, EPOCHS, EPOCHS/10))
                            plt.legend(loc="lower right")
                            plt.savefig("{}_{}_{}_{}_{}_{}_{}.png".format(plot_acc, l1, l2, l3, dr, act, lr), dpi=500)

                            plt.figure(n+500)
                            plt.suptitle('Loss learning curve', fontsize=20)
                            plt.xlabel('epochs', fontsize=14)
                            plt.ylabel('loss', fontsize=14)
                            plt.plot(loss, label='training loss')
                            plt.plot(val_loss, label='validation loss')
                            plt.xticks(np.arange(0, EPOCHS, EPOCHS/10))
                            plt.legend(loc="upper right")
                            plt.savefig("{}_{}_{}_{}_{}_{}_{}.png".format(plot_loss, l1, l2, l3, dr, act, lr), dpi=500)


                            f.write("layer1: {}, layer2: {}, layer3: {}, Dropout: {}, Activation: {}, Learning rate: {}\n".format(l1, l2, l3, dr, act, lr))
                            f.write("Validation accuracy = {}\n".format(val_acc))
                            f.write("----------------------\n")
                            f.write("Validation loss = {}\n".format(val_loss))
                            f.write("----------------------\n")
                            f.write("----------------------\n\n")

                            n+=1

            

f.close()                    
# -



# +
# model_names = ["Densenet", "Resnet50", "Mobilenetv2"]
# model_names = ["VGG16 custom"]

# +

# for i, mdl in enumerate(models):

#     # initialize the model
#     model = mdl

#     model.summary()
#     model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
#                            optimizer=tf.keras.optimizers.Adam(),
#                            metrics=['accuracy'])


#     print("-------------------")
#     print("Training {}".format(model_names[i]))
#     print("-------------------")


#     with tf.device('/device:GPU:0'):

#         history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

#         # acc = accuracy_score(y_test, y_pred)
#         # pre = precision_score(y_test, y_pred, average='micro')
#         # rec = recall_score(y_test, y_pred, average='micro')
#         # f1 = f1_score(y_test, y_pred, average='micro')
#         #
#         # print('Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}'.format(acc, pre, rec, f1))

#         acc = history.history['acc']
#         val_acc = history.history['val_acc']
#         loss = history.history['loss']
#         val_loss = history.history['val_loss']


#         plt.figure(i)
#         plt.suptitle('Peformance', fontsize=20)
#         plt.xlabel('epochs', fontsize=14)
#         plt.ylabel('performance', fontsize=14)
#         plt.plot(acc)
#         plt.plot(val_acc)
#         plt.plot(loss)
#         plt.plot(val_loss)
#         plt.xticks(np.arange(0, EPOCHS, EPOCHS/10))
#         plt.savefig("./performance/{}_performance.png".format(model_names[i]), dpi=500)
# -




