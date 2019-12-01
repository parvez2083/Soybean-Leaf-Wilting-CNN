# -*- coding: utf-8 -*-
import os
import csv
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

"""
Two different methods for reading training and testing images since train_files.csv does not preserve order of files in the training images folder
"""


def read_training_images(train_files, train_dir):
    """
    A method that reads the training image files in a given directory and returns the list of all images as numpy arrays
    """

    images=[]
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
        images.append(img)
        n+=1
    print("Number of training images read = {}".format(n))

    return images



def read_testing_images(directory):
    """
    A method that reads the test image files in a given directory and returns the list of all images as numpy arrays
    """
    files = os.listdir(directory)

    images=[]
    m=0
    for fil in files:
        image_path = os.path.join(directory, fil)
        img = cv2.imread(image_path)
        # img = np.expand_dims(cv2.resize(img, dsize=(COLS, ROWS), interpolation=cv2.INTER_CUBIC), axis=2)
        img = img.astype('float32')
        images.append(img)
        m+=1
    print("Number of testing images read = {}".format(m))

    return images



def custom_vgg16(layer1_size=512, layer2_size=512, layer3_size=256, dropout=0.3, activation="relu", cols=480, rows=640, channel=3):
    """
    A model that generates a custom VGG16 model by adding dense layer to its bottom.
    The top layers from VGG16 were not used. The extracted VGG16 layers were used for
    feature representation and hence were not made trainable (frozen).
    This function retuns a custom VGG16 model with all the layers added. 
    """

    
    # initialize the VGG16 model from the keras library
    vgg16_model = tf.keras.applications.VGG16(include_top=False,
                                              weights='imagenet',
                                              input_tensor=None,
                                              input_shape=(cols, rows, channel),
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

    # Add 'softmax' instead of earlier 'prediction' layer.
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    return model



def generate_plots(model_history, epochs):
    """
    A method that takes the model history of a trained model and plots its:
    1. Training accuracy
    2. Training loss
    3. Validation accuracy
    4. Validation loss
    """
    acc = model_history.history['acc']
    val_acc = model_history.history['val_acc']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']


    plt.figure(1)
    plt.suptitle('Accuracy learning curve', fontsize=20)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.plot(acc, label='training accuracy')
    plt.plot(val_acc, label='validation accuracy')
    plt.xticks(np.arange(0, epochs, epochs/10))
    plt.legend(loc="lower right")
    plt.savefig("accuracy.png", dpi=300)

    plt.figure(2)
    plt.suptitle('Loss learning curve', fontsize=20)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.plot(loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.xticks(np.arange(0, epochs, epochs/10))
    plt.legend(loc="upper right")
    plt.savefig("loss.png", dpi=300)



def best_results(model_history):
    """
    A method that prints the best validation accuracy and loss to a best_results.txt file.
    """
    acc = model_history.history['acc']
    val_acc = model_history.history['val_acc']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    
    f = open("best_results.txt", "w+")
    f.write("Training accuracy = {}\n".format(acc))
    f.write("----------------------\n")
    f.write("Training loss = {}\n".format(loss))
    f.write("----------------------\n")
    f.write("Validation accuracy = {}\n".format(val_acc))
    f.write("----------------------\n")
    f.write("Validation loss = {}\n".format(val_loss))
    f.write("----------------------\n")
    f.write("----------------------\n\n")
    f.close()




def main():
    # Initialize the directories
    root_dir = os.path.abspath('./')
    train_dir = os.path.join(root_dir, 'Training')
    test_dir = os.path.join(root_dir, 'Project_C2_Testing')


    train_files = pd.read_csv(os.path.join(root_dir, 'train_files.csv'))

    # set constants
    COLS, ROWS, CHANNEL = 480, 640, 3

    # training constants
    EPOCHS = 30
    BATCH_SIZE = 16

    # read the training and testing images
    training_images = read_training_images(train_files, train_dir)
    testing_images = read_testing_images(test_dir)


    X_train = np.stack(training_images)
    y_train = train_files.annotation.values
    X_test = np.stack(testing_images)


    # Split the training data in training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=None, shuffle=True, stratify=y_train)

    print("Shape of training set: {}".format(X_train.shape))
    print("Shape of validation set: {}".format(X_val.shape))
    print("Shape of testing set: {}".format(X_test.shape))


    print("-------------------")
    print("Compiling model")
    print("-------------------")

    # initialize the model
    model = custom_vgg16(512, 512, 256, 0.4, "relu", COLS, ROWS, CHANNEL)

    model.summary()

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])


    print("-------------------")
    print("Training model")
    print("-------------------")

    with tf.device('/device:GPU:0'):
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    

    predictions = model.predict(X_test, batch_size=16)

    generate_plots(history, EPOCHS)

    best_results(history)

    with open('predictions.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(test_labels)



if __name__ == '__main__':
    main()
