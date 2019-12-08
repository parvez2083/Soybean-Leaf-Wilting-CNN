# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


# -

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


# +
def custom_model(layer1_size, layer2_size, layer3_size, dropout):
    
    # initialize the model from the keras library   
#     custom_model = tf.keras.applications.VGG19(include_top=False,
#                                          weights="imagenet",
#                                          input_shape=(480, 640, 3),
#                                          classes=5)
    
    
    # initialize the VGG16 model from the keras library
    custom_model = tf.keras.applications.InceptionV3(include_top=False,
                                              weights="imagenet",
                                              input_shape=(480, 640, 3),
                                              classes=5)
    
    model = tf.keras.Sequential()   
    
#     custom_model.summary()

    # Freeze the layers
    custom_model.trainable = False
    
    model.add(custom_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(layer1_size, activation="relu"))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(layer2_size, activation="relu"))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(layer3_size, activation="relu"))
    model.add(tf.keras.layers.Dropout(dropout))

    # Add 'softmax' instead of earlier 'prediction' layer.
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    return model


# -

def generate_plots(model_history, epochs):
    """
    A method that takes the model history of a trained model and plots its:
    1. Training accuracy
    2. Training loss
    3. Validation accuracy
    4. Validation loss
    """
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']


    plt.figure(1)
    plt.suptitle('Accuracy learning curve', fontsize=20)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.plot(acc, label='training accuracy')
    plt.plot(val_acc, label='validation accuracy')
    plt.xticks(np.arange(0, epochs + epochs/10, epochs/10))
    plt.legend(loc="lower right")
    plt.savefig("accuracy.png", dpi=300)

    plt.figure(2)
    plt.suptitle('Loss learning curve', fontsize=20)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.plot(loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.xticks(np.arange(0, epochs + epochs/10, epochs/10))
    plt.legend(loc="upper right")
    plt.savefig("loss.png", dpi=300)


def best_results(save_model, model_name, model_history):
    """
    A method that prints the best validation accuracy and loss to a best_results.txt file.
    """
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    
    f = open("./{}/{}results.txt".format(save_model, model_name), "w+")
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

# +
save_model = "saved_models"
os.makedirs(save_model, exist_ok=True)

# Initialize the directories
root_dir = os.path.abspath('./')
train_dir = os.path.join(root_dir, 'Training')
model_save_dir = os.path.join(root_dir, save_model)


train_files = pd.read_csv(os.path.join(root_dir, 'train_files.csv'))

# set constants
COLS, ROWS, CHANNEL = 480, 640, 3

# read the training and testing images
training_images = read_training_images(train_files, train_dir)

X_train = np.stack(training_images)
y_train = train_files.annotation.values


# Split the training data in training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=None, shuffle=True, stratify=y_train)

print("Shape of training set: {}".format(X_train.shape))
print("Shape of validation set: {}".format(X_val.shape))

# +
# training variables
EPOCHS = 20
BATCH_SIZE = 8

model_name = "InceptionV3"
l1 = 512
l2 = 512
l3 = 256
dropout = 0.4
PARAM_STRING = "{}_{}_{}_{}".format(l1, l2, l3, dropout)

# +
print("-------------------")
print("Compiling model")
print("-------------------")

# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# mc = tf.keras.callbacks.ModelCheckpoint('{}/best_model.h5'.format(model_save_dir), monitor='val_loss', mode='main', save_best_only=True, verbose=1)

# initialize the model
model = custom_model(l1, l2, l3, dropout)

model.summary()

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

# +
print("-------------------")
print("Training model")
print("-------------------")

with tf.device('/device:GPU:0'):
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))
#     callbacks=[es, mc]

# +
# model.save('{}/best_model.h5'.format(model_save_dir)) 

# Recreate the exact same model, including its weights and the optimizer
# new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
# new_model.summary()
# -

generate_plots(history, EPOCHS)
best_results(save_model, model_name, history)


