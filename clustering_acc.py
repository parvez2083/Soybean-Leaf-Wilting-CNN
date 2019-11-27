from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import data4projC as dt
import model4projC as mdl
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# Global variables
colours=['r','g','b','c','m','y','k']
num_epoch = 20
epochs = np.arange(num_epoch)+1
# Number of last epochs to show the average of recent loss/accuracy values
num_latest = math.ceil(num_epoch/5)

# Keys for the 'history' object
key_trn_loss = 'loss'
key_val_loss = 'val_loss'
key_trn_acc = 'sparse_categorical_accuracy'
key_val_acc = 'val_sparse_categorical_accuracy'

# Starting Training parameters.
learn_rate = 0.001      # Learning rate for the optimizer
batch_size = 32         # Training mini-batch size
hidden = 128            # Number of hidden units in the FCN
lmbda = 0.004           # Rgularization parameter for the conv. kernels
dropout = 0.4           # Dropout, probability to drop a unit
activation = 'tanh'     # Activation functions, except the output layer


def print_params():
    print('Lrate=',learn_rate, 
        ', batch size=',batch_size, 
        ', hidden neurons=',hidden, 
        ', dropout=',dropout, 
        ', lambda=',lmbda, 
        ', activation=',activation)

# Import training and testing data
x_train, y_train = dt.read_data_sets("../", isTrain=True, read_labels=True)
#x_test = dt.read_data_sets("../", isTrain=False, read_labels=False)
#print(len(x_train),len(x_test))
#x_train = x_train.reshape(-1,19200)
vgg16_model = tf.keras.applications.VGG16(include_top=False, 
                                          weights='imagenet', 
                                          input_tensor=None, 
                                          input_shape=(dt.rows, dt.cols, dt.channels), 
                                          pooling=None, 
                                          classes=5)
x_train = vgg16_model.predict(x_train)
print(x_train.shape)
x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
#x_train = tf.reshape(x_train,[896,15*512])
print(x_train.shape)

x_test = x_train[:dt.count_tst]
y_test = y_train[:dt.count_tst]
x_train = x_train[dt.count_tst:]
y_train = y_train[dt.count_tst:]
    

km = KMeans(init='k-means++', n_jobs=-1, n_clusters=5)
y_pred_train = km.fit_predict(x_train)
count_matrix = np.zeros((5, 5), dtype='uint16')
for ytr,ypr in zip(y_train, y_pred_train):
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

print(dict_class[y_pred_train],y_train)
print(np.sum(dict_class[y_pred_train]==y_train)/dt.count_trn)
    



