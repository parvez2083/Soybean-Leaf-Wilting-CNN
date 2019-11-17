import os
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
import numpy as np
import tensorflow as tf


root_dir = os.path.abspath('.')
# Set the directory of the training data
train_dir = os.path.join(root_dir, 'Training')



train_files = pd.read_csv(os.path.join(root_dir, 'train_files.csv'))
print(train_files)

temp = []

for img_name in train_files.file_name:
    image_path = os.path.join(train_dir, img_name)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).flatten()
    # print(type(img))
    # print(img.shape)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)

train_x /= 255.0
train_x = train_x.reshape(-1, 307200).astype('float32')

train_y = train_files.annotation.values

split_size = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]

km = KMeans(n_jobs=-1, n_clusters=5, n_init=20)

# km.fit(train_x)
#
# pred = km.predict(val_x)
#
# score = normalized_mutual_info_score(val_y, pred)
# print(score)


# this is our input placeholder
input_img = tf.keras.Input(shape=(307200,))

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

train_history = autoencoder.fit(train_x, train_x, epochs=10, batch_size=16, validation_data=(val_x, val_x))

pred_auto_train = encoder.predict(train_x)
pred_auto = encoder.predict(val_x)

km.fit(pred_auto_train)
pred = km.predict(pred_auto)

score = normalized_mutual_info_score(val_y, pred)
print(score)
print(km.labels_)

