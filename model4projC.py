import tensorflow as tf
import tensorflow.keras.layers as tfL
import data4projC as dt

# Network parameters.
conv1_filters = 8       # number of filters for 1st conv layer.
conv1_size = [21, 11]   # size of filters for 1st conv layer.
pool1_size = [2, 3]
conv2_filters = 16      # number of filters for 2nd conv layer.
conv2_size = 7          # size of filters for 2nd conv layer.
pool2_size = [2, 2]
conv3_filters = 32      # number of filters for 2nd conv layer.
conv3_size = 3          # size of filters for 2nd conv layer.
pool3_size = [2, 2]
#fc1_units = 128        # number of neurons for 1st fully-connected layer.
#fc2_units = 64         # number of neurons for 2nd fully-connected layer.

def get_model(Lrate, lmbda, drop, fc1_units, act):

    myModel = tf.keras.Sequential([
        tfL.Conv2D(conv1_filters, conv1_size, activation=act,
                    input_shape=(dt.rows, dt.cols, dt.chan),
                    kernel_regularizer=tf.keras.regularizers.l2(lmbda)),
        tfL.MaxPooling2D(pool_size=pool1_size),
        tfL.Conv2D(conv2_filters, conv2_size, activation=act,
                    kernel_regularizer=tf.keras.regularizers.l2(lmbda)),
        tfL.MaxPooling2D(pool_size=pool2_size),
        tfL.Conv2D(conv3_filters, conv3_size, activation=act,
                    kernel_regularizer=tf.keras.regularizers.l2(lmbda)),
        tfL.MaxPooling2D(pool_size=pool3_size),
        tfL.Flatten(),
        tfL.Dropout(drop),
        tfL.Dense(fc1_units, activation=act),
        tfL.Dropout(drop),
        #tfL.Dense(fc2_units, activation=act),
        #tfL.Dropout(dropout),
        tfL.Dense(dt.num_class, activation='softmax')
    ])
    #myModel.summary()

    # Specify the training configuration (optimizer, loss, metrics)
    myModel.compile(
                    # Optimization algorithm
                    optimizer=tf.keras.optimizers.Adam(Lrate),
                    # Loss function to minimize
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    # List of metrics to monitor
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return myModel