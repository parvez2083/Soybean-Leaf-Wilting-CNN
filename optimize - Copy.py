from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from loadMnist import read_data_sets
import model4mnist as mm

# Training parameters.
learn_rate = 0.001
batch_size = 256
hidden = 128
dropout = 0.5 # Dropout, probability to drop a unit
activation = 'relu'
colours=['r','g','b','c','m','y','k']
trn_acc = 'sparse_categorical_accuracy'
val_acc = 'val_sparse_categorical_accuracy'

# Prepare MNIST data.
#from tensorflow.keras.datasets import mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Import MNIST data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = \
                    read_data_sets("../data/", one_hot=False)

x_train = x_train.astype('float32') / 255
x_val = x_val.astype('float32') / 255
x_test = x_test.astype('float32') / 255
#print(len(x_train[0]),len(x_val[0]),len(x_test[0]))

y_train = y_train.astype('float32')
y_val = y_val.astype('float32')
y_test = y_test.astype('float32')

num_iter = 6
iterations = np.arange(num_iter)
num_epoch = 10
epochs = np.arange(num_epoch)+1
for lr in iterations:
    learn_rate = (lr+1)/2000.
    print('\nTrying learning rate = ', learn_rate)
    model = mm.get_model(learn_rate, dropout, hidden, activation)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epoch,
                        verbose=2,
                        validation_data=(x_val, y_val))
    plt.plot(epochs,history.history['loss'],
            colours[lr],label=learn_rate)

plt.legend(loc = 'upper right', title='Learning rate')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
#plt.title('Training loss vs learning rate')
plt.savefig('../images/learn.png')
plt.show()
learn_rate = float(input('\nChoose the best learning rate: '))
print('learning rate=',learn_rate, ', batch size=',batch_size, 
    ', dropout=',dropout, ', hidden neurons=',hidden, ', activation=',activation)

batch_size = [16, 32, 64, 128, 256, 512]
ac = np.zeros(num_iter)
tm = np.zeros(num_iter)
for B in iterations:
    print('\nTrying batch size = ', batch_size[B])
    model = mm.get_model(learn_rate, dropout, hidden, activation)

    start_time = time.time()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size[B],
                        epochs=num_epoch,
                        verbose=2,
                        validation_data=(x_val, y_val))
    tm[B] = (time.time() - start_time)
    ac[B] = history.history[trn_acc][-1]
    plt.plot(epochs,history.history['loss'],
            colours[B],label=batch_size[B])

plt.legend(loc = 'upper right', title='Batch size')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
#plt.title('Training loss vs batch size')
plt.savefig('../images/batch_loss.png')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.semilogx(batch_size,tm,'b',basex=2)
ax1.set_ylabel('Execution time (sec)', color='b')
ax1.set_xlabel('Batch size')
ax2 = ax1.twinx()
ax2.semilogx(batch_size,ac,'r',basex=2)
#ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax2.yticks(rotation='vertical')
ax2.set_ylabel('Training accuracy', color='r')
#plt.title('trade-off between training accuracy and execution time')
fig.tight_layout()
plt.savefig('../images/batch_time.png')
plt.show()
batch_size = int(input("\nChoose the best batch size: "))
print('learning rate=',learn_rate, ', batch size=',batch_size, 
    ', dropout=',dropout, ', hidden neurons=',hidden, ', activation=',activation)

num_epoch = num_epoch*2
epochs = np.arange(num_epoch)+1
trn = np.zeros(num_iter)
val = np.zeros(num_iter)
plt.figure()
f, axes = plt.subplots(2, 1)
for dr in iterations:
    dropout = (dr+1)/10.
    print('\nTrying dropout probability = ', dropout)
    model = mm.get_model(learn_rate, dropout, hidden, activation)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epoch,
                        verbose=2,
                        validation_data=(x_val, y_val))
    trn[dr] = history.history[trn_acc][-1]
    val[dr] = history.history[val_acc][-1]
    axes[0].plot(epochs,history.history['loss'],
            colours[dr],label=dropout)
    axes[1].plot(epochs,history.history['val_loss'],
            colours[dr],label=dropout)

axes[0].legend(loc = 'upper right', ncol=2, title='Dropout probability')
axes[0].set_ylabel('Training loss')
#axes[0].set_xlabel('iterations')
axes[1].legend(loc = 'upper right', ncol=2, title='Dropout probability')
axes[1].set_ylabel('Validation loss')
axes[1].set_xlabel('Epochs')
#plt.title('Training and validation loss vs dropout probability')
plt.savefig('../images/drop_loss.png')
plt.show()

plt.figure()
plt.plot((iterations+1)/10,trn,'b',(iterations+1)/10,val,'r')
#plt.plot([(dr+1)/10 for dr in iterations],trn,'b',
#        [(dr+1)/10 for dr in iterations],val,'r')
plt.legend(('Training','Validation'), loc = 'lower left')
plt.xlabel('Dropout probability')
plt.ylabel('Accuracy')
#plt.title('training and validation accuracy vs dropout probability')
plt.savefig('../images/drop_acc.png')
plt.show()
dropout = float(input("\nChoose the best dropout probability: "))
print('learning rate=',learn_rate, ', batch size=',batch_size, 
    ', dropout=',dropout, ', hidden neurons=',hidden, ', activation=',activation)

num_iter = 4
iterations = np.arange(num_iter)
hidden = [64, 128, 256, 512]
trn = np.zeros(num_iter)
val = np.zeros(num_iter)
plt.figure()
f, axes = plt.subplots(2, 1)
for H in iterations:
    print('\nTrying hidden layer size = ', hidden[H])
    model = mm.get_model(learn_rate, dropout, hidden[H], activation)

    start_time = time.time()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epoch,
                        verbose=2,
                        validation_data=(x_val, y_val))
    trn[H] = history.history[trn_acc][-1]
    val[H] = history.history[val_acc][-1]
    axes[0].plot(epochs,history.history['loss'],
            colours[H],label=hidden[H])
    axes[1].plot(epochs,history.history['val_loss'],
            colours[H],label=hidden[H])

axes[0].legend(loc = 'upper right', title='Hidden neurons')
axes[0].set_ylabel('Training loss')
#axes[0].set_xlabel('iterations')
axes[1].legend(loc = 'upper right', title='Hidden neurons')
axes[1].set_ylabel('Validation loss')
axes[1].set_xlabel('Epochs')
#plt.title('Training loss vs size of hidden layer')
plt.savefig('../images/hidden_loss.png')
plt.show()

plt.figure()
plt.semilogx(hidden,trn,'b',basex=2)
plt.semilogx(hidden,val,'r',basex=2)
plt.legend(('Training','Validation'))
plt.xlabel('Size of hidden layer')
plt.ylabel('Accuracy')
#plt.title('training and validation accuracy vs size of hidden layer')
plt.savefig('../images/hidden_acc.png')
plt.show()
hidden = int(input("\nChoose the best size of hidden layer: "))
print('learning rate=',learn_rate, ', batch size=',batch_size, 
    ', dropout=',dropout, ', hidden neurons=',hidden, ', activation=',activation)

#num_epoch = num_epoch*2
#epochs = np.arange(num_epoch)+1
trn = np.zeros(num_iter)
val = np.zeros(num_iter)
model = [0] *num_iter
activation = ['sigmoid', 'relu', 'tanh', 'softmax']
plt.figure()
f, axes = plt.subplots(2, 1)
for A in iterations:
    print('\nTrying activation function = ', activation[A])
    model[A] = mm.get_model(learn_rate, dropout, hidden, activation[A])

    history = model[A].fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epoch,
                        verbose=2,
                        validation_data=(x_val, y_val))
    trn[A] = history.history[trn_acc][-1]
    val[A] = history.history[val_acc][-1]
    axes[0].plot(epochs,history.history['loss'],
            colours[A],label=activation[A])
    axes[1].plot(epochs,history.history['val_loss'],
            colours[A],label=activation[A])

axes[0].legend(loc = 'upper right')
axes[0].set_ylabel('Training loss')
#axes[0].set_xlabel('iterations')
axes[1].legend(loc = 'upper right')
axes[1].set_ylabel('Validation loss')
axes[1].set_xlabel('Epochs')
#plt.title('Training and validation loss vs activation function')
plt.savefig('../images/act_loss.png')
plt.show()

w = 0.4
plt.subplots()
plt.bar(iterations-w/2,trn, width=w, color='b')
plt.bar(iterations+w/2,val, width=w, color='r')
plt.xticks(iterations, activation)
plt.legend(('Training','Validation'))
plt.ylabel('Accuracy')
plt.ylim(bottom=math.floor(trn[-1]*20)/20)
#plt.title('training and validation accuracy vs activation function')
plt.savefig('../images/act_acc.png')
plt.show()
A = int(input('\nChoose the best activation function: '))
print('learning rate=',learn_rate, ', batch size=',batch_size, 
    ', dropout=',dropout, ', hidden neurons=',hidden, ', activation=',activation[A])

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
#results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
#print('test loss, test acc:', results)

predictions = model[A].predict(x_test)
row_max = predictions.max(axis=1)
pred_normed = predictions / row_max[:, np.newaxis]
np.savetxt("mnist.csv", pred_normed, 
            delimiter=",", fmt='%i')
#pred_csv = predictions.to_csv()
#with open("mnist.csv", "w") as csv_file:
#  csv_file.write(pred_csv)
y_pred = np.argmax(predictions, axis=1)
accuracy = np.count_nonzero(y_test==y_pred)/10000.
print('test accuracy:', accuracy)

# Visualize predictions.

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\nShow predictions for some failed test samples')
idx = np.random.randint(10000, size = 200)

# Display image and model prediction.
for i in idx:
    if y_pred[i] == y_test[i]:
        continue
    print("Model prediction: %i, test_label: %i" % (y_pred[i],y_test[i]))
    print(predictions[i])
    plt.imshow(np.reshape(x_test[i], [mm.input_size, mm.input_size]), cmap='gray')
    plt.show()

