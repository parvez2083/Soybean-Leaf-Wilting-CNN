from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import data4projC as dt
import model4projC as mdl

# Global variables
colours=['r','g','b','c','m','y','k']
num_epoch = 20
epochs = np.arange(num_epoch)+1
# Number of last epochs to average the loss/accuracy values
num_latest = math.ceil(num_epoch/5)

# Keys for the 'history' object
key_trn_loss = 'loss'
key_val_loss = 'val_loss'
key_trn_acc = 'sparse_categorical_accuracy'
key_val_acc = 'val_sparse_categorical_accuracy'

# Starting Training parameters.
learn_rate = 0.001      # Learning rate for the optimizer
batch_size = 64         # Training mini-batch size
hidden = 128            # Number of hidden units in the FCN
lmbda = 0.001           # Rgularization parameter for the conv. kernels
dropout = 0.3           # Dropout, probability to drop a unit
activation = 'relu'     # Activation functions, except the output layer


def print_params():
    print('Lrate=',learn_rate, 
        ', batch size=',batch_size, 
        ', hidden neurons=',hidden, 
        ', dropout=',dropout, 
        ', lambda=',lmbda, 
        ', activation=',activation)

# Import training, validation, and testing data
(x_train, y_train), (x_val, y_val) = \
                    dt.read_data_sets("../", one_hot=False)

def train_model():
    # Create a CNN model and train with data
    model = mdl.get_model(learn_rate, lmbda, dropout, hidden, activation)
    hist = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epoch,
                    verbose=2,
                    validation_data=(x_val, y_val))
    return model, hist

def tune_learning_rate(Lrate_options):
    global learn_rate
    #Lrate_options = (iterations+1)/2000
    num_iter = len(Lrate_options)
    iterations = np.arange(num_iter)
    for iter in iterations:
        learn_rate = Lrate_options[iter]
        print('\nTrying learning rate = ', learn_rate)
        model, history = train_model()
        plt.plot(epochs,history.history[key_trn_loss],
                colours[iter],label=learn_rate)

    plt.legend(loc = 'upper right', title='Learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    #plt.title('Training loss vs learning rate')
    plt.savefig('./plots/learn.png')
    plt.show()
    learn_rate = float(input('\nChoose the best learning rate: '))
    print_params()

def tune_batch_size(batch_options):
    global batch_size
    #batch_options = [16, 32, 64, 128, 256, 512]
    num_iter = len(batch_options)
    iterations = np.arange(num_iter)
    ac = np.zeros(num_iter)
    tm = np.zeros(num_iter)
    for iter in iterations:
        batch_size = batch_options[iter]
        print('\nTrying batch size = ', batch_size)
        start_time = time.time()
        model, history = train_model()
        tm[iter] = (time.time() - start_time)
        ac[iter] = np.mean(history.history[key_trn_acc][-num_latest:])
        plt.plot(epochs,history.history[key_trn_loss],
                colours[iter],label=batch_size)

    plt.legend(loc = 'upper right', title='Batch size')
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    #plt.title('Training loss vs batch size')
    plt.savefig('./plots/batch_loss.png')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.semilogx(batch_options,tm,'b',basex=2)
    ax1.set_ylabel('Execution time (sec)', color='b')
    ax1.set_xlabel('Batch size')
    ax2 = ax1.twinx()
    ax2.semilogx(batch_options,ac,'r',basex=2)
    ax2.set_ylabel('Training accuracy', color='r')
    #plt.title('trade-off between training accuracy and execution time')
    fig.tight_layout()
    plt.savefig('./plots/batch_time.png')
    plt.show()
    batch_size = int(input("\nChoose the best batch size: "))
    print_params()

def tune_dropout_probability(dropout_options):
    global dropout
    #dropout_options = (iterations+1)/10
    num_iter = len(dropout_options)
    iterations = np.arange(num_iter)
    trn = np.zeros(num_iter)
    val = np.zeros(num_iter)
    #plt.figure()
    f, axes = plt.subplots(2, 1)
    for iter in iterations:
        dropout = dropout_options[iter]
        print('\nTrying dropout probability = ', dropout)
        model, history = train_model()
        trn[iter] = np.mean(history.history[key_trn_acc][-num_latest:])
        val[iter] = np.mean(history.history[key_val_acc][-num_latest:])
        axes[0].plot(epochs,history.history[key_trn_loss],
                colours[iter],label=dropout)
        axes[1].plot(epochs,history.history[key_val_loss],
                colours[iter],label=dropout)

    axes[0].legend(loc = 'upper right', ncol=2, title='Dropout probability')
    axes[0].set_ylabel('Training loss')
    #axes[0].set_xlabel('iterations')
    axes[1].legend(loc = 'upper right', ncol=2, title='Dropout probability')
    axes[1].set_ylabel('Validation loss')
    axes[1].set_xlabel('Epochs')
    #plt.title('Training and validation loss vs dropout probability')
    plt.savefig('./plots/drop_loss.png')
    plt.show()

    plt.figure()
    plt.plot(dropout_options,trn,'b',dropout_options,val,'r')
    plt.legend(('Training','Validation'), loc = 'best')
    plt.xlabel('Dropout probability')
    plt.ylabel('Accuracy')
    #plt.title('training and validation accuracy vs dropout probability')
    plt.savefig('./plots/drop_acc.png')
    plt.show()
    dropout = float(input("\nChoose the best dropout probability: "))
    print_params()

def tune_regularization(lmbda_options):
    global lmbda
    #lmbda_options = (iterations+1)/1000
    num_iter = len(lmbda_options)
    iterations = np.arange(num_iter)
    trn = np.zeros(num_iter)
    val = np.zeros(num_iter)
    #plt.figure()
    f, axes = plt.subplots(2, 1)
    for iter in iterations:
        lmbda = lmbda_options[iter]
        print('\nTrying regularization parameter = ', lmbda)
        model, history = train_model()
        trn[iter] = np.mean(history.history[key_trn_acc][-num_latest:])
        val[iter] = np.mean(history.history[key_val_acc][-num_latest:])
        axes[0].plot(epochs,history.history[key_trn_loss],
                colours[iter],label=lmbda)
        axes[1].plot(epochs,history.history[key_val_loss],
                colours[iter],label=lmbda)

    axes[0].legend(loc = 'upper right', ncol=2, title='Regularization parameter')
    axes[0].set_ylabel('Training loss')
    #axes[0].set_xlabel('iterations')
    axes[1].legend(loc = 'upper right', ncol=2, title='Regularization parameter')
    axes[1].set_ylabel('Validation loss')
    axes[1].set_xlabel('Epochs')
    #plt.title('Training and validation loss vs regularization parameter')
    plt.savefig('./plots/lambda_loss.png')
    plt.show()

    plt.figure()
    plt.plot(lmbda_options,trn,'b',lmbda_options,val,'r')
    plt.legend(('Training','Validation'), loc = 'best')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Accuracy')
    #plt.title('training and validation accuracy vs regularization parameter')
    plt.savefig('./plots/lambda_acc.png')
    plt.show()
    lmbda = float(input("\nChoose the best L2 regularization parameter: "))
    print_params()

def tune_hidden_layer(hidden_options):
    global hidden
    #hidden_options = [16, 32, 64, 128, 256, 512]
    num_iter = len(hidden_options)
    iterations = np.arange(num_iter)
    trn = np.zeros(num_iter)
    val = np.zeros(num_iter)
    #plt.figure()
    f, axes = plt.subplots(2, 1)
    for iter in iterations:
        hidden = hidden_options[iter]
        print('\nTrying hidden layer size = ', hidden)
        model, history = train_model()
        trn[iter] = np.mean(history.history[key_trn_acc][-num_latest:])
        val[iter] = np.mean(history.history[key_val_acc][-num_latest:])
        axes[0].plot(epochs,history.history[key_trn_loss],
                colours[iter],label=hidden)
        axes[1].plot(epochs,history.history[key_val_loss],
                colours[iter],label=hidden)

    axes[0].legend(loc = 'upper right', ncol=2, title='Hidden neurons')
    axes[0].set_ylabel('Training loss')
    #axes[0].set_xlabel('iterations')
    axes[1].legend(loc = 'upper right', ncol=2, title='Hidden neurons')
    axes[1].set_ylabel('Validation loss')
    axes[1].set_xlabel('Epochs')
    #plt.title('Training loss vs size of hidden layer')
    plt.savefig('./plots/hidden_loss.png')
    plt.show()

    plt.figure()
    plt.semilogx(hidden_options,trn,'b',basex=2)
    plt.semilogx(hidden_options,val,'r',basex=2)
    plt.legend(('Training','Validation'), loc = 'best')
    plt.xlabel('Size of hidden layer')
    plt.ylabel('Accuracy')
    #plt.title('training and validation accuracy vs size of hidden layer')
    plt.savefig('./plots/hidden_acc.png')
    plt.show()
    hidden = int(input("\nChoose the best size of hidden layer: "))
    print_params()

def select_activation_function(activation_options):
    global activation
    #activation_options = ['sigmoid', 'relu', 'tanh', 'softplus']
    num_iter = len(activation_options)
    iterations = np.arange(num_iter)
    trn = np.zeros(num_iter)
    val = np.zeros(num_iter)
    model = [0] *num_iter
    for iter in iterations:
        activation = activation_options[iter]
        print('\nTrying activation function = ', activation)
        model[iter], history = train_model()
        trn[iter] = np.mean(history.history[key_trn_acc][-num_latest:])
        val[iter] = np.mean(history.history[key_val_acc][-num_latest:])
        
        f, axes = plt.subplots(2, 1)
        axes[0].plot(epochs,history.history[key_trn_loss],'b',
                    epochs,history.history[key_val_loss],'r')
        axes[1].plot(epochs,history.history[key_trn_acc],'b',
                    epochs,history.history[key_val_acc],'r')
        axes[0].legend(('Training','Validation'), loc = 'upper right')
        axes[0].set_ylabel('Loss')
        #axes[0].set_xlabel('iterations')
        axes[1].legend(('Training','Validation'), loc = 'lower right')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_xlabel('Epochs')
        plt.suptitle('Activation function: ' + activation)
        plt.savefig('./plots/act_' + activation + '.png')
        plt.show()

    w = 0.4
    plt.figure()
    plt.bar(iterations-w/2,trn, width=w, color='b')
    plt.bar(iterations+w/2,val, width=w, color='r')
    plt.xticks(iterations, activation_options)
    plt.legend(('Training','Validation'), loc = 'best')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=math.floor(np.min(trn)*20)/20)
    #plt.title('training and validation accuracy vs activation function')
    plt.savefig('./plots/act_acc.png')
    plt.show()
    A = int(input('\nChoose the best activation function: '))
    activation = activation_options[A]
    print_params()
    return model[A]
    
'''
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
    plt.imshow(np.reshape(x_test[i], [mdl.input_size, mdl.input_size]), cmap='gray')
    plt.show()

'''

def main():
    x = np.arange(6)
    
    tune_learning_rate((x+1)/2000)
    tune_batch_size([16, 32, 64, 128, 256, 512])
    tune_hidden_layer([16, 32, 64, 128, 256, 512])
    tune_dropout_probability((x+1)/10)
    tune_regularization((x+1)/1000)
    
    final_model = select_activation_function(['relu', 'tanh'])
    final_model.summary()


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--epoch', action='store_true',
    #                    help='Check data loading.')
    main()
