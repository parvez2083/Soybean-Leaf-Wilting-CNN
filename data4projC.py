#from __future__ import print_function
import numpy as np
import csv
import cv2
#import matplotlib.image as mpimg

# Original sizes for ECE542 ProjC
(orig_rows, orig_cols) = (480, 640)
# Dimensions for training dataset
# Count of training data; number of rows, colomns and channels in each image
# 'channels' is 3 for RGB, 1 for Grayscale
# Change these as needed
(count, rows, cols, channels) = (889, 120, 160, 1)

keep_orig_size = ((rows is orig_rows) and (cols is orig_cols))
num_class = 5

def dense_to_one_hot(labels_dense, num_classes=10):
  #Convert class labels from scalars to one-hot vectors.
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def read_data_sets(file_dir, val_size=100, one_hot=False):
    # Read a list of data files and labels from a CSV file
    # then read those image data and preprocess/split them
    global num_class, count
    with open(file_dir+'train_files.csv') as csvfile:
        
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV)   #ignore the header row
        
        #Pre-initialize arrays to avoid ddynamic allocation
        if (channels is 1) and (not keep_orig_size):
            img = np.empty((orig_rows, orig_cols), dtype='uint8')
        train_images = np.empty((count, rows, cols, channels), dtype='float32')
        train_labels = np.empty(count, dtype='uint8')
        
        i = 0
        for row in readCSV:
            file_path = file_dir+'Training/'+row[0]
            if channels is 1:       # read as Grayscale
                if keep_orig_size:
                    train_images[i] = np.expand_dims(
                                            cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), 
                                        axis=2)
                else:
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    train_images[i] = np.expand_dims(
                                            cv2.resize(img, dsize=(cols, rows), 
                                            interpolation=cv2.INTER_CUBIC), 
                                        axis=2)
            else:               # read as RGB image
                if keep_orig_size:
                    train_images[i] = cv2.imread(file_path)
                else:
                    train_images[i] = cv2.resize(cv2.imread(file_path), dsize=(cols, rows), 
                                                interpolation=cv2.INTER_CUBIC)
            
            # this snippet is for visualization
            #cv2.imshow('image',train_images[i].astype('uint8'))
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #return
            
            # Mean Normalization, optinal, comment if not needed
            train_images[i] = (train_images[i] - np.mean(train_images[i]))/np.std(train_images[i])
            
            # Reading the true labels
            train_labels[i] = row[1]
            i = i+1
            if (i == count):
                break
    
    # Updating the variables from the read data
    num_class = np.max(train_labels)+1
    #count = train_labels.shape[0]
    
    # One-hot encoding for the labels, if needed
    if one_hot:
        train_labels = dense_to_one_hot(train_labels, num_classes=num_class)
    
    #Splitting the dataset for training and validation
    validation_images = train_images[:val_size]
    validation_labels = train_labels[:val_size]
    train_images = train_images[val_size:]
    train_labels = train_labels[val_size:]
    return (train_images, train_labels), (validation_images, validation_labels)




#read_data_sets('../')


