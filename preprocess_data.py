#! /usr/bin/env python

import numpy as np
import os
import data_helpers
from sys import exit

print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(False,"train")
x_test, y_test = data_helpers.load_data_and_labels(False,"test")
print("Total number of samples: {}".format(len(x_text))) 
splitPercentage = 0.1
numberTestSamples = int(splitPercentage*int(len(x_text)))
print("Number of test samples: {}".format(numberTestSamples)) 

# Randomly shuffle data
np.random.seed(10)
x = np.array(x_text)

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

x_train, x_dev = x_shuffled[:-numberTestSamples], x_shuffled[-numberTestSamples:]
y_train, y_dev = y_shuffled[:-numberTestSamples], y_shuffled[-numberTestSamples:]

vocabulary = data_helpers.create_vocabulary(x_train.tostring().split())

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


vocabulary = data_helpers.create_vocabulary(x_train.tostring().split())
x_train = data_helpers.substitute_oov(x_train,vocabulary)
x_dev = data_helpers.substitute_oov(x_dev,vocabulary)

x_test = data_helpers.substitute_oov(np.array(x_test),vocabulary) 
print x_test[0]


with open('train_set.txt', 'w') as thefile:
    for item in x_train:
        thefile.write("%s\n" % item)

with open('dev_set.txt', 'w') as thefile:
    for item in x_dev:
        thefile.write("%s\n" % item)

with open('test_set.txt', 'w') as thefile:
    for item in x_test:
        thefile.write("%s\n" % item)

