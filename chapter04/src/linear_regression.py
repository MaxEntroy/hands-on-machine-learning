#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler

# The function is used to simulate load samples.
# From the process, we can get the raw samples.
def load_samples():
    lines = np.loadtxt('USA_Housing.csv', delimiter=',', dtype='str')
    header = lines[0]
    lines = lines[1:].astype(float)
    print('raw sample features：', '｜'.join(header[:-1]))
    print('raw sample label：', header[-1])
    print('number of total raw samples：', len(lines))

    # get training set and test set.
    ratio = 0.8
    split = int(len(lines) * ratio)

    # shuffle raw samples
    np.random.seed(0)
    lines = np.random.permutation(lines)

    train, test = lines[:split], lines[split:]

    # A raw sample
    print('----- This is an raw sample. -----')
    print(train[0])

    return train, test

# The function is used to simulate feature extraction.
# From the process, raw features can be transformed to a more
# effective set of inputs.
# In real-time rs, label is not necessary for feature extraction.
def feature_extraction(train, test):
    # It's a process of feature extraction.
    # Transform raw data in a more effective set of inputs.
    #
    # Standardization.(Make data conform to normal distribution)
    # This technique to rescale features value with the distribution value between 0 and 1 is useful for the optimization algorithms,
    # such as gradient descent, that are used within machine-learning algorithms that weight inputs
    scaler = StandardScaler()
    scaler.fit(train)  #  mean and variance are calculated only by training set.
    train = scaler.transform(train)
    test = scaler.transform(test)

    # A result sample
    print('----- This is an result sample. -----')
    print(train[0])

    return train, test

# The function is used to simulate splitting input features and labels
# from the total samples.
def split_total_sample(train, test):
    # Get feature and label.
    x_train, y_train = train[:, :-1], train[:, -1].flatten()
    x_test, y_test = test[:, :-1], test[:, -1].flatten()

    print('----- Features -----')
    print(x_train)
    print('----- Labels -----')
    print(y_train)
    print('------------------')

    return x_train, y_train, x_test, y_test

# The function is used to simulate training process of model based on training data of X and Y.
def train_model(x_train, y_train):
    ## Add const coeffcient
    X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)
    theta = np.linalg.inv(X.T @ X) @ X.T @ y_train
    print('--------- Final theta ---------')
    print(theta)
    return theta

# After training process is finished, a new model is updated.
# The new model's effectiveness should be evaluated.
# Apply the theta to test set.
def effectiveness_evaluation(theta, x_test, y_test):
    X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)
    y_pred = X_test @ theta

    rsme_loss = np.sqrt( np.square(y_test  - y_pred).mean() )
    print('RSME:', rsme_loss)

def total_offline_training_process():
    train, test = load_samples()
    train, test = feature_extraction(train, test)
    x_train, y_train, x_test, y_test = split_total_sample(train, test)
    theta = train_model(x_train, y_train)
    effectiveness_evaluation(theta, x_test, y_test)

total_offline_training_process()