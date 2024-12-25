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
    print('数据特征：', ', '.join(header[:-1]))
    print('数据标签：', header[-1])
    print('数据总条数：', len(lines))
    print('-------------------')

    # get training set and test set.
    ratio = 0.8
    split = int(len(lines) * ratio)
    np.random.seed(0)
    lines = np.random.permutation(lines)  # shuffle lines
    train, test = lines[:split], lines[split:]

    # A raw sample
    print('----- This is an raw sample. -----')
    print(train[0])

    return train, test

# The function is used to simulate feature extraction.
# From the process, raw features can be transformed to a more
# effective set of inputs.
# In real-time rs, label is not necessary to do the feature extraction.
def feature_extraction(train, test):
    # It's a process of feature extraction.
    # Transform raw data in a more effective set of inputs.
    #
    # Standardization.(Make data conform to normal distribution)
    # This technique to rescale features value with the distribution value between 0 and 1 is useful for the optimization algorithms,
    # such as gradient descent, that are used within machine-learning algorithms that weight inputs
    scaler = StandardScaler()
    scaler.fit(train)  # 只使用训练集的数据计算均值和方差
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

    return x_train, y_train, x_test, y_test

# The function is used to simulate training process of model based on training data of X and Y.
def train_model(x_train, y_train):
    ## Add const coeffcient
    X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)


def total_offline_training_process():
    train, test = load_samples()
    train, test = feature_extraction(train, test)
    x_train, y_train, x_test, y_test = split_total_sample(train, test)
    train_model(x_train, y_train)

total_offline_training_process()