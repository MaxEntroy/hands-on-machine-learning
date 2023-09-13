import matplotlib.pyplot as plt
import numpy as np
import os

def load_samples_and_labels():
    # load raw samples and labels
    samples = np.loadtxt('mnist_x.txt', delimiter=' ')
    labels = np.loadtxt('mnist_y.txt')

    # shuffle raw samples and labels
    np.random.seed(0)
    shuffle_indices = np.random.permutation(np.arange(len(samples)))
    samples = samples[shuffle_indices]
    labels = labels[shuffle_indices]

    # make training_set and test_set
    ratio = 0.8
    split = int(len(samples) * ratio)
    samples_train_set, samples_test_set = samples[:split], samples[split:]
    labels_train_set, labels_test_set = labels[:split], labels[split:]
    return samples_train_set, samples_test_set, labels_train_set, labels_test_set

def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

