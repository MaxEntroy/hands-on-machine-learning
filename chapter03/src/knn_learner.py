import matplotlib.pyplot as plt
import numpy as np
import os

def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def show_a_sample(sample):
    data = np.reshape(np.array(sample, dtype=int), [28, 28])
    plt.figure()
    plt.imshow(data, cmap='Blues')
    plt.show()

def load_samples_and_labels():
    # load raw samples and labels
    samples = np.loadtxt('mnist_x.txt', delimiter=' ')
    labels = np.loadtxt('mnist_y.txt')
    ## show_a_sample(samples[1])

    # shuffle raw samples and labels
    np.random.seed(0)
    shuffle_indices = np.random.permutation(np.arange(len(samples)))
    ## print (shuffle_indices)
    samples = samples[shuffle_indices]
    labels = labels[shuffle_indices]

    # make training_set and test_set
    ratio = 0.8
    split = int(len(samples) * ratio)
    samples_train_set, samples_test_set = samples[:split], samples[split:]
    labels_train_set, labels_test_set = labels[:split], labels[split:]
    return samples_train_set, samples_test_set, labels_train_set, labels_test_set

class KNN:
    ## k and label_num are prior knowldedge
    def __init__(self, k, label_num):
        self.k = k
        self.label_num = label_num

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    ## For a sample x, get knn samples.
    def get_knn_labels(self, x):
      ## Step1: Calculate the distance between x and all the samples.
      dis = list(map(lambda ele : distance(ele, x), self.x_train))

      ## Step2: Sort the samples and get topK.
      sorted_dis = np.argsort(dis)
      knn = sorted_dis[:self.k]

      return knn

    ## For a sample, predict its label.
    def predict(self, x):
        knn = self.get_knn_labels(x)
        label_counter = np.zeros(shape = [self.label_num], dtype = int)
        for idx in knn:
          label = int(self.y_train[idx])
          label_counter[label] += 1
        return np.argmax(label_counter)

    ## For batch samples, predict its label.
    def batch_predict(self, x_test):
      predicted_test_labels = [self.predict(x) for x in x_test]
      return predicted_test_labels

def main():
  x_train, x_test, y_train, y_test = load_samples_and_labels()
  for k in range(1, 10):
    knn = KNN(k, 10)
    knn.fit(x_train, y_train)

    predicted_y = knn.batch_predict(x_test)

    accuray = np.mean(predicted_y == y_test)
    print(k, accuray)

main()
