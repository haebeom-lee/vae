from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def mnist_1000(MNIST_PATH):
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True, validation_size=0)
    x, y = mnist.train.images, mnist.train.labels
    y_ = np.argmax(y, axis=1)

    xtr = [x[y_==k][:100,:] for k in range(10)]
    ytr = [y[y_==k][:100,:] for k in range(10)]
    xtr, ytr = np.concatenate(xtr, 0), np.concatenate(ytr, 0)

    xte = [x[y_==k][100:200,:] for k in range(10)]
    yte = [y[y_==k][100:200,:] for k in range(10)]
    xte, yte = np.concatenate(xte, 0), np.concatenate(yte, 0)

    return xtr, ytr, xte, yte
