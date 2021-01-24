import numpy as np
import h5py
import os


# load MNIST data
MNIST_data = h5py.File(os.getcwd()+"/CW/mnist.hdf5", 'r')
# print((MNIST_data['t_valid']))
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['t_train'])).reshape(-1, 1)
x_test  = np.float32(MNIST_data['x_test'][:])
y_test  = np.int32(np.array(MNIST_data['t_test'])).reshape(-1, 1)
MNIST_data.close()


# stack together for next step
X = np.vstack((x_train, x_test))
y = np.vstack((y_train, y_test))


# one-hot encoding
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

# number of training set
m = 60000
m_test = X.shape[0] - m
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]

# shuffle training set
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

def sigmoid(z):
    """
    sigmoid activation function.

    inputs: z
    outputs: sigmoid(z)
    """
    s = 1. / (1. + np.exp(-z))
    return s

def compute_loss(Y, Y_hat):
    """
    compute loss function
    """
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L

def feed_forward(X, params):
    """
    feed forward network: 2 - layer neural net

    inputs:
        params: dictionay a dictionary contains all the weights and biases

    return:
        cache: dictionay a dictionary contains all the fully connected units and activations
    """
    cache = {}

    # Z1 = W1.dot(x) + b1
    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]

    # A1 = sigmoid(Z1)
    cache["A1"] = sigmoid(cache["Z1"])

    # Z2 = W2.dot(A1) + b2
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]

    # A2 = softmax(Z2)
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

    return cache

def back_propagate(X, Y, params, cache, m_batch):
    """
    back propagation

    inputs:
        params: dictionay a dictionary contains all the weights and biases
        cache: dictionay a dictionary contains all the fully connected units and activations

    return:
        grads: dictionay a dictionary contains the gradients of corresponding weights and biases
    """
    # error at last layer
    dZ2 = cache["A2"] - Y

    # gradients at last layer (Py2 need 1. to transform to float)
    dW2 = (1. / m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    # back propgate through first layer
    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))

    # gradients at first layer (Py2 need 1. to transform to float)
    dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads

import argparse

parser = argparse.ArgumentParser()

# hyperparameters setting
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--n_x', type=int, default=784, help='number of inputs')
parser.add_argument('--n_h', type=int, default=64,
                    help='number of hidden units')
parser.add_argument('--beta', type=float, default=0.9,
                    help='parameter for momentum')
parser.add_argument('--batch_size', type=int,
                    default=64, help='input batch size')

# parse the arguments
opt = parser.parse_args()

# training
def train(batches, params):
    for i in range(opt.epochs):

        # shuffle training set
        permutation = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = Y_train[:, permutation]

        for j in range(batches):

            # get mini-batch
            begin = j * opt.batch_size
            end = min(begin + opt.batch_size, X_train.shape[1] - 1)
            X = X_train_shuffled[:, begin:end]
            Y = Y_train_shuffled[:, begin:end]
            m_batch = end - begin

            # forward and backward
            cache = feed_forward(X, params)
            grads = back_propagate(X, Y, params, cache, m_batch)

            # with momentum (optional)
            dW1 = (opt.beta * dW1 + (1. - opt.beta) * grads["dW1"])
            db1 = (opt.beta * db1 + (1. - opt.beta) * grads["db1"])
            dW2 = (opt.beta * dW2 + (1. - opt.beta) * grads["dW2"])
            db2 = (opt.beta * db2 + (1. - opt.beta) * grads["db2"])

            # gradient descent
            params["W1"] = params["W1"] - opt.lr * dW1
            params["b1"] = params["b1"] - opt.lr * db1
            params["W2"] = params["W2"] - opt.lr * dW2
            params["b2"] = params["b2"] - opt.lr * db2

        # forward pass on training set
        cache = feed_forward(X_train, params)
        train_loss = compute_loss(Y_train, cache["A2"])

        # forward pass on test set
        cache = feed_forward(X_test, params)
        test_loss = compute_loss(Y_test, cache["A2"])
        print("Epoch {}: training loss = {}, test loss = {}".format(
            i + 1, train_loss, test_loss))

train()