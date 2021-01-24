from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time
import os
import csv
import pickle as pck
from matplotlib import pyplot as plt
import copy
from collections import Counter
from collections import defaultdict as ddict
# import ddict
cwd=os.getcwd()

def discretize(x):
    return np.where(x > 0.5, 1, 0)

x, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='~/Documents/IPB/CW')
x = (x/255).astype('float32')
y = to_categorical(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1/7, random_state=42)

def generate_batch(dataset, batch_size):
    #differentiate inputs (features) from targets and transform each into 
    #numpy array with each row as an example
    inputs = dataset[0]
    targets = dataset[1]
    #randomly choose batch_size many examples; note there will be
    #duplicate entries when batch_size > len(dataset) 
    rand_inds = np.random.randint(0,len(inputs), batch_size)
    inputs_batch = [inputs[i] for i in rand_inds]
    targets_batch = [targets[i] for i in rand_inds]
    
    return inputs_batch, targets_batch

class DeepNeuralNetwork():
    def __init__(self, sizes, epochs, lr=0.05):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr
        self.initialization()

    def sigmoid(self, x):
        sigmoid = (1/(1 + np.exp(-x)))
        return sigmoid
    def sigmoid_prime(self, a):
        dsigmoid_da = self.sigmoid(a)*(1-self.sigmoid(a))
        return dsigmoid_da
    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)
    def softmax_prime(self, x):
        exps = np.exp(x - x.max())
        # print(exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0)))
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    def initialization(self):
        # number of nodes in each layer
        input_size=self.sizes[0]
        hidden_size=self.sizes[1]
        output_size=self.sizes[2]

        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(1/hidden_size),
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(1/output_size)
        self.B = np.random.randn(output_size, hidden_size) * np.sqrt(1/output_size)

    def forward_pass(self, x_train):

        # Input layer is batch sample of training input
        U = x_train
        # input layer to hidden layer 1 + activation
        H_1 = np.dot(self.W1, U)
        # print(np.shape(H_1[0]))
        Z = self.sigmoid(H_1[0])
        # hidden layer to output layer + activation
        H_2 = np.dot(self.W2, Z)
        V = self.softmax(H_2)

        return U, Z, H_1, H_2, V

    def backward_pass(self, y_train, dL_dv, U, Z, H_1, H_2, V):

        # W2 Update
        dL_dH2 = 2*dL_dv/V.shape[0]  * self.softmax_prime(H_2)
        dL_dW2 = np.outer(dL_dH2, H_1)
        # dL_dW2 = np.matmul(, dL_dH2)
        # W1 update
        dL_dZ = np.dot((self.W2).T, dL_dH2)
        dL_dH1 = dL_dZ * self.sigmoid_prime(H_1)
        dL_dW1 = np.outer(dL_dH1, U)

        return dL_dW1, dL_dW2
 
    def update_network_parameters(self, dL_dW1, dL_dW2):

        W1_old = copy.deepcopy(self.W1[0])
        W2_old = copy.deepcopy(self.W2[0])
        self.W1 -= self.lr * dL_dW1
        self.W2 -= self.lr * dL_dW2
        W1_new = copy.deepcopy(self.W1[0])
        W2_new = copy.deepcopy(self.W2[0])


        W1_deltas = np.sum(np.abs(np.subtract(np.ndarray.flatten(W1_old), np.ndarray.flatten(W1_new))))
        W2_deltas = np.sum(np.abs(np.subtract(np.ndarray.flatten(W2_old), np.ndarray.flatten(W2_new))))
        total_deltas = W2_deltas + W1_deltas
        average_iter_delta = total_deltas/len(np.ndarray.flatten(W1_old))

        return average_iter_delta

    def compute_error(self, x_val, y_val, iteration):

        predictions = []
        
        all_zs = []
        for x, y in zip(x_val, y_val):
            # print(y)
            _,Z,_,_, V = self.forward_pass(x)
            Z = np.round(Z,0)
            all_zs.append([Z,y])
            pred = np.argmax(V)
            predictions.append(pred == np.argmax(y))

        return 1 - np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val, batch_size):
        entropy = []
        mut_inf = []
        tic = time.time()
        tocs = []
        losses = []
        deltas = []
        iterations = []
        hyx=0
        # averages = []
        # all_zs = []
        for iteration in range(self.epochs):
            iter_delta = 0
            x_batch, y_batch = generate_batch([x_train, y_train], batch_size)
            i=0
            for x,y in zip(x_batch, y_batch):
                i +=1
                U, Z, H_1, H_2, V = self.forward_pass(x)
                # all_zs.append(Z)
                dL_dPred = V-y
                dL_dW1, dL_dW2 = self.backward_pass(y, dL_dPred, U, Z, H_1, H_2, V)
                iter_delta += self.update_network_parameters(dL_dW1, dL_dW2)


            average_batch_delta = iter_delta/batch_size
            # print('average', average_batch_delta)
            error = self.compute_error(x_val, y_val, iteration)
            toc = time.time() - tic
            if iteration%20==19:
                print('Epoch: {0}, Time Spent: {1:.2f}s, Error: {2:.5f}'.format(
                
                    iteration+1, toc, error
                ))
            losses.append(error)
            tocs.append(iteration)
            deltas.append(average_batch_delta)

        with open('losses.pkl','wb') as file:
            # writer = csv.writer(file, delimiter = ',')
            pck.dump([tocs, losses], file)
            file.close()
        with open('deltas.pkl','wb') as file:
            # writer = csv.writer(file, delimiter = ',')
            pck.dump([tocs, deltas], file)
            file.close()
        return [tocs, losses]



dnn = DeepNeuralNetwork(sizes=[784, 32, 10], epochs=1000)
dnn.train(x_train, y_train, x_val, y_val, batch_size=1000)

