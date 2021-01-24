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
plt.style.use('seaborn')

cwd=os.getcwd()


x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
y = to_categorical(y)
# for j in y:
#     new_y = []
#     new_y.append(np.array([int(i == j) for i in range(10)]))
#     # print(new_y_train)
# y = new_y

# print(len(x[0]))
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)
# print(len(y_val))
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


inputs, targets = generate_batch([x_train, y_train], batch_size=100)


def sigmoid(a):
    siga = 1/(1+np.exp(-a))
    return siga
    
def loss_mse(preds, targets):
    loss = np.sum(np.square(np.subtract(preds, targets)))/2
#     print('loss', loss)
    return loss

#derivative of loss function with respect to predictions
def loss_deriv(preds, targets):
    dL_dPred = preds - targets
    return dL_dPred


class nn_one_layer():
    def __init__(self, input_size, hidden_size, output_size):
        #define the input/output weights W1, W2
        self.W1 = 0.1 * np.random.randn(input_size, hidden_size)
        # print(self.W1)
        self.W2 = 0.1 * np.random.randn(hidden_size, output_size)
        # print(self.W2)
        self.f = sigmoid
        
    #for matrix multiplication use np.matmul()
    def forward(self, u):
        z = np.matmul(u, self.W1)
        h = self.f(z)
        v = np.matmul(h, self.W2)
        # v = loss_softmax(h_2)

        return v, h, z
        
input_size = 784    
hidden_size = 128
output_size = 10

nn = nn_one_layer(input_size, hidden_size, output_size) #initialise model
preds, _, _,_ = nn.forward(inputs) #prediction made by model on batch xor input

_, inds = np.unique(inputs, return_index=True, axis=0)


# # plot target vs predictions along with a histogram of each
# fig = plt.figure()
# ax1 = plt.subplot2grid((2,2), (0,0), rowspan=1, colspan=2)
# plt.scatter(targets[inds], preds[inds], marker='x', c='black')
# for i in inds:
#     coord = '({}, {})'.format(inputs[i][0], inputs[i][1])
#     xoffset = 0.05 if targets[i] == 0 else -0.1
#     yoffset = 0.003 if preds[i] > np.mean(preds[inds]) else -0.005
#     plt.text(targets[i] + xoffset, preds[i] + yoffset, coord)
# plt.xlabel('target values')
# plt.ylabel('predicted values')
# plt.ylim([np.min(preds) - 0.01, np.max(preds) + 0.01])
# ax2 = plt.subplot2grid((2,2), (1,0), rowspan=1, colspan=1)
# plt.hist(targets, color='blue')
# ax2.set_title('target values')
# plt.ylabel('# in batch')
# ax3 = plt.subplot2grid((2,2), (1,1), rowspan=1, colspan=1, sharey=ax2)
# plt.hist(preds, color='red')
# ax3.set_title('predicted values')

# fig.tight_layout()

#loss function as defined above


#derivative of the sigmoid function
#for element-wise multiplication use A*B or np.multiply(A,B)
def sigmoid_prime(a):
    dsigmoid_da = sigmoid(a)*(1-sigmoid(a))
    return dsigmoid_da

#compute the derivative of the loss wrt network weights W1 and W2
#dL_dPred is (precomputed) derivative of loss wrt network prediction
#X is (batch) input to network, H is (batch) activity at hidden layer

def backprop(W1, W2, dL_dPred, U, H, Z):
    #hints: for dL_dW1 compute dL_dH, dL_dZ first.
    #for transpose of numpy array A use A.T

    dL_dW2 = np.matmul(H.T, dL_dPred)
    dL_dH = np.matmul(dL_dPred, W2.T)
    dL_dZ = np.multiply(sigmoid_prime(Z), dL_dH)
    dL_dW1 = np.matmul(U.T, dL_dZ)
    # error = H.T * dL_dPred
    # dL_dW2 = np.outer(error, Z)
    # error = np.dot((W2).T, error) * sigmoid_prime(Z)
    # dL_dW1 = np.outer(error, U)

    return dL_dW1, dL_dW2

#train the provided network with one batch according to the dataset
#return the loss for the batch
def train_one_batch(nn, dataset, batch_size, lr):
    inputs, targets = generate_batch(dataset, batch_size)
    preds, H, Z = nn.forward(inputs)

    # loss = loss_mse(preds, targets)

    dL_dPred = loss_deriv(preds, targets)
    dL_dW1, dL_dW2 = backprop(nn.W1, nn.W2, dL_dPred, U=inputs, H=H, Z=Z)

    nn.W1 -= lr * dL_dW1
    nn.W2 -= lr * dL_dW2
    
    return loss

#test the network on a given dataset
def test(nn, dataset):
    # inputs, targets = generate_batch([x_val, y_val], batch_size=200)
    # preds, H, Z = nn.forward(inputs) 
    # loss = loss_softmax(preds, targets)
    # return loss
    predictions = []
    for x, y in zip(x_val, y_val):
        V,_,_,_ = nn.forward(x)
        pred = np.argmax(V)
        predictions.append(pred == np.argmax(y))
    # print(predictions)
    return 1 - np.mean(predictions)

chosen_dataset = [x_train, y_train]

batch_size = 70000 #number of examples per batch
nbatches = 5 #number of batches used for training
lr = 0.1 #learning rate

losses = [] #training losses to record
for i in range(nbatches):
    loss = train_one_batch(nn, chosen_dataset, batch_size=batch_size, lr=lr)
    print(test())
    losses.append(loss)

# plt.plot(np.arange(1, nbatches+1), losses)
# plt.xlabel("# batches")
# plt.ylabel("training MSE")

# inputs, targets = generate_batch(chosen_dataset, batch_size=100)
# preds, _, _ = nn.forward(inputs) #prediction made by model

# _, inds = np.unique(inputs, return_index=True, axis=0)

# # plot target vs predictions along with a histogram of each
# fig = plt.figure()
# ax1 = plt.subplot2grid((2,2), (0,0), rowspan=1, colspan=2)
# plt.scatter(targets[inds], preds[inds], marker='x', c='black')

# yup = 0.1
# ydown = -0.1
# for i in inds:
#     coord = '({}, {})'.format(inputs[i][0], inputs[i][1])
#     if np.isclose(preds[i], 0, atol=0.1):
#         yup = 2 * yup
#         yoffset = yup
#     else:
#         ydown = 2 * ydown
#         yoffset = ydown
    
#     xoffset = 0.05 if targets[i] == 0 else -0.1
#     plt.text(targets[i] + xoffset, preds[i] + yoffset, coord)
# plt.xlabel('target values')
# plt.ylabel('predicted values')
# plt.ylim([np.min(preds) - 0.1, np.max(preds) + 0.1])
# ax2 = plt.subplot2grid((2,2), (1,0), rowspan=1, colspan=1)
# plt.hist(targets, color='blue')
# ax2.set_title('target values')
# plt.ylabel('# in batch')
# ax3 = plt.subplot2grid((2,2), (1,1), rowspan=1, colspan=1, sharey=ax2)
# plt.hist(preds, color='red')
# ax3.set_title('predicted values')

# fig.tight_layout()

# dataset_names = ['AND gate', 'OR gate', 'XOR gate', 'XNOR gate']
# test_scores = [test(nn, dataset) for dataset in [dataset_and, 
#                             dataset_or, dataset_xor, dataset_xnor]]

# x = range(4)
# plt.bar(x, test_scores)
# plt.xticks(x, dataset_names, rotation='vertical')
# plt.ylabel("test MSE")
