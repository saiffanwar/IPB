#!/usr/bin/env python
# coding: utf-8

from scipy import interpolate

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
# utils
def mse_loss(preds, targets):
    return np.sum((preds - targets)**2) / 2

def deriv_mse_loss(preds, targets):
    return preds - targets

def deriv_sigmoid(x):
    sigx = sigmoid(x)
    return sigx * (1-sigx)

def sigmoid(x):
    return 1/(1+np.exp(-x))
B = np.random.randn(32, 10)
class BatchGenerator(object):
    def __init__(self, data, labels, batch_size):
        self.n = data.shape[0]
        self.batch_size = batch_size
        self.data = data
        self.labels = labels

    def __iter__(self):
        return self

    def __next__(self):
        idx = np.random.choice(np.arange(self.n), self.batch_size)
        return self.data[idx], self.labels[idx]

class SimpleNeuralNetwork():
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        #randomly init connection weights with mean of 0.1
        self.w1 = 0.1*np.random.randn(input_size, hidden_size) # input to hidden layer
        self.w2 = 0.1*np.random.randn(hidden_size, output_size) # hidden to output layer)

    def forward(self, batch):
        hidden_values = batch.dot(self.w1)
        hidden_activations = sigmoid(hidden_values)
        output = hidden_activations.dot(self.w2)
        return output, hidden_values, hidden_activations

class SimpleBackpropTrainer():
    def __init__(self, nn, learn_rate, train_batch_generator, test_batch_generator):
        self.nn = nn
        self.learn_rate = learn_rate
        self.train_batch_generator = train_batch_generator
        self.test_batch_generator = test_batch_generator

    def backprop(self, W2, dL_dV, U, H, Z, use_deriv):
        dL_dW2 = np.matmul(H.T, dL_dV)
        dL_dH = np.matmul(dL_dV, W2.T)
        if use_deriv:
            dL_dZ = np.multiply(deriv_sigmoid(Z), dL_dH)
        else:
            dL_dZ = dL_dH
        dL_dW1 = np.matmul(U.T, dL_dZ)
        return dL_dW1, dL_dW2, dL_dV, dL_dH, dL_dZ

    def train(self, epochs, use_weight=True, use_deriv=True, prob_not_backprop=0):
        
        epoch = 0
        train_losses = []
        test_losses = []
        derivs = []
        test_accuracy= []
        # generate random training batch
        for batch, targets in self.train_batch_generator:

            # feed-forward to get predictions and hidden states
            predictions, hidden_values, hidden_activations = self.nn.forward(batch)

            # calculate loss
            loss = mse_loss(predictions, targets)

            if epoch==0 or np.random.uniform()>prob_not_backprop:
                # calculate derivative of loss with respect to predictions
                deriv_loss = deriv_mse_loss(predictions, targets)

                # backprop to calculate derivative of loss with respect to weights
                dL_dW1, dL_dW2, dL_dV, dL_dH, dL_dZ = self.backprop(W2=self.nn.w2 if use_weight else B, dL_dV=deriv_loss, U=batch, H=hidden_activations, Z=hidden_values, use_deriv=use_deriv)
                
                # gradient descent to update weights
                self.nn.w1 -= self.learn_rate * dL_dW1
                self.nn.w2 -= self.learn_rate * dL_dW2

                # log batch loss
            train_losses.append(loss)

            derivs.append((np.linalg.norm(dL_dW1), np.linalg.norm(dL_dW2), np.linalg.norm(dL_dV), np.linalg.norm(dL_dH), np.linalg.norm(dL_dZ)))


            # log losses and validate on test set every so often
            if epoch % 5 == 0:
                T = int(self.test_batch_generator.n/self.test_batch_generator.batch_size)
                t = 0
                test_loss_ = []
                acc = 0
                n=0
                for test, target in self.test_batch_generator:
                    prediction, _, _ = self.nn.forward(test)
                    test_loss_.append(mse_loss(prediction, target))
                    t+=1
                    if t==T:
                        test_losses.append(np.mean(test_loss_))
                        break

            # log losses and validate on test set every so often
            if epoch % 50 == 0:
                T = int( self.test_batch_generator.n/self.test_batch_generator.batch_size)
                t = 0
                test_loss_ = []
                acc = 0
                n=0
                for test, target in self.test_batch_generator:
                    prediction, _, _ = self.nn.forward(test)
                    prediction = np.argmax(prediction, axis=1)
                    target = np.argmax(target, axis=1)
                    acc += np.sum(prediction == target)
                    n+= len(target)
                    t+=1
                    if t==T:
                        test_accuracy.append(acc/n)
                        break                    
            # count epochs
            epoch+=1
            if epoch+1 > epochs:
                break
        return self.nn, train_losses, test_losses, derivs, test_accuracy

def plot_derivs(derivs):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot([a for a in range(epochs)], derivs[:, 1], 'b', label=r"$\frac{\partial L}{\partial W_{2}}$")
    ax1.plot([a for a in range(epochs)], derivs[:, 0], 'r', label=r"$\frac{\partial L}{\partial W_{1}}$", alpha=0.8)

    ax2.plot([a for a in range(epochs)], derivs[:, 2], 'g', label=r"$\frac{\partial L}{\partial {V}'}$", alpha=0.8)
    ax2.plot([a for a in range(epochs)], derivs[:, 4], 'y', label=r"$\frac{\partial L}{\partial Z}$", alpha=0.8)
    ax2.plot([a for a in range(epochs)], derivs[:, 3], 'k', label=r"$\frac{\partial L}{\partial H}$")

    for ax in [ax1, ax2]:
        ax.legend(loc="upper right", fontsize=20, ncol=3)
    # ax1.set_ylim([-0.06,0.06])
    # ax2.set_ylim([-0.03,0.03])
    ax1.set_xlabel('Training iterations')
    ax1.set_ylabel('Mean')

    plt.show()

def load_data(file):
    data = np.loadtxt(file, delimiter=",")
    imgs = np.asfarray(data[:, 1:]) * (0.99/255) + 0.01
    labels = np.asfarray(data[:, :1])
    classes = np.arange(10)
    labels_one_hot = (classes==labels).astype(np.float)
    labels_one_hot[labels_one_hot==0]=0.01
    labels_one_hot[labels_one_hot==1]=0.9
    return imgs, labels_one_hot

def plot_losses(train_losses, test_losses, test_accuracy):
    fig, ax = plt.subplots()
    ax.plot([a for a in range(epochs)], train_losses, label="Training")
    ax.plot([5*a for a in range(int(epochs/5))], test_losses, label="Testing", linewidth=2.5, alpha=0.8)
    ax.set_xlabel('Training iterations')
    ax.set_ylabel('Loss')
    ax2 = ax.twinx() 
    ax2.set_ylabel('Accuracy')  # we already handled the x-label with ax1
    ax2.plot([50*a for a in range(int(epochs/50))], test_accuracy, 'm')
    plt.show()



train_imgs, train_labels = load_data("mnist_train.csv")
test_imgs, test_labels = load_data("mnist_test.csv")
train_generator = BatchGenerator(train_imgs, train_labels, 48)
test_generator = BatchGenerator(test_imgs, test_labels, 48)
epochs=10000

from collections import defaultdict as ddict

def discretize(x):
    return np.where(x > 0.5, 1, 0)

from pprint import pprint

def collect_data(epochs=10000, hidden_size=32):
    input_size = 28*28
    output_size = 10
    hidden_size = hidden_size

    W1 = 0.1*np.random.randn(input_size, hidden_size)
    W2 = 0.1*np.random.randn(hidden_size, output_size)
    
    epoch = 0
    HYX = []
    IYX = []
    HZX = []
    IZX = []
    for U, V in train_generator:

        Z = U.dot(W1)
        H = sigmoid(Z)
        V_ = H.dot(W2)
        # calculate derivative of loss with respect to predictions
        dL_dV = deriv_mse_loss(V_, V)
        if epoch % 100 == 99:
            px = ddict(int)
            py = ddict(int)
            pz = ddict(int)
            pxy = ddict(lambda: ddict(int))
            pxz = ddict(lambda: ddict(int))            
            n = 0
            
            Z_ = np.zeros(V_.shape)
            Z_[:, np.argmax(V_,axis=1)]=1

            for x, y, z in zip(discretize(V), discretize(H), Z_):
                px[str(x)]+=1
                py[str(y)]+=1
                pz[str(z)]+=1
                pxy[str(x)][str(y)]+=1
                pxz[str(x)][str(z)]+=1
                n+=1

            for x in px.keys():
                px[x] /= n
            for y in py.keys():
                py[y] /= n
            for z in pz.keys():
                pz[z] /= n

            hyx = 0
            iyx = 0
            hzx = 0
            izx = 0

            for x in pxy.keys():
                for y in pxy[x].keys():
                    pxy[x][y] /= n
                    pxz[x][z] /= n
                    hyx += pxy[x][y] * np.log2(pxy[x][y]/px[x])
                    iyx += pxy[x][y] * np.log2(pxy[x][y]/(px[x]*py[y]))
                    hzx += pxz[x][z] * np.log2(pxz[x][z]/px[x])
                    izx += pxz[x][z] * np.log2(pxz[x][z]/(px[x]*pz[z]))
            hyx = -hyx
            hzx = -hzx

            HYX.append(hyx)
            IYX.append(iyx)
            HZX.append(hzx)
            IZX.append(izx)

        dL_dW2 = np.matmul(H.T, dL_dV)
        dL_dH = np.matmul(dL_dV, W2.T)
        dL_dZ = np.multiply(deriv_sigmoid(Z), dL_dH)
        dL_dW1 = np.matmul(U.T, dL_dZ)
        
        # gradient descent to update weights
        W1 -= 0.01 * dL_dW1
        W2 -= 0.01 * dL_dW2

        if epoch+1 == epochs:
            break
        epoch+=1

    return HYX, IYX, HZX, IZX
# print('running')
HYXs = []
IYXs = []
HZXs = []
IZXs = []
for size in [16, 32, 64]:
    HYXs_ = []
    IYXs_ = []
    HZXs_ = []
    IZXs_ = []
    for i in range(50):
        hyx, iyx, hzx, izx= collect_data(hidden_size=size)
        HYXs_.append(hyx)
        IYXs_.append(iyx)
        HZXs_.append(hzx)
        IZXs_.append(izx)
        print(i)
    HYXs.append(HYXs_)
    IYXs.append(IYXs_)
    HZXs.append(HZXs_)
    IZXs.append(IZXs_)

import pickle as pck
with open('infos2.pkl','wb') as file:
    pck.dump([HYXs,IYXs,HZXs,IZXs], file)
    file.close()



pprint(np.mean(HYXs[0], axis=0))
# pprint(np.mean(HYXs[1], axis=0))
# pprint(np.mean(HYXs[2], axis=0))



def get_points(data):
    # infile = open(filename,'rb')
    # x = np.linspace(0,10000,)
    y = data
    # if mode == 'entropy':
    #     return x,y
    # else:
    #     return x, z
    # print(y[-1])
    x = [100*a for a in range(int(100))]
    x_new = np.linspace(0,10000, 100)
    intfunc = interpolate.interp1d(x,y,fill_value='extrapolate', kind='nearest')
    y_interp = intfunc(x_new)
    # infile.close()
    return x_new, y_interp
plt.style.use('seaborn')


def plotnorm():
    USABLE_WIDTH_mm = 100
    USABLE_HEIGHT_mm = 120
    YANK_RATIO = 0.0393701
    USABLE_WIDTH_YANK = USABLE_WIDTH_mm*YANK_RATIO
    USABLE_HEIGHT_YANK = USABLE_HEIGHT_mm*YANK_RATIO
    SUBPLOT_FONT_SIZE = 10
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(USABLE_WIDTH_YANK, USABLE_HEIGHT_YANK), tight_layout=True)
    x,y = get_points(np.mean(HYXs[0], axis=0))
    axes[0][0].plot(x,y)
    x,y = get_points(np.mean(IYXs[0], axis=0))
    axes[1][0].plot(x,y)
    x,y = get_points(np.mean(HZXs[0], axis=0))
    axes[0][1].plot(x,y)
    x,y = get_points(np.mean(IZXs[0], axis=0))
    axes[1][1].plot(x,y)
    # axes[0].set_ylabel('Mean Error',fontsize=SUBPLOT_FONT_SIZE)
    # axes[1].set_ylabel(r'Average $\Delta w$ per epoch',fontsize=SUBPLOT_FONT_SIZE)
    # axes[0].set_xticklabels([])
    # axes[1].set_xlabel('Epochs Elapsed',fontsize=SUBPLOT_FONT_SIZE)
    # axes[0].set_xlim(xmin=0)
    # axes[1].set_xlim(xmin=0)
    fig.savefig('Results.pdf')
    plt.show()
# plotnorm()
# def plotinfo():
#     USABLE_WIDTH_mm = 100
#     USABLE_HEIGHT_mm = 120
#     YANK_RATIO = 0.0393701
#     USABLE_WIDTH_YANK = USABLE_WIDTH_mm*YANK_RATIO
#     USABLE_HEIGHT_YANK = USABLE_HEIGHT_mm*YANK_RATIO
#     SUBPLOT_FONT_SIZE = 10
#     fig, axes = plt.subplots(nrows=2, ncols=1, sharey='row', figsize=(USABLE_WIDTH_YANK, USABLE_HEIGHT_YANK), tight_layout=True)
#     x,y = get_points('hyx.pkl', 'entropy')
#     print(x,y)
#     axes[0].plot(x,y)
#     x,y = get_points('hyx.pkl', 'mutinfo')
#     print(x,y)
#     axes[1].plot(x,y)
#     axes[0].set_ylabel('H(Y|X)',fontsize=SUBPLOT_FONT_SIZE)
#     axes[1].set_ylabel(r'I(X;Y)',fontsize=SUBPLOT_FONT_SIZE)
#     axes[0].set_xticklabels([])
#     axes[1].set_xlabel('Epochs Elapsed',fontsize=SUBPLOT_FONT_SIZE)
#     axes[0].set_xlim(xmin=0)
#     axes[1].set_xlim(xmin=0)
#     fig.savefig('info.pdf')
#     plt.show()

