from collections import defaultdict as ddict
import numpy as np
from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

def discretize(x):
    return np.where(x > 0.5, 1, 0)


x, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='~/Documents/IPB/CW')
x = (x/255).astype('float32')
y = to_categorical(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1/7, random_state=42)

def loss_mse(preds, targets):
    loss = np.sum(np.square(np.subtract(preds, targets)))/2
    print('loss', loss)
    return loss

#derivative of loss function with respect to predictions
def loss_deriv(preds, targets):
    dL_dPred = preds - targets
    return dL_dPred

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

def sigmoid(x):
    sigmoid = (1/(1 + np.exp(-x)))
    return sigmoid
def sigmoid_prime(a):
    dsigmoid_da = sigmoid(a)*(1-sigmoid(a))
    return dsigmoid_da
from pprint import pprint

def collect_data(epochs=10000, hidden_size=32):
    input_size = 28*28
    output_size = 10
    hidden_size = hidden_size
    batch_size=48
    W1 = 0.1*np.random.randn(input_size, hidden_size)
    W2 = 0.1*np.random.randn(hidden_size, output_size)
    
    epoch = 0
    HYX = []
    IYX = []
    x_batch, y_batch = generate_batch([x_train, y_train], batch_size)
    print(x_batch)
    for U,V in zip(x_batch, y_batch):

        Z = U.dot(W1)
        H = sigmoid(Z)
        V_ = H.dot(W2)
        # calculate derivative of loss with respect to predictions
        dL_dV = loss_deriv(V_, V)
        if epoch % 100 == 99:
            px = ddict(int)
            py = ddict(int)
            pxy = ddict(lambda: ddict(int))
            n = 0

            for x, y in zip(discretize(V), discretize(H)):
                px[str(x)]+=1
                py[str(y)]+=1
                pxy[str(x)][str(y)]+=1
                n+=1

            for x in px.keys():
                px[x] /= n
            for y in py.keys():
                py[y] /= n

            hyx = 0
            iyx = 0
            for x in pxy.keys():
                for y in pxy[x].keys():
                    pxy[x][y] /= n
                    hyx += pxy[x][y] * np.log2(pxy[x][y]/px[x])
                    iyx += pxy[x][y] * np.log2(pxy[x][y]/(px[x]*py[y]))
            hyx = -hyx

            HYX.append(hyx)
            IYX.append(iyx)

        dL_dW2 = np.matmul(H.T, dL_dV)
        dL_dH = np.matmul(dL_dV, W2.T)
        dL_dZ = np.multiply(sigmoid_prime(Z), dL_dH)
        dL_dW1 = np.matmul(U.T, dL_dZ)
        
        # gradient descent to update weights
        W1 -= 0.01 * dL_dW1
        W2 -= 0.01 * dL_dW2

        if epoch+1 == epochs:
            break
        epoch+=1

    return HYX, IYX

HYXs = []
IYXs = []
for i in range(50):
    hyx, iyx = collect_data()
    HYXs.append(hyx)
    IYXs.append(iyx)
    print(i)