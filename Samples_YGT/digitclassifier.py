from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt 


digits = load_digits() #getting the data set

X_scale = StandardScaler() #Standard scaler function for scaling the data around mean and dividing by standard deviation
X = X_scale.fit_transform(digits.data) # first fit the data then transform

y = digits.target # getting the label of the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4) # dividing the test and training data after shuffling to avoid overfitting

def convert_image_to_vector(y):         # function to convert image data to vector form
    y_v = np.zeros((len(y), 10))

    for i in range(len(y)):
        y_v[i,y[i]] = 1

    return y_v

Y_train = convert_image_to_vector(y_train)
Y_test = convert_image_to_vector(y_test)

nn = [64, 30, 10] # declaring the dimensions of neural net input layer is of  8 * 8 pixel size so 64 input nodes,
                  # we have 10 classes(0 - 9) so 10 output layer nodes and middle layer can be anything between the pervious two 

def f(x):                          # activation function sigmoidal
    return 1 / ( 1 + np.exp(-x))

def fd(x):                         # derivative of above activation function
    return f(x) * (1 - f(x))

def init_weights(nn):              # function to intialize weigths and bias of each layer randomly
    w = {}
    b = {}
    for i in range(1, len(nn)):
        w[i] = np.random.random_sample((nn[i], nn[i - 1]))
        b[i] = np.random.random_sample((nn[i], ))
    return w, b

def init_deltas(nn):                # function to intialize delta_weigths and delta_bias of each layer as zero in the beginning of each iteration
    delta_w = {}
    delta_b = {}
    for i in range(1, len(nn)):
        delta_w[i] = np.zeros((nn[i], nn[i - 1]))
        delta_b[i] = np.zeros((nn[i], ))
    return delta_w, delta_b

def get_mini_batches(X, Y, batch_size): # function to generate mini batches
    random_idx = np.random.choice(len(Y), len(Y), replace = False)
    x = X[random_idx, :]
    y = Y[random_idx]
    mini_batches = []
    for i in range(0, len(y), batch_size):
        mini_batches.append((x[i : i + batch_size, :], y[i : i + batch_size]))
    return mini_batches


def forward_pass(nn, W, B, X):  #function for forward pass
    h = {1 : X}  # output of 1st layer is the input X itself
    z = {}  
    for i in range(1, len(nn)):
        z[i + 1] = W[i].dot(h[i]) + B[i]
        h[i + 1] = f(z[i + 1])
    return h, z

def last_layer_delta(h, z, y): #function to calculate delta for last layer
    return -(y - h) * fd(z)

def hidden_layer_delta(delta, w, z): #function to calculate delta for hidden layer, here delta is of next layer and w and z id of the present layer
    return np.dot(np.transpose(w), delta) * fd(z)


def training_nn(X, Y, nn, alpha = 0.25, iter_count = 3000, lamb = 0.001):
    w, b = init_weights(nn)
    m = len(Y) # length of the sample
    avg_cost_func = []
    for cnt in range(iter_count):

        avg_cost = 0
        total_delta_w, total_delta_b = init_deltas(nn)
        mini_batches = get_mini_batches(X, Y, 100)
        for k in mini_batches:
            xmb = k[0]
            ymb = k[1]
            
            for i in range(len(ymb)):
                delta = {}
                h, z  = forward_pass(nn, w, b, xmb[i,:])
                for j in range(len(nn), 0, -1):

                    if(j == len(nn)):
                        delta[j] = last_layer_delta(h[j], z[j], ymb[i,:])
                        avg_cost += np.linalg.norm((ymb[i,:] - h[j]))
                    else:
                        if j > 1:
                            delta[j] = hidden_layer_delta(delta[j + 1], w[j], z[j])
                        total_delta_w[j] += np.dot(delta[j + 1][:,np.newaxis], np.transpose(h[j][:,np.newaxis]))
                        total_delta_b[j] += delta[j + 1]

            for j in range(1, len(nn)): 
                w[j] += -alpha * (1.0 / m * total_delta_w[j] + lamb * w[j])
                b[j] += -alpha * (1.0 / m * total_delta_b[j])
                
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)

    return w, b, avg_cost_func

                
W, B, avg_cost = training_nn(X_train, Y_train, nn)

plt.plot(avg_cost)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()

def predict(X, w, b, nn):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = forward_pass(nn, w, b, X[i,:])
        y[i] = np.argmax(h[len(nn)])
    return y

y_pred = predict(X_test, W, B, nn)
print(accuracy_score(y_test, y_pred) * 100)




