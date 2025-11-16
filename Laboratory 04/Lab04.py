import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_ = 0.0):
    """Implements the neural network cost function and gradient for a two layer nerual network
    which performs classification"""
    
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
    
    m = y.size
    
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    #part 1 FeedForward
    
    inputBias = np.concatenate([np.ones((m,1)), X], axis=1)
    
    inputsToLayer2 = inputBias @ Theta1.T
    outputsOfLayer2 = utils.sigmoid(inputsToLayer2) #activ function
    
    hiddenLayerBias = np.concatenate([np.ones((m,1)), outputsOfLayer2], axis=1)
    
    inputsToLayer3 = hiddenLayerBias @ Theta2.T
    outputsOfLayer3 = utils.sigmoid(inputsToLayer3)
    
    yMatrix = np.eye(num_labels)[y]
    
    J = (-1/ m) * np.sum(yMatrix * np.log(outputsOfLayer3) + (1 - yMatrix) * np.log(1 - outputsOfLayer3))
    
    #regularized cost function
    
    regJ = (lambda_ / (2 * m)) * (np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:] ** 2))
    J = J + regJ
    
    #part 2 BackPropagation
    
    outputLayerDelta = outputsOfLayer3 - yMatrix
    
    hiddenLayerDelta = (outputLayerDelta @ Theta2) * hiddenLayerBias * (1 - hiddenLayerBias)
    hiddenLayerDelta = hiddenLayerDelta[:, 1:]
    
    Theta2_grad = (1 / m) * (outputLayerDelta.T @ hiddenLayerBias)
    Theta1_grad = (1 / m) * (hiddenLayerDelta.T @ inputBias)
    
    #part 3 Regularizing grads
    
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]
    
    
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    return J, grad


    
def sigmoidGradient(z):
    
    g = np.zeros(z.shape)
    
    g = utils.sigmoid(z) * (1 - utils.sigmoid(z)) #formula g = sigm(z) * (1 - sigm(z))
    
    return g    



def randInitializeWeights(L_in, L_out, epsilon_init = 0.12):
    """Randomly initialize the weights of a layer in a neural network"""
    
    W = np.zeros((L_out, 1, + L_in))
    
    randValues = np.random.rand(L_out, 1 + L_in) #rand nums in [0,1)
    scaledValues = randValues * 2 * epsilon_init #rand vals * 2 * 0.12
    
    W = scaledValues - epsilon_init #simmetry to 0 ex: [-0.048, 0.048]
        
    return W

#===================== Data Loading and viewing =================================


os.chdir(r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 04')
data = loadmat(os.path.join('Data','ex4data1.mat'))

X, y = data['X'], data['y'].ravel()
y[y == 10] = 0

m = y.size

rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel)
#pyplot.show()

#===================== Neural Network Setup =================================


input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

weights = loadmat(os.path.join('Data','ex4weights.mat'))

Theta1, Theta2 = weights['Theta1'], weights['Theta2']

Theta2 = np.roll(Theta2, 1, axis=0)

nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

#===================== Neural Network Cost Function Test =================================

lambda_ = 0
J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)
#print('Cost at parameters (loaded from ex4weights): %.6f ' % J)
#print('The cost should be about                   : 0.287629.')

#===================== Sigmoid Grad Test =================================
z = np.array([-1, -0.5, 0, 0.5, 1])
g = sigmoidGradient(z)
#print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
#print(g)

#===================== Random Weights Test =================================
print('Initializing Neural Network Parameters ...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)
