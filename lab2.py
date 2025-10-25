import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def featureNormalization(X):
    
    X_norm = X.copy()
    mu = np.mean(X, axis=0) #mean of each feature
    sigma = np.std(X, axis=0, ddof=0) #standard dev of each feature
    
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma

def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J = 0
    
    preds = X.dot(theta)
    multi_errors = preds - y
    
    J = (1 / (2 * m)) * np.dot(multi_errors, multi_errors)
    return J
    
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    m = y.shape[0]
    theta_new = theta.copy()
    J_history = []
    
    for i in range(num_iters):
        preds = X.dot(theta_new)
        errors = preds - y
        gradient = (1 / m) * X.T.dot(errors)
        
        theta_new = theta_new - alpha * gradient
        J_history.append(computeCostMulti(X, y, theta_new))
    
    return theta_new, J_history
    
    
        
    

data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[: ,2]
m = y.size

'''print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))'''



#=============================================== 3.1 Feature Normalization =====================================================================
    
X_norm, mu, sigma = featureNormalization(X)
#print('Computed mean (expected [2000.680    3.17]):', mu)
#print('Computed deviation (expected [786.202    0.752]):', sigma)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

#=============================================== 3.2 Gradient Descent =====================================================================

alpha = 0.04
num_iters = 142

theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost function')

print('Theta computed from gradient descent: {:s}'.format(str(theta)))
#pyplot.show()

x = np.array([1650, 3])
x_norm = (x - mu) / sigma
x_aux = np.concatenate(([1], x_norm))
price = np.dot(x_aux, theta)
print(f'Predicted price of a 1650 sq-ft, 3 beedroom house using grad descent is ${price:.2f}')





