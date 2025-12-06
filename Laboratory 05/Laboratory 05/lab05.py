import os

import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat

import utils


def linearRegCostFunction(X, y, theta, lambda_=0.0):
    m = y.size
    
    J = 0
    grad = np.zeros(theta.shape)
    
    pred = X.dot(theta)
    errors = pred - y
    
    J = (1 / (2 * m)) * np.sum(errors ** 2)
    J = J + (lambda_ / (2 * m) * np.sum(theta[1:] ** 2))
    
    grad = (1/ m) * X.T.dot(errors)
    grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]
    
    return J, grad

def learningCurve(X, y, Xval, yval, lambda_ = 0):
    
    m = y.size
    
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    
    for i in range(1, m+1):
        Xtrain = X[:i, :]
        ytrain = y[:i]
        
        theta_params = utils.trainLinearReg(linearRegCostFunction, Xtrain, ytrain, lambda_)
        
        error_train[i - 1], _ = linearRegCostFunction(Xtrain, ytrain, theta_params, 0)
        error_val[i - 1], _ = linearRegCostFunction(Xval, yval, theta_params, 0)
             
    return error_train, error_val

def polyFeatures(X, p):
    
    X_poly = np.zeros((X.shape[0], p))
    
    #we turn [2] [3] [4] into [2]   [4]     [8]
    #                         [3]   [9]     [9]
    #                         [4]   [16]    [64]
    
    for i in range(1, p + 1):
        X_poly[:, i - 1] = X[:, 0] ** i 
    
    return X_poly

def validationCurve(X, y, Xval, yval):
    
    lambda_vec = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))
    
    for i in range(len(lambda_vec)):
        
        lam_param = lambda_vec[i]
        
        theta_params = utils.trainLinearReg(linearRegCostFunction, X, y, lam_param)
        
        error_train[i], _ = linearRegCostFunction(X, y, theta_params, 0)
        error_val[i], _  = linearRegCostFunction(Xval, yval, theta_params, 0)
        
    
    return lambda_vec, error_train, error_val

'====================================== Main =================================================='

script_dir = os.path.dirname(os.path.abspath(__file__))
data = loadmat(os.path.join(script_dir, 'Data', 'ex5data1.mat'))

X, y = data['X'], data['y'][:, 0]

Xtest, ytest = data['Xtest'], data['ytest'][:, 0]
Xval, yval = data['Xval'], data['yval'][:, 0]

m = y.size

pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1)
pyplot.xlabel('Change in water level (x)')
pyplot.ylabel('Water flowing out of the dam (y)')
#pyplot.show()

X = X.reshape(-1, 1)
X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
Xval = Xval.reshape(-1, 1)
Xval_aug = np.concatenate([np.ones((yval.size, 1)), Xval], axis=1)
error_train, error_val = learningCurve(X_aug, y, Xval_aug, yval, lambda_=0)


pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
pyplot.title('Learning curve for linear regression')
pyplot.legend(['Train', 'Cross Validation'])
pyplot.xlabel('Number of training examples')
pyplot.ylabel('Error')
pyplot.axis([0, 13, 0, 150])
#pyplot.show()

#print('# Training Examples\tTrain Error\tCross Validation Error')
#for i in range(m):
#   print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = utils.featureNormalize(X_poly)
X_poly = np.concatenate([np.ones((m, 1)), X_poly], axis=1)

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.concatenate([np.ones((ytest.size, 1)), X_poly_test], axis=1)

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.concatenate([np.ones((yval.size, 1)), X_poly_val], axis=1)

print('Normalized Training Example 1:')
X_poly[0, :]

lambda_ = 0
theta = utils.trainLinearReg(linearRegCostFunction, X_poly, y,
                             lambda_=lambda_, maxfun=55)


# Plot training data and fit
pyplot.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')

utils.plotFit(polyFeatures, np.min(X), np.max(X), mu, sigma, theta, p)

pyplot.xlabel('Change in water level (x)')
pyplot.ylabel('Water flowing out of the dam (y)')
pyplot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
pyplot.ylim([-20, 50])

pyplot.figure()
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
pyplot.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val)

pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
pyplot.xlabel('Number of training examples')
pyplot.ylabel('Error')
pyplot.axis([0, 13, 0, 100])
pyplot.legend(['Train', 'Cross Validation'])
#pyplot.show()

print('Polynomial Regression (lambda = %f)\n' % lambda_)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))
    
lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

print(error_train, error_val)

pyplot.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
pyplot.legend(['Train', 'Cross Validation'])
pyplot.xlabel('lambda')
pyplot.ylabel('Error')
pyplot.show()

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))