import os
import numpy as np

import re

from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat

import utils


def gaussianKernel(x1, x2, sigma):
    
    sim = 0
    
    sim = np.exp(-np.sum((x1-x2)**2)/(2*sigma**2))
    
    return sim

def dataset3Params(X, y, Xval, yval):
    C = 1
    sigma = 0.3
    
    Cs =[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    results = np.zeros((len(Cs)**2, 3))
    exp_no = 0
    for ii, C in enumerate(Cs):
        for jj, sigma in enumerate(sigmas):
            gamma = 1 / 2 / sigma**2

            model= utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))

            predictions = utils.svmPredict(model, Xval)

            results[exp_no, :] = C, sigma, np.mean(predictions != yval)
            exp_no += 1
    ind = np.argmin(results[:,2])
    C, sigma = results[ind,0], results[ind,1]
    print(f'For C = {C}, sigma = {sigma} we have the following prediction error: {results[ind,2]}')
    
    return C, sigma
    
'============================= MAIN ============================='
#data = loadmat(os.path.join('Data','ex6data1.mat'))
#X, y = data['X'], data['y'][:, 0]

#utils.plotData(X, y)
#pyplot.show()

#data = loadmat(os.path.join('Data','ex6data2.mat'))
#X, y = data['X'], data['y'][:, 0]
#utils.plotData(X , y)
#pyplot.show()

data = loadmat(os.path.join('Data', 'ex6data3.mat'))
X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]
utils.plotData(X, y)

C, sigma = dataset3Params(X, y, Xval, yval)

models = utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
utils.visualizeBoundary(X, y, models)
pyplot.show()
print(C, sigma)


#C = 1

#model = utils.svmTrain(X, y, C, utils.linearKernel, 1e-3, 20)
#utils.visualizeBoundaryLinear(X, y, model)
#pyplot.show()

#x1 = np.array([1, 2, 1])
#x2 = np.array([0, 4, -1])
#sigma = 0.01

#model = utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
#utils.visualizeBoundary(X, y, model)
#pyplot.show()

#sim = gaussianKernel(x1, x2, sigma)

#print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f:'
#      '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))