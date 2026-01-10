import os
import numpy as np
import re

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

from IPython.display import HTML, display, clear_output

try:
    pyplot.rcParams["animation.html"] = "jshtml"
except:
    pyplot.rcParams["animation.html"] = "html5"
    
from scipy import optimize

from scipy.io import loadmat

import utils

data = loadmat(r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 07\Data\ex7data1.mat')
X = data['X']

pyplot.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=1)
pyplot.axis([0.5, 6.5, 2, 8])
pyplot.gca().set_aspect('equal')
pyplot.grid(False)
#pyplot.show()

def pca(X):
    
    m, n = X.shape
    
    U = np.zeros(shape=(n,n))
    S = np.zeros(n)
    
    sigma = (1/m) * X.T @ X
    
    U, S, aux = np.linalg.svd(sigma)
    
    return U, S

def projectData(X, U, K):
    
    Z = np.zeros((X.shape[0], K))
    
    Ureduce = U[:, :K]
    
    Z = np.dot(X, Ureduce)    
    
    return Z

def recoverData(Z, U, K):
    
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    
    Ureduce = U[:, :K]
    
    X_rec = np.dot(Z, Ureduce.T)
    
    return X_rec

'=============================== MAIN ==============================='


#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = utils.featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
fig, ax = pyplot.subplots()
ax.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=0.25)

for i in range(2):
    ax.arrow(mu[0], mu[1], 1.5 * S[i]*U[0, i], 1.5 * S[i]*U[1, i],
             head_width=0.25, head_length=0.2, fc='k', ec='k', lw=2, zorder=1000)

ax.axis([0.5, 6.5, 2, 8])
ax.set_aspect('equal')
ax.grid(False)

print('Top eigenvector: U[:, 0] = [{:.6f} {:.6f}]'.format(U[0, 0], U[1, 0]))
print(' (you should expect to see [-0.707107 -0.707107])')

K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: {:.6f}'.format(Z[0, 0]))
print('(this value should be about    : 1.481274)')

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: [{:.6f} {:.6f}]'.format(X_rec[0, 0], X_rec[0, 1]))
print('       (this value should be about  [-1.047419 -1.047419])')

#  Plot the normalized dataset (returned from featureNormalize)
fig, ax = pyplot.subplots(figsize=(5, 5))
ax.plot(X_norm[:, 0], X_norm[:, 1], 'bo', ms=8, mec='b', mew=0.5)
ax.set_aspect('equal')
ax.grid(False)
pyplot.axis([-3, 2.75, -3, 2.75])

# Draw lines connecting the projected points to the original points
ax.plot(X_rec[:, 0], X_rec[:, 1], 'ro', mec='r', mew=2, mfc='none')
for xnorm, xrec in zip(X_norm, X_rec):
    ax.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], '--k', lw=1)
    

#  Load Face dataset
data2 = loadmat(r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 07\Data\ex7faces.mat')
X = data2['X']

utils.displayData(X[:100, :], figsize=(8, 8))