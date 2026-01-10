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

def findClosestCentroids(X, centroids):
    
    K = centroids.shape[0]
    
    idx = np.zeros(X.shape[0], dtype=int)
    
    for i in range(X.shape[0]):
    
        squaredDistances = np.sum((centroids - X[i]) ** 2, axis=1)
        
        idx[i] = np.argmin(squaredDistances)
    
    return idx

def computeCentroids(X, idx, K):
    
    m, n = X.shape
    
    centroids = np.zeros((K,n))
    
    for k in range(K):
        
        points = X[idx == k]
        
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
        else:
            centroids[k] = np.zeros(n)
    
    return centroids

def kMeansInitCentroids(X, K):
    
    m, n = X.shape
    
    centroids = np.zeros((K, n))

    randidx = np.random.permutation(m)[:K]
    
    centroids = X[randidx, :]
    
    return centroids

'=============================== MAIN ==============================='
data = loadmat(r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 07\Data\ex7faces.mat')
X = data['X']

utils.displayData(X[:100, :], figsize=(8, 8))
pyplot.show()

#  normalize X by subtracting the mean value from each feature
X_norm, mu, sigma = utils.featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Visualize the top 36 eigenvectors found
utils.displayData(U[:, :36].T, figsize=(8, 8))

#  Project images to the eigen space using the top k eigenvectors
#  If you are applying a machine learning algorithm
K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a shape of: ', Z.shape)

pyplot.show()

#  Project images to the eigen space using the top K eigen vectors and
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed
K = 100
X_rec  = recoverData(Z, U, K)

# Display normalized data
utils.displayData(X_norm[:100, :], figsize=(6, 6))
pyplot.gcf().suptitle('Original faces')

# Display reconstructed data from only k eigenfaces
utils.displayData(X_rec[:100, :], figsize=(6, 6))
pyplot.gcf().suptitle('Recovered faces')
pass

A = mpl.image.imread(r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 07\Data\bird_small.png')
A /= 255
X = A.reshape(-1, 3)

# perform the K-means clustering again here
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = utils.runkMeans(X, initial_centroids,
                                 findClosestCentroids,
                                 computeCentroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = np.random.choice(X.shape[0], size=1000)

fig = pyplot.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], cmap='rainbow', c=idx[sel], s=8**2)
ax.set_title('Pixel dataset plotted in 3D.\nColor shows centroid memberships')
pass

X_norm, mu, sigma = utils.featureNormalize(X)

# PCA and project the data to 2D
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)



fig = pyplot.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

ax.scatter(Z[sel, 0], Z[sel, 1], cmap='rainbow', c=idx[sel], s=64)
ax.set_title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
ax.grid(False)
pass

pyplot.show()