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
data = loadmat(r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 07\Data\ex7data2.mat')
X = data['X']

K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for first 3 examples: ')
print(idx[:3])
print('(the closest centroids should be 0, 2, 1 respectively)')

centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids:')
print(centroids)
print('\nThe centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]')

#data = loadmat(r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 07\Data\ex7data2.mat')

max_iters = 10

initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
#centroids, idx, anim = utils.runkMeans(X, initial_centroids, findClosestCentroids,
#                                       computeCentroids, max_iters, True)

#pyplot.show()

K = 16
max_iters = 10

A = mpl.image.imread(r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 07\Data\bird_small.png')
A = A.copy()
A /= 255
X = A.reshape(-1, 3)

initial_centroids = kMeansInitCentroids(X, K)

centroids, idx = utils.runkMeans(X, initial_centroids, findClosestCentroids,
                                 computeCentroids, max_iters)

X_recovered = centroids[idx, :].reshape(A.shape)

fig, ax = pyplot.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(A* 255)
ax[0].set_title('Original')
ax[0].grid(False)

ax[1].imshow(X_recovered * 255)
ax[1].set_title('Compressed with %d colors' % K)
ax[1].grid(False)

pyplot.show()