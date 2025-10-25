import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def normalEqn(X, y):
    
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)    
    return theta

data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis = 1)

theta_norm = normalEqn(X, y)
print('Theta computed from the normal equations is {:s}'.format(str(theta_norm)))

x = np.array([1, 1650, 3])
price = x.dot(theta_norm)
print(f'Predicted price of a 1650 sq-ft, 3 bedroom house using normal eq is: ${price:.2f}')
