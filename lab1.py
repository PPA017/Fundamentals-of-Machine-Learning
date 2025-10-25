
import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D



def warmUpExercise():
    A = []
    A = np.eye(5)
    return A

def plotData(x, y):
    fig = pyplot.figure()
    
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Profit in $10,000')
    pyplot.xlabel('Population in 10,000s')
    pyplot.show()

def computeCost(X, y, theta) -> float:
    m = y.size
    J = 0
    
    pred = X.dot(theta) #h = X * theta
    errors = pred - y #h - y
    J = (1 / (2 * m)) * np.dot(errors,errors) #  1/2m * sum(h-y)^2
    
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta_new = theta.copy()
    J_history = []
    
    for i in range(num_iters):
        preds = X.dot(theta_new)     #h = X * theta
        errors = preds - y #h - y
        gradient = (1/m * X.T.dot(errors)) # 1/m * X^T(h - y)
        
        theta_new = theta_new - alpha * gradient
        
        J_history.append(computeCost(X,y, theta_new))
    
    return theta_new, J_history
    

#1
#print(warmUpExercise())

#============================= Linear Regression with one Variable ========================================

#2.1
data = np.loadtxt(os.path.join('Data','ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1]
m = y.size
#plotData(X, y)

#2.2
X = np.stack([np.ones(m),X], axis = 1)

#J = computeCost(X, y, theta = np.array([0.0, 0.0]))
#print('With theta = [0, 0] \n Cost computed = %.2f' %J)

#J = computeCost(X, y, theta=np.array([-1,2]))
#print('With theta = [-1,2]\n Cost computed = %.2f' %J)

theta = np.zeros(2)

iterations = 1500
alpha = 0.01

theta, J_History = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values aprox: [-3.6303, 1.1664]')

plotData(X[:,1],y)
#pyplot.plot(X[:,1], np.dot(X,theta), '-')
#pyplot.legend(['Training Data, Linear Regression'])
#pyplot.show()

predict1 = np.dot([1, 3.5], theta)
#print('For population = 35.000 we predict a profit of {:.2f} (Expected 4519.77)\n'.format(predict1*10000))

predict2 = np.dot([1, 7], theta)
#print('For population of 70.000 people we expect a profit of {:.2f} (Expected 45342.45)\n'.format(predict2*10000))


#==========================================================================================================================

#2.4 Visualizing Cost Function
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = computeCost(X, y, [theta0, theta1])

J_vals = J_vals.T

fig = pyplot.figure(figsize=(12,5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.title('Surface')        

ax = pyplot.subplot(122)
pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
pyplot.title('Contour, showing minimum')
pyplot.show()
pass

#======================================= Linear Regression with Multiple Variables ========================================================

