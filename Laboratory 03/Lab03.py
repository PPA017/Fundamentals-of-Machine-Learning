import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat

#%matplotlib inline

def displayData(X, example_width = None, figsize = (10,10)):
    
    #Computing rows and columns
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None] #Promoting to a 2d array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')
    
    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width
    
    #Compute the number of rows and columns to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    
    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize = figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)
    
    ax_array = [ax_array] if m == 1 else ax_array.ravel()
    
    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order = 'F'), 
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')



def lrCostFunction(theta, X, y, lambda_):
    
    '''Computes the cost of using theta as the parameter for
    regularized logistic regression and the gradient of the cost w.r.t to the
    pameters.'''
    
    if y.dtype == bool:
        y = y.astype(int)
    
    J = 0
    grad = np.zeros(theta.shape)
    m = y.size
    #compute the hypothesis h(x) = sigmoid(X * theta)
    
    preds = X.dot(theta)
    h = 1 / (1 + np.exp(-preds)) # sigmoid function
    
    #compute the regularized cost function
    
    J = (1/m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1- h))
    
    #regularization
    
    J += (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    
    #gradient
    
    grad = (1/m) * X.T.dot(h - y)
    temp = theta.copy()
    temp[0] = 0
    grad = grad + (lambda_ / m) * temp
    
    return J, grad



def oneVsAll(X, y, num_labels, lambda_):
    '''Trains num_labels logistic regression classifiers and returns each of these classifiers
    in a matrix all_theta, where the i-th row of all_theta corresponds to the classifier
    for label i'''
    
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.concatenate([np.ones((m, 1)), X], axis = 1)
    
    #we loop over each class label
    
    for c in range(num_labels):
        initialTheta = np.zeros(n + 1)
        options = {'maxfun' : 50}
        res = optimize.minimize(lrCostFunction, initialTheta, (X, (y == c), lambda_),
                                jac=True,
                                method='TNC',
                                options=options)
        all_theta[c] = res.x
        
    return all_theta


def predictOneVsAll(all_theta, X):
    '''Return a vector of preditions for each example in the matrix X
    We note that X ontains the examples in rows. all_theta is a matrix where
    the i-th row is a trained logistic regression tehta vector for the i-th class
    . You should set p to a vector of values from 0..K-1'''
    
    m = X.shape[0];
    num_labels = all_theta.shape[0]
    
    p = np.zeros(m)
    X = np.concatenate([np.ones((m, 1)), X], axis=1)    
    
    preds = X.dot(all_theta.T)
    
    h = 1 / (1+ np.exp(-preds)) #sigmoid for probs
    
    p = np.argmax(h, axis=1)
    

    
    return p


input_layer_size = 400 #20 x 20 input images of digits
num_labels = 10 #10 labels for 10 nums (0 is labeled as 10)

os.chdir(r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 03')
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

#swapping digit 0 to be recognized as 0 instead of 10 which is a matlab artifact where there is 
#no 0 digit

y[y == 10] = 0
m = y.size


rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]
displayData(sel)
#pyplot.show()

#===================== Vectorizing Logistic Regression =================================

theta_t = np.array([-2, -1, 1, 2], dtype=float)

X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

y_t = np.array([1, 0, 1, 0, 1])

lambda_t = 3

#print("X_t:\n", X_t)
#print("y_t:", y_t)
#print("theta_t:", theta_t)

#========================== Testing lrCostFunction ====================================

J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

#print('Cost         : {:.6f}'.format(J))
#print('Expected cost: 2.534819')
#print('-----------------------')
#print('Gradients:')
#print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
#print('Expected gradients:')
#print(' [0.146561, -0.548558, 0.724722, 1.398003]');


#========================== Testing oneVsAll ====================================
lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

#print("Shape of all_theta:", all_theta.shape)
#print("Expected shape: (10, 401)")
#print("\nFirst few values of all_theta for class 0:")
#print(all_theta[0, :5])

#========================== Testing predictOneVsAll ====================================
pred = predictOneVsAll(all_theta, X)
print('Training set Accuracy: {:.2f}%'.format(np.mean(pred==y) * 100))
