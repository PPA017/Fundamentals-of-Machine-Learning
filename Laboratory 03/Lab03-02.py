import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat

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


def predict(Theta1, Theta2, X):
    """Predict the label of an input given a trained neural network"""
    
    if X.ndim == 1:
        X = X[None]
        
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    
    p = np.zeros(X.shape[0])
    
    XBias = np.concatenate([np.ones((m,1)), X], axis=1)
    
    inputsToLayer2 = XBias @ Theta1.T
    outputsOfLayer2 = 1 / (1 + np.exp(-inputsToLayer2))
    
    hiddenLayerBias = np.concatenate([np.ones((m,1)), outputsOfLayer2], axis=1)
    
    inputsToLayer3 = hiddenLayerBias @ Theta2.T
    outputsOfLayer3 = 1 / (1 + np.exp(-inputsToLayer3))
    
    p = np.argmax(outputsOfLayer3, axis=1)
    
    return p
            
    

os.chdir(r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 03')
data = loadmat(os.path.join('Data','ex3data1.mat'))

X, y = data['X'], data['y'].ravel()
y[y == 10] = 0
m = y.size

indices = np.random.permutation(m)

rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

displayData(sel)
#pyplot.show()

#===================== Model Setup =================================
input_layer_size = 400 #20x20 images of digits
hidden_layer_size = 25 #25 hidden units
num_labels = 10 #10 digits

weights = loadmat(os.path.join('Data','ex3weights.mat'))

#Theta1 is 25x40 and Theta2 is 10x26. Model weights are from the dicitonary

Theta1, Theta2 = weights['Theta1'], weights['Theta2']

Theta2 = np.roll(Theta2, 1, axis=0) #We roll the last column of theta2 because of matlab indexing

#===================== Testing Neural Network =================================

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))

if indices.size > 0:
    i, indices = indices[0], indices[1:]
    displayData(X[i, :], figsize=(4, 4))
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')
    
#pyplot.show()