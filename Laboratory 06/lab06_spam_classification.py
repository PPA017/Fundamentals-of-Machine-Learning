import os
import numpy as np

import re

from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat

import utils

def processEmail(email_contents, verbose=True):
    
    vocabList = utils.getVocabList()
    
    word_indices = []
    
    email_contents = email_contents.lower()
    
    email_contents =re.compile('<[^<>]+>').sub(' ', email_contents)
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)
    email_contents = re.compile(r'(http|https)://[^\s]*').sub(' httpaddr ', email_contents)

    email_contents = re.compile(r'[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)    
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
    email_contents = re.split(r'[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)    
    email_contents = [word for word in email_contents if len(word) > 0]
    
    stemmer = utils.PorterStemmer()
    processed_email = []
    
    for word in email_contents:
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)
        
        if len(word) < 1:
            continue
        
        if word in vocabList:
            word_indices.append(vocabList.index(word))
    
    if verbose:
        print('----------------')
        print('Processed email:')
        print('----------------')
        print(' '.join(processed_email))
    return word_indices


def emailFeatures(word_indices):
    
    n = 1899
    x = np.zeros(n)
    
    x[np.array(word_indices)] = 1
    
    return x

'============================= MAIN ============================='

#with open(os.path.join('Data', 'emailSample1.txt')) as fid:
#    file_contents = fid.read()

#word_indices = processEmail(file_contents)
#print('-------------')
#print('Word Indices:')
#print('-------------')
#print(word_indices)

#data = loadmat(os.path.join('Data', 'spamTrain.mat'))
#X, y = data['X'].astype(float), data['y'][:, 0]

#print('Training Linear SVM (Spam Classification)')
#print('This may take 1 to 2 minutes ...\n')

#C = 0.1
#model = utils.svmTrain(X, y, C, utils.linearKernel)

#p = utils.svmPredict(model, X)

#print('Training Accuracy: %.2f' % (np.mean(p == y) * 100))

#data2 = loadmat(os.path.join('Data', 'spamTest.mat'))
#Xtest, ytest = data2['Xtest'].astype(float), data2['ytest'][:, 0]

#print('Evaluating the trained Linear SVM on a test set ...')
#p = utils.svmPredict(model, Xtest)

#print('Test Accuracy: %.2f' % (np.mean(p == ytest) * 100))


#idx = np.argsort(model['w'])
#top_idx = idx[-15:][::-1]
#vocabList = utils.getVocabList()

#print('Top predictors of spam:')
#print('%-15s %-15s' % ('word', 'weight'))
#print('----' + ' '*12 + '------')
#for word, w in zip(np.array(vocabList)[top_idx], model['w'][top_idx]):
#    print('%-15s %0.2f' % (word, w))

#filename = os.path.join('Data', 'emailSample1.txt')

#with open(filename) as fid:
#    file_contents = fid.read()

#word_indices = processEmail(file_contents, verbose=False)
#x = emailFeatures(word_indices)
#p = utils.svmPredict(model, x)

#rint('\nProcessed %s\nSpam Classification: %s' % (filename, 'spam' if p else 'not spam'))