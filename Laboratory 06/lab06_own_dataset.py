from lab06_spam_classification import processEmail, emailFeatures
import utils
import os
import numpy as np

ham_dir = r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 06\Data\easy_ham'
spam_dir = r'C:\Users\pusca\Desktop\Uni\Master\Sem1\FML\Laboratory 06\Data\spam'

ham_files = os.listdir(ham_dir)
spam_files = os.listdir(spam_dir)

X_list = []
y_list = []

for f in ham_files:
    filepath = os.path.join(ham_dir, f)
    with open(filepath, 'r', encoding='latin1') as file:
        content = file.read()
    
    word_indices = processEmail(content, verbose=False)
    x = emailFeatures(word_indices)
    
    X_list.append(x)
    y_list.append(0)
    
for f in spam_files:
    filepath = os.path.join(spam_dir, f)
    with open(filepath, 'r', encoding='latin1') as file:
        content = file.read()
    
    word_indices = processEmail(content, verbose=False)
    x = emailFeatures(word_indices)
    
    X_list.append(x)
    y_list.append(1)
    
X = np.array(X_list)
y = np.array(y_list)

data = list(zip(X,y))
np.random.seed(42)
np.random.shuffle(data)

X, y = zip(*data)
X = np.array(X)
y = np.array(y)

m = X.shape[0]

train_end = int(0.7 * m)
val_end = int(0.85 * m)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

print("Training set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Test set size:", X_test.shape[0])