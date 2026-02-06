import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

class Sentiment_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(Sentiment_LSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        last_step = hidden[-1]
        
        return self.fc(last_step)

def pad_and_encode(text_list, max_len = 50):
    features = np.zeros((len(text_list), max_len), dtype=int)
    
    for i, text in enumerate(text_list):
        words = text.split()
        encoded = [word_idx.get(w, 0) for w in words[:max_len]]
        features[i, :len(encoded)] = encoded
    
    return features

dataframe = pd.read_csv(os.path.join('Data','all-data.csv'),names=['Sentiment', 'News'] ,encoding='latin-1')
dataframe['News'] = dataframe['News'].str.lower()

labelenc = LabelEncoder()
y = labelenc.fit_transform(dataframe['Sentiment'])

all_words = ' '.join(dataframe['News']).split()
count = Counter(all_words)

vocabulary = sorted(count, key = count.get, reverse=True)[:2000]
word_idx = {word: i + 1 for i, word in enumerate(vocabulary)}

X = pad_and_encode(dataframe['News'].values)

X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

vocab_size = len(word_idx) + 1
model = Sentiment_LSTM(vocab_size=vocab_size, embedding_dim=64, hidden_dim=64, output_dim=3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        output = model(inputs)
        
        loss = criterion(output, labels)
        
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')
    
model.eval()
correct = 0
total = 0

with torch.no_grad():
    
    for inputs, labels in test_loader:
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')