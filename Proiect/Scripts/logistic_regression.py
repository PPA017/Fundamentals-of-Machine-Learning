import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

dataframe = pd.read_csv(os.path.join('Data','all-data.csv'), names=['Sentiment', 'News'], encoding='latin-1')
#print(dataframe)
dataframe['News'] = dataframe['News'].str.lower()
labelenc = LabelEncoder()
dataframe['Sentiment_Nr'] = labelenc.fit_transform(dataframe['Sentiment'])

vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')

X = vectorizer.fit_transform(dataframe['News'])
y = dataframe['Sentiment_Nr']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Classification report: ')
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'], 
            yticklabels=['Negative', 'Neutral', 'Positive'])

print(f'Accuracy : {model.score(X_test, y_test) * 100:.2f}%')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()