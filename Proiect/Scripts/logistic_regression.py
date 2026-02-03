import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

dataframe = pd.read_csv(os.path.join('Data', 'train.csv'), encoding='latin-1')

dataframe.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

dataframe['Sex'] = dataframe['Sex'].map({'female': 0, 'male': 1})

dataframe['Age'] = dataframe['Age'].fillna(dataframe['Age'].median())

dataframe['Has_Cabin'] = dataframe['Cabin'].notnull()
dataframe['Has_Cabin'] = dataframe['Has_Cabin'].astype(int)

dataframe['Embarked'] = dataframe['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

dataframe['Fare'] = dataframe['Fare'].fillna(dataframe['Fare'].median())