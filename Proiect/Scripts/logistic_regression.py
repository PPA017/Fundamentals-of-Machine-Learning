import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

dataframe = pd.read_csv(os.path.join('Data', 'train.csv'), encoding='latin-1')