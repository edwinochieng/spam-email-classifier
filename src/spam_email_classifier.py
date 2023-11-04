import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('../data/spambase.data', header=None)

X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target (class labels)

