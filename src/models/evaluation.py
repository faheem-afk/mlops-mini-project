import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import warnings
import joblib


warnings.filterwarnings('ignore')


xgb_model = joblib.load('models/model.joblib')

test_bow = pd.read_csv(f"data/features/test_bow.csv")

X_test_bow = test_bow.iloc[:, :-1].values
y_test = test_bow.iloc[:, -1].values

# Make predictions
y_pred = xgb_model.predict(X_test_bow)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Make predictions
y_pred = xgb_model.predict(X_test_bow)
y_pred_proba = xgb_model.predict_proba(X_test_bow)[:, 1]

# Calculate evaluation metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)


import json

results = {
    'precision': float(precision),
    'recall': float(recall),
    'auc': float(auc),
    'y_pred': y_pred.tolist(),

}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
