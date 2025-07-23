import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
import yaml 


warnings.filterwarnings('ignore')

max_features = yaml.safe_load(open('params.yaml', 'r'))['feature_engineering']['max_features']

train_preprocessed_df = pd.read_csv(f"data/preprocessed/train_preprocessed_data.csv").dropna(axis=0)
test_preprocessed_df = pd.read_csv(f"data/preprocessed/test_preprocessed_data.csv").dropna(axis=0)

X_train = train_preprocessed_df['content'].values
y_train = train_preprocessed_df['sentiment'].values

X_test = test_preprocessed_df['content'].values
y_test = test_preprocessed_df['sentiment'].values

        
# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=max_features)

# # Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

train_bow = pd.DataFrame(X_train_bow.toarray())
train_bow['label'] = y_train

test_bow = pd.DataFrame(X_test_bow.toarray())
test_bow['label'] = y_test

os.makedirs('data/features', exist_ok=True)

train_bow.to_csv("data/features/train_bow.csv", index=False)
test_bow.to_csv("data/features/test_bow.csv", index=False)



