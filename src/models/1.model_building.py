import warnings
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

train_bow = pd.read_csv(f"data/features/train_bow.csv")

X_train_bow = train_bow.iloc[:, :-1]
y_train = train_bow.iloc[:, -1]

# Define and train the XGBoost model
log_model = LogisticRegression(C=1, solver='liblinear', penalty='l2')
log_model.fit(X_train_bow, y_train)


joblib.dump(log_model, 'models/model.joblib')


