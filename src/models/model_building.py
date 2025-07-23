import xgboost as xgb
import warnings
import pandas as pd
import joblib
import yaml

warnings.filterwarnings('ignore')

params = yaml.safe_load(open('params.yaml', 'r'))['model_building']

train_bow = pd.read_csv(f"data/features/train_bow.csv")

X_train_bow = train_bow.iloc[:, :-1].values
y_train = train_bow.iloc[:, -1].values

# Define and train the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=params.get('use_label_encoder'), eval_metric=params.get('eval_metric'))
xgb_model.fit(X_train_bow, y_train)


joblib.dump(xgb_model, 'models/model.joblib')


