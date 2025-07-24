import pandas as pd
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import warnings
import joblib
import dagshub
import mlflow
from dotenv import load_dotenv
import os
import json
from mlflow.models.signature import infer_signature


load_dotenv()

mlflow.set_tracking_uri("MLFLOW_TRACKING_URI")

dagshub.init(repo_owner='faheem-afk', repo_name='mlops-mini-project', mlflow=True)

warnings.filterwarnings('ignore')

def save_model_info(run_id_:str, model_name_:str, file_path_:str):
    model_info_ = {
        "run_id": run_id_,
        "model_name": model_name_
    }
    json.dump(model_info_, open(file_path_, 'w'), indent=4)

mlflow.set_experiment("mlops-mini-project")
with mlflow.start_run() as run:
    log_model = joblib.load('models/model.joblib')

    test_bow = pd.read_csv(f"data/features/test_bow.csv")

    X_test_bow = test_bow.iloc[:, :-1]
    y_test = test_bow.iloc[:, -1]

    # Make predictions
    y_pred = log_model.predict(X_test_bow)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Make predictions
    y_pred = log_model.predict(X_test_bow)
    y_pred_proba = log_model.predict_proba(X_test_bow)[:, 1]

    # Calculate evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    results = {
        'precision': float(precision),
        'recall': float(recall),
        'auc': float(auc),

    }
    
    mlflow.log_metrics(results)
    
    if hasattr(log_model, 'get_params'):
        params= log_model.get_params()
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
            
    signature = infer_signature(X_test_bow, y_pred)
    
    input_examples = X_test_bow.iloc[:5, :]
    
    mlflow.log_artifact("data/features/test_bow.csv", artifact_path="data")
    mlflow.sklearn.log_model(log_model, "logisticRegression", signature=signature, input_example=input_examples)
    
    save_model_info(run.info.run_id, "logisticRegression", 'reports/experiment_info.json')
    
    with open('reports/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    mlflow.log_artifact("reports/metrics.json")