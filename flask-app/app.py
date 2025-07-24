import os
import sys

import mlflow.artifacts

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request
import warnings
import joblib
import json
import mlflow
from dotenv import load_dotenv
from src.data.data_preprocessing import normalize_text 
import pandas as pd


app = Flask(__name__)

load_dotenv()
warnings.filterwarnings('ignore')

model_name = 'logisticRegression'
model_info = json.load(open('reports/experiment_info.json', 'r'))

model_uri = f"runs:/{model_info['run_id']}/{model_name}"
log_model = mlflow.sklearn.load_model(model_uri)

artifact_uri_ = f"runs:/{model_info['run_id']}/preprocessors/vectorizer.joblib"
local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri_)
vectorizer_ = joblib.load(local_path)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def prediction():
    text_ = request.form['chilgozu']

    df_ = pd.DataFrame([[text_]], columns=['content'])
    
    processed_df_ = normalize_text(df_)
    
   

    mlflow.set_experiment("mlops-mini-project")
    with mlflow.start_run():
        
        transformed_df_ = vectorizer_.transform(processed_df_)
        
        y_pred = log_model.predict(transformed_df_)
        
        if y_pred == 0:
            return "Sad"
        else:
            return "Happy"

        
app.run(debug=True, port=5008)

