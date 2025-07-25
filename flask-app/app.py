import os
import sys
import mlflow.artifacts

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
import warnings
import joblib
import json
import mlflow
from dotenv import load_dotenv
from src.data.data_preprocessing import normalize_text
import pandas as pd
import dagshub


load_dotenv()
warnings.filterwarnings('ignore')


app = Flask(__name__)

# mlflow.set_tracking_uri("https://dagshub.com/faheem-afk/mlops-mini-project.mlflow")
# dagshub.init(repo_owner='faheem-afk', repo_name='mlops-mini-project', mlflow=True)

# model_name = 'logisticRegression'
# model_info = json.load(open('reports/experiment_info.json', 'r'))

# model_uri = f"runs:/{model_info['run_id']}/{model_name}"
# log_model = mlflow.sklearn.load_model(model_uri)

# artifact_uri_ = f"runs:/{model_info['run_id']}/vectorizer/vectorizer.joblib"
# local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri_)
# vectorizer_ = joblib.load(local_path)


log_model_ = joblib.load('models/model.joblib')
vectorizer_ = joblib.load('models/vectorizer.joblib')

@app.route("/")
def home():
    return render_template("index.html", result=None)


@app.route("/predict", methods=['POST'])
def prediction():
    response = request.get_json()
    
    text_ = response.get('text', '')
    
    df_ = pd.DataFrame([[text_]], columns=['content'])
    
    # processed_df_ = normalize_text(df_)
    
    transformed_df_ = vectorizer_.transform(df_['content'].tolist())
    
    y_pred = log_model_.predict(transformed_df_)

    return jsonify({'sentiment': f"{y_pred[0]}"})

        
app.run(debug=True, port=5000)

