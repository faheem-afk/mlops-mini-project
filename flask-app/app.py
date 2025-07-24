from flask import Flask, render_template, request
import warnings
import joblib
import dagshub
import mlflow
from dotenv import load_dotenv
from src.data.data_preprocessing import normalize_text 
import pandas as pd


app = Flask(__name__)

load_dotenv()
warnings.filterwarnings('ignore')


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def prediction():
    text_ = request.form['chilgozu']

    df_ = pd.DataFrame([[text_]], columns=['content'])
    
    processed_df_ = normalize_text(df_)
    
    mlflow.set_tracking_uri("MLFLOW_TRACKING_URI")

    dagshub.init(repo_owner='faheem-afk', repo_name='mlops-mini-project', mlflow=True)

    mlflow.set_experiment("mlops-mini-project")
    with mlflow.start_run() as run:
        log_model = joblib.load('models/model.joblib')
        
        vectorizer_ = joblib.load('models/vectorizer.joblib')
        
        transformed_df_ = vectorizer_.transform(processed_df_)
        
        y_pred = log_model.predict(transformed_df_)
        
        if y_pred == 0:
            return "Sad"
        else:
            return "Happy"

        
app.run(debug=True, port=5007)

