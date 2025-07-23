import dagshub
import mlflow
from dotenv import load_dotenv
import os


load_dotenv()

mlflow.set_tracking_uri(os.getenv('MLFLOW-TRACKING-URI'))

dagshub.init(repo_owner='faheem-afk', repo_name='mlops-mini-project', mlflow=True)


with mlflow.start_run():
    mlflow.log_param('parameter_name', 'value')
    mlflow.log_metric('metric_name', 1)