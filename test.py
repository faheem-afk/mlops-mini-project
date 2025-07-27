from mlflow.tracking import MlflowClient
import mlflow
import time

# mlflow.set_tracking_uri("https://dagshub.com/faheem-afk/mlops-mini-project.mlflow")

client = MlflowClient()

run_id = "3da94e7d4e9c487ba9b54b263ed95f03"

model_path = "mlflow-artifacts:/b71038428594457092c6fe31ad132127/61667d3beee24f0e9b305e5516abef03/artifacts/logisticRegression"

model_uri = f"runs:/{run_id}/{model_path}"

model_name = "logisticRegression"

result = mlflow.register_model(model_uri, model_name)

print(result)