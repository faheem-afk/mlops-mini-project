import mlflow
import json


def register_model(model_info_: dict):
    
    model_uri_ = f"runs:/{model_info_['run_id']}/{model_info_['model_name']}"
    model_version_ = mlflow.register_model(model_uri_, model_info_['model_name'])
    stage_='Staging'
    
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_info_['model_name'],
        version=model_version_,
        stage=stage_,
        archive_existing_versions=False
    )


model_info = json.load(open('reports/experiment_info.json', 'r'))

# register_model(model_info)
