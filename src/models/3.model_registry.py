import mlflow
import mlflow.tracking

def register_model(model_name_:str, model_info_: dict):
    
    model_uri_ = f"runs:/{model_info_['run_id']}/{model_info_['model_path']}"
    model_version_ = mlflow.register_model(model_uri_, model_name_)
    stage_='Staging'
    
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name_,
        version=model_version_,
        stage=stage_,
        archive_existing_versions=False
    )