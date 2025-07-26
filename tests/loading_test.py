import unittest
import mlflow
import os
import json
import warnings
from dotenv import load_dotenv
import pandas as pd
import joblib

load_dotenv()


class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore')
        dagshub_token = os.getenv("CI")
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        mlflow.set_tracking_uri(
            "https://dagshub.com/faheem-afk/mlops-mini-project.mlflow")
        model_name = 'logisticRegression'
        model_info = json.load(open('reports/experiment_info.json', 'r'))
        model_uri = f"runs:/{model_info['run_id']}/{model_name}"
        model_uri = f"runs:/{model_info['run_id']}/{model_name}"
        cls.model = mlflow.sklearn.load_model(model_uri)
        artifact_uri_ = f"""
        runs:/{model_info['run_id']}/vectorizer/vectorizer.joblib"""
        local_path = mlflow.artifacts. \
            download_artifacts(artifact_uri=artifact_uri_)
        cls.vectorizer = joblib.load(local_path)
        return (
            (cls.model, cls.vectorizer) if
            (cls.model and cls.vectorizer) else None)

    def test_model_loaded(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        input_text = 'hi how are you'
        input_data_df = pd.DataFrame([[input_text]], columns=['content'])
        input_transformed = self.vectorizer.transform(
            input_data_df['content'].tolist())
        prediction = self.model.predict(input_transformed)
        self.assertEqual(input_transformed.shape[1],
                         len(self.vectorizer.get_features_names_out()))
        self.assertEqual(len(prediction), input_data_df.shape.shape[0])
        self.assertEqual(len(prediction), 1)


if __name__ == "__main__":
    unittest.main()
