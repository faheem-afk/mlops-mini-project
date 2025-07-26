import unittest
import mlflow
import os
import json
import warnings
from dotenv import load_dotenv

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
        return cls.model if cls.model else None

    def test_model_loaded(self):
        self.assertIsNotNone(self.model)


if __name__ == "__main__":
    unittest.main()
