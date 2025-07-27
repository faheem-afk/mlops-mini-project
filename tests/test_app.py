import os
import sys
import unittest
import warnings

from dotenv import load_dotenv
import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask_app import app  # noqa: E402

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
        cls.client = app.app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', json={"text": "I like this"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'{"sentiment":"1"}' in
            response.data or b'{"sentiment":"0"}' in response.data)


if __name__ == "__main__":
    unittest.main()
