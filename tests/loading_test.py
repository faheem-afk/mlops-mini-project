import unittest
import mlflow
import os
import json
import warnings
from dotenv import load_dotenv
import pandas as pd
import joblib
from sklearn.metrics import \
    accuracy_score, precision_score, f1_score, recall_score

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

        artifact_uri_vec = f"""
        runs:/{model_info['run_id']}/vectorizer/vectorizer.joblib"""
        local_path = mlflow.artifacts. \
            download_artifacts(artifact_uri=artifact_uri_vec)
        cls.vectorizer = joblib.load(local_path)

        artifact_uri_csv = f"""
        runs:/{model_info['run_id']}/data/train_bow.csv"""
        local_path_csv = mlflow.artifacts. \
            download_artifacts(artifact_uri=artifact_uri_csv)
        cls.holdout_data = pd.read_csv(local_path_csv)

    def test_model_loaded(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        input_text = 'hi how are you'
        input_data_df = pd.DataFrame([[input_text]], columns=['content'])

        input_transformed = self.vectorizer.transform(
            input_data_df['content'].tolist())

        prediction = self.model.predict(input_transformed)

        self.assertEqual(input_transformed.shape[1],
                         len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_data_df.shape[0])
        self.assertEqual(len(prediction), 1)

    def test_model(self):
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_hold = self.holdout_data.iloc[:, -1]

        y_pred_new = self.model.predict(X_holdout)

        accuracy_new = accuracy_score(y_hold, y_pred_new)
        precision_new = precision_score(y_hold, y_pred_new)
        recall_new = recall_score(y_hold, y_pred_new)
        f1_new = f1_score(y_hold, y_pred_new)

        expected_accuracy = 0.70
        expected_precision = 0.70
        expected_recall = 0.70
        expected_f1 = 0.70

        self.assertGreaterEqual(accuracy_new, expected_accuracy)
        self.assertGreaterEqual(precision_new, expected_precision)
        self.assertGreaterEqual(recall_new, expected_recall)
        self.assertGreaterEqual(f1_new, expected_f1)


if __name__ == "__main__":
    unittest.main()
