import unittest
import requests


class TestModelLoading(unittest.TestCase):

    def test_doc(self):
        response = requests.get("http://0.0.0.0:8888")
        self.assertEqual(response.status_code, 200)

    def test_predict_page(self):
        response = requests.post(
            'http://0.0.0.0:8888/predict', json={"text": "I like this"})
        self.assertEqual(response.status_code, 200)
        response = response.json()
        self.assertTrue('sentiment' in response)


if __name__ == "__main__":
    unittest.main()
