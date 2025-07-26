import os
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import yaml
warnings.filterwarnings('ignore')


def load_params(param_path: str) -> float:
    loaded = yaml.safe_load(open(param_path, 'r'))
    test_size = loaded['data_ingestion']['test_size']
    return test_size


def read_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df


def process_data(df: str, test_size: float) -> tuple:
    # delete tweet id
    df.drop(columns=['tweet_id'], inplace=True)
    final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
    final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
    train_data, test_data = train_test_split(
        final_df,
        test_size=test_size,
        random_state=42)
    data_path = os.path.join('data', 'interim')
    return data_path, train_data, test_data


def save_data(
    data_path: str,
    train_data: pd.DataFrame,
        test_data: str) -> None:
    os.makedirs(data_path, exist_ok=True)
    train_data.to_csv(f"{data_path}/train_data.csv", index=False)
    test_data.to_csv(f"{data_path}/test_data.csv", index=False)


def main() -> None:
    test_size = load_params('params.yaml')
    df = read_data(
        'https://raw.githubusercontent.com/campusx-official/'
        'jupyter-masterclass/main/tweet_emotions.csv')
    data_path, train_data, test_data = process_data(df, test_size)
    save_data(data_path, train_data, test_data)


if __name__ == "__main__":
    main()
