from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report, f1_score
import dagshub
import mlflow 
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

load_dotenv()

mlflow.set_tracking_uri(os.getenv('MLFLOW-TRACKING-URI'))

dagshub.init(repo_owner='faheem-afk', repo_name="mlops-mini-project", mlflow=True)
                
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
df.head()

def process_data(df: pd.DataFrame, test_size: float) -> tuple:
# delete tweet id
    df.drop(columns=['tweet_id'],inplace=True)

    final_df = df[df['sentiment'].isin(['happiness','sadness'])]

    final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)
    
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

    data_path = os.path.join('data', 'raw')
    
    return data_path, train_data, test_data

_, train_data, test_data = process_data(df, 0.2)

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    df.content=df.content.apply(lambda content : removing_numbers(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df


train_data_processed = normalize_text(train_data)
test_data_processed = normalize_text(test_data)

train_preprocessed_df = train_data_processed
test_preprocessed_df = test_data_processed

X_train_list = train_preprocessed_df['content'].values
y_train_list = train_preprocessed_df['sentiment'].values

X_test_list = test_preprocessed_df['content'].values
y_test_list = test_preprocessed_df['sentiment'].values

vectorizer = CountVectorizer()

X_train_bow = vectorizer.fit_transform(X_train_list)
X_test_bow = vectorizer.transform(X_test_list)

train_bow = pd.DataFrame(X_train_bow.toarray())
train_bow['label'] = y_train_list

test_bow = pd.DataFrame(X_test_bow.toarray())
test_bow['label'] = y_test_list

vectorized_df = pd.concat([train_bow, test_bow], axis=0)

X = vectorized_df.iloc[:, :-1]
y = vectorized_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

mlflow.set_experiment("Hyperparameter tuning")

with mlflow.start_run() as parent_run:
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    
    for params, mean_score, std_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
       
        with mlflow.start_run(run_name=f"LR with params: {params}" ,nested=True) as child_run:
          
            model=LogisticRegression(**params)
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)

            accuracy_ = accuracy_score(y_test, y_pred)
            recall_ = recall_score(y_test, y_pred)
            precision_ = precision_score(y_test, y_pred)
            f1_score_ = f1_score(y_test, y_pred)

            mlflow.log_metric("mean_score", mean_score)
            mlflow.log_metric("std_score", std_score)
            mlflow.log_metric("accuracy", accuracy_)
            mlflow.log_metric("recall", recall_)
            mlflow.log_metric("precision", precision_)
            mlflow.log_metric("f1_score", f1_score_)

            input_example = X_train.iloc[:5, : ]

            signature = infer_signature(X_train, model.predict(X_train))
            
            mlflow.sklearn.log_model(model, "LogisticRegression", signature=signature, input_example=input_example)

            mlflow.log_artifact(__file__)
            
    
    best_params= grid_search.best_params_
    best_score = grid_search.best_score_
    
    mlflow.log_params(best_params)
    mlflow.log_metric("best f1 score", best_score)
    
    