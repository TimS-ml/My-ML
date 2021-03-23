import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn_pandas import DataFrameMapper


categorical_features = ['feat_5', 'feat_6', 'feat_7', 'feat_8']
numerical_features = ['feat_1', 'feat_2', 'feat_3', 'feat_4']

X, y = make_classification(n_samples=10000, 
                           n_features=4, 
                           n_redundant=0, 
                           random_state=42, 
                           weights=[0.5])

# Add categorical columns
for col in range(4):
    num_classes = np.random.randint(2, 10)
    cat_col = np.random.randint(num_classes, size=X.shape[0]).reshape(-1,1)
    X = np.hstack((X, cat_col))

# To DataFrame
columns = [f'feat_{i+1}' for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=columns)
y = pd.DataFrame(y, columns=['label'])

# Scale regressors, modify categoricals
for col in numerical_features:
    mean = np.random.randint(10, 1000)
    std = np.random.randint(1, 100)
    X[col] = X[col].apply(lambda x: mean + std * x).astype(int)

for col in categorical_features:
    X[col] = X[col].apply(lambda x: f'str_{x}' if np.isnan(x)==False else x)

# Create Nans in dataset
for col in categorical_features + numerical_features:
    X[col] = X[col].sample(frac=0.7)
    
df = X.merge(y, left_index=True, right_index=True)


df.sample(3)


train_df, test_df = train_test_split(df, test_size=0.1, shuffle=False)
X_train, y_train = train_df[categorical_features + numerical_features], train_df['label']
X_test, y_test = test_df[categorical_features + numerical_features], test_df['label']


cat = [([c], [SimpleImputer(strategy='constant', fill_value='UNK'),
              OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)]) for c in categorical_features]
num = [([n], [SimpleImputer()]) for n in numerical_features]
mapper = DataFrameMapper(num + cat, df_out=True)
clf = CatBoostClassifier(iterations=1000,
                         learning_rate=0.01,
                         metric_period=100)

pipeline = Pipeline([
    ('preprocess', mapper),
    ('clf', clf)
])

pipeline.fit(X_train, y_train)


preprocessed_X_test = mapper.transform(X_test)


X_test[numerical_features + categorical_features].head()


preprocessed_X_test[numerical_features + categorical_features].head()


from joblib import dump, load
dump(pipeline, 'params/pipeline.joblib')
test_df.to_csv('params/test_df.csv')


def evaluation(pipeline, X, y):
    y_predict_proba = pipeline.predict_proba(X)[:, 1]
    return{
        'auc': roc_auc_score(y, y_predict_proba)
    }


evaluation(pipeline, X_train, y_train)


evaluation(pipeline, X_test, y_test)


cat = [([c], [SimpleImputer(strategy='constant', fill_value='UNK'),
              OneHotEncoder()]) for c in categorical_features]
num = [([n], [SimpleImputer(), StandardScaler()]) for n in numerical_features]
mapper = DataFrameMapper(num + cat, df_out=True)
clf = LogisticRegression()

pipeline = Pipeline([
    ('preprocess', mapper),
    ('clf', clf)
])

pipeline.fit(X_train, y_train)


preprocessed_X_test = mapper.transform(X_test)


X_test[numerical_features + categorical_features].head().T


preprocessed_X_test.head().T


evaluation(pipeline, X_train, y_train)


evaluation(pipeline, X_test, y_test)



