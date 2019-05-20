import pandas as pd
from config import UDPIPE_MODEL, BASE_DIR
from utils.get_feature_values import GetFeatures
import os
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline


def tokenizer(text):
    gf.get_info(text)
    return gf.lemmas


SEED = 23
gf = GetFeatures(UDPIPE_MODEL)
tfidf = TfidfVectorizer(lowercase=True)
ohe = CountVectorizer()
DON_DATASET_PATH = os.path.join(BASE_DIR, 'data', 'shell_dataset.csv')
DON_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'shell.pickle')

# load dataset

df = pd.read_csv(DON_DATASET_PATH)
df = df.fillna('')
df = df[df.target != 'unclear']
df = df[df.target != 'undefined']
df['target'] = df['target'].map({'false': 0, 'true': 1})
df = df[['target', 'left', 'right', 'token']]
cols = [col for col in df.columns if col not in ['target']]
X = df[cols]
y = df['target']


# train model
preprocessor = ColumnTransformer(
    transformers=[
        ('left', tfidf, 0),
        ('right', tfidf, 1),
        ('token', tfidf,  2)
    ]
)
steps = [('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=SEED,
                                                                           class_weight='balanced',
                                                                           solver='lbfgs'))]
pipeline = Pipeline(steps=steps)
pipeline.fit(X, y)

# save model
with open(DON_MODEL_PATH, 'wb') as f:
    pickle.dump(pipeline, f)

print('Model is saved.')

