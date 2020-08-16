import math

import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
import tqdm


df = pd.read_pickle('data/interesting.pkl')

def score(model, X_fit, X_val, y_fit, y_val):

    model.fit(X_fit, y_fit)
    y_pred = model.predict(X_val)
    return metrics.f1_score(y_val, y_pred)

pool = df.columns.difference(['part', 'sequence', 'frame', 'r', 'c', 'is_satellite']).tolist()
selected = []

train = df.query('part == "train"')
X = train[pool].copy()
X[:] = preprocessing.scale(X)
y = train['is_satellite']

X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X, y, random_state=42)

model = linear_model.LogisticRegression(solver='sag')

best_score = -math.inf

while pool:

    best_candidate = None
    no_improvement = True

    for feature in tqdm.tqdm(pool):
        new_score = score(
            model=model,
            X_fit=X_fit[[*selected, feature]],
            X_val=X_val[[*selected, feature]],
            y_fit=y_fit,
            y_val=y_val
        )

        if new_score > best_score:
            best_candidate = feature
            best_score = new_score
            no_improvement = False

    if no_improvement:
        print('No more improvement, stopping')
        break
    else:
        selected.append(best_candidate)
        pool.remove(best_candidate)
        print(f'Added {best_candidate}, current score is {best_score:.3f}')
