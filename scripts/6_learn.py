import lightgbm
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn import utils

df = pd.read_pickle('data/interesting.pkl')
df = df.set_index(['part', 'sequence', 'frame', 'r', 'c'])

features = df.columns.difference(['is_satellite'])
train = df.loc['train']

X_train = train[features]
y_train = train['is_satellite']

X_test = df.loc['test'][features]

model = lightgbm.LGBMClassifier(
    metric='binary',
    n_estimators=10_000,
    num_leaves=2 ** 6,
    learning_rate=.01,
    min_child_samples=30,
    #scale_pos_weight=2,
    random_state=42,
)

cv = model_selection.GroupKFold(n_splits=5)
groups = X_train.index.get_level_values('sequence')

oof = pd.Series(dtype=bool, index=X_train.index)
y_test = pd.DataFrame(index=X_test.index)

for i, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups=groups)):

    X_fit = X_train.iloc[fit_idx]
    y_fit = y_train.iloc[fit_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    model.fit(
        X_fit, y_fit,
        eval_set=[(X_fit, y_fit), (X_val, y_val)],
        eval_names=['fit', 'val'],
        early_stopping_rounds=20,
        verbose=100
    )
    oof.iloc[val_idx] = model.predict_proba(X_val)[:, 1]
    y_test[i] = model.predict_proba(X_test)[:, 1]

    print()  # for readability

print(metrics.classification_report(y_train, oof > .5, digits=4))
print()
print(pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False))

print()
print(oof.head())
oof.to_pickle('data/oof.pkl')

print()
print(y_test.head())
y_test.to_pickle('data/y_test.pkl')
