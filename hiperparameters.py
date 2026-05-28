import numpy as np

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
#from sklearn.feature_selection import VarianceThreshold

from utils import get_data, get_data_splits, ModelSelector, log_transform
from config import URL, RANDOM_STATE, MODELS_MAP, PARAM_GRID, CAT_COLS

import joblib
import os


data = get_data(url=URL)
data = data.iloc[:int(len(data)/ 10)]

X_train, X_valid, X_test, y_train, y_valid, y_test = get_data_splits(
    df=data,
    random_state=RANDOM_STATE
)




print("\n" + "=" * 80)
print(f"Zbior treninogwy - kształt: {X_train.shape}")

num_cols = [c for c in X_train.columns if c not in CAT_COLS]

ct = ColumnTransformer([
    ("cat_preprocess", OneHotEncoder(drop="first"), CAT_COLS),
    ("num_preprocess",
     FunctionTransformer(lambda x: x.astype(np.float32)),
     num_cols)
])


print("\n" + "=" * 80)
X_train, X_valid, X_test = ct.fit_transform(X_train), ct.transform(X_valid), ct.transform(X_test) 
y_train, y_valid, y_test = log_transform(y_train), log_transform(y_valid), log_transform(y_test)





print(f"Zbior treninogwy po transofrmacji  - kształt: {X_train.shape}")

selector = ModelSelector(
    random_state=RANDOM_STATE,
    scoring="neg_mean_squared_error"
)

results, best_models = selector.params_search(
    models=list(MODELS_MAP.values()),
    models_names=list(MODELS_MAP.keys()),
    params_grid=[PARAM_GRID[m] for m in MODELS_MAP.keys()],
    X_train=X_train,
    y_train=y_train,
    cv=3
)


eval_results, _ = selector.evaluate(
    best_models,
    X_eval=X_valid,
    y_eval=y_valid
)


print("\n" + "=" * 80)
print("TRAIN RESULTS")
print("=" * 80)
print(results.to_string(index=False))

print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)
print(eval_results.to_string(index=False))

print("\n" + "=" * 80)
print("BEST MODELS")
print("=" * 80)

for name, model in best_models.items():
    joblib.dump(model, f"models/{name}.pkl")
    print(f"\n{name}")
    print("-" * len(name))
    print(model)

