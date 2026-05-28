import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
#from sklearn.feature_selection import VarianceThreshold

from utils import get_data, get_data_splits, ModelSelector, log_transform, exp_transform, report_metrics
from config import URL, RANDOM_STATE, MODELS_MAP, PARAM_GRID, CAT_COLS


import joblib



data = get_data(url=URL)
data = data.iloc[:int(len(data)/ 10)]


TRAINED_MODELS = {
    m: joblib.load(f"models/{m}.pkl")
    for m in MODELS_MAP.keys()
}


X_train, X_valid, X_test, y_train, y_valid, y_test = get_data_splits(
    df=data,
    random_state=RANDOM_STATE
)
print("\n" + "=" * 80)
print(f"Zbior testowy - kształt: {X_test.shape}")

num_cols = [c for c in X_train.columns if c not in CAT_COLS]

ct = ColumnTransformer([
    ("cat_preprocess", OneHotEncoder(drop="first"), CAT_COLS),
    ("num_preprocess",
     FunctionTransformer(lambda x: x.astype(np.float32)),
     num_cols)
])


print("\n" + "=" * 80)
print(f"Zbior test - kształt: {X_test.shape}")

num_cols = [c for c in X_train.columns if c not in CAT_COLS]

X_train, X_valid, X_test = ct.fit_transform(X_train), ct.transform(X_valid), ct.transform(X_test) 
y_train, y_valid, y_test = log_transform(y_train), log_transform(y_valid), log_transform(y_test)



print("\n" + "=" * 80)

print(f"Zbior testowy po transofrmacji  - kształt: {X_test.shape}")


test_results = {}

for name, m in TRAINED_MODELS.items():
    y_pred = m.predict(X_test)
    test_results[name] = report_metrics(y_test, y_pred)


print(pd.DataFrame(test_results).T)