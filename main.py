import pandas as pd
import numpy as np

from utils import get_data, get_data_splits, ModelSelector
from config import URL, RANDOM_STATE, MODELS_MAP, PARAM_GRID


# -------------------------
# DATA
# -------------------------
data = get_data(url=URL)
data = data.iloc[:int(len(data)/ 10)]

X_train, X_valid, X_test, y_train, y_valid, y_test = get_data_splits(
    df=data,
    random_state=RANDOM_STATE
)


# -------------------------
# MODEL SELECTION
# -------------------------
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
    cv=3,
    n_iter=5
)


# -------------------------
# EVALUATION
# -------------------------
eval_results, _ = selector.evaluate(
    best_models,
    X_eval=X_valid,
    y_eval=y_valid
)


# -------------------------
# OUTPUT
# -------------------------
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
    print(f"\n{name}")
    print("-" * len(name))
    print(model)

