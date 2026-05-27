import pandas as pd 
import numpy as np
from utils import log_transform, get_data, get_data_splits, ModelSelector
from config import URL, RANDOM_STATE, MODELS_MAP, PARAM_GRID



data = get_data(url=URL)


X_train, X_valid, X_test, y_train, y_valid, y_test = get_data_splits(df=data, random_state=RANDOM_STATE)





selector = ModelSelector(random_state=RANDOM_STATE, scoring="mse")

for name, model in MODELS_MAP.items():
    results, best_models = selector.params_search(
    models=list(MODELS_MAP.values()),
    models_names=list(MODELS_MAP.keys()),
    params_grid=[PARAM_GRID[m] for m in MODELS_MAP.keys()],
    X_train=X_train,
    y_train=y_train,
    cv=5,
    n_iter=30
)