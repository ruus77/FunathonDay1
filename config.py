from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy import stats
import numpy as np


MLFLOW_TRACKING_URI = "http://localhost:5000"

URL = "https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/1_input/transactions_EN.parquet"

RANDOM_STATE = 77



models = [RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, tree_method='hist')]

MODELS_MAP = {m.__class__.__name__: m for m in models}


PARAM_GRID = {
    
        list(MODELS_MAP.keys())[0] : {
        "n_estimators": stats.randint(100, 800),
        "max_depth": np.arange(3, 20),
        "min_samples_split": stats.randint(10, 30),
        "min_samples_leaf": stats.randint(5, 30),
        "max_features": ["sqrt", "log2"],
    },



      list(MODELS_MAP.keys())[1] : {
        "n_estimators": stats.randint(400, 600),
        "max_depth": np.arange(2, 12),
        "learning_rate": stats.loguniform(.001, 0.3),
        "subsample": stats.uniform(0.6, 0.4),       
        "colsample_bynode": stats.uniform(0.6, 0.4),
        "reg_lambda": stats.loguniform(.1, 10),
        "min_child_weight": stats.randint(1, 10)
    }
}
