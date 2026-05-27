from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
from scipy import stats




URL = "https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/1_input/transactions_EN.parquet"

RANDOM_STATE = 77



models = [RandomForestRegressor(random_state=RANDOM_STATE), XGBRFRegressor(random_state=RANDOM_STATE)]

MODELS_MAP = {m.__class__.__name__: m for m in models}


PARAM_GRID = [
    
        MODELS_MAP.keys[0] : {
        "n_estimators": stats.randint(100, 600),
        "max_depth": np.arange(3, 30),
        "min_samples_split": stats.randint(2, 20),
        "min_samples_leaf": stats.randint(1, 10),
        "max_features": ["sqrt", "log2"],
    },
      MODELS_MAP.keys[1] :  {
        "n_estimators": stats.randint(100, 600),
        "max_depth": np.arange(3, 12),
        "learning_rate": stats.loguniform(1e-3, 0.3),
        "subsample": stats.uniform(0.6, 0.4),       
        "colsample_bynode": stats.uniform(0.6, 0.4),
        "reg_lambda": stats.loguniform(.1, 10),
        "min_child_weight": stats.randint(1, 10)
    }
]