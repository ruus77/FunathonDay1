import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Total time: {time.time() - start}")
        return result
    return wrapper

def log_transform(y: pd.Series)->pd.Series:
    return y.transform(np.log1p)

def exp_transform(y: pd.Series)->pd.Series:
    return y.transform(np.expm1)

def get_data(url: str) -> pd.DataFrame:
    df = pd.read_parquet(url)
    df = df.convert_dtypes()
    return df

def report_metrics(y_true, y_pred):
    report = {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

    return pd.Series(report)

def get_data_splits(df: pd.DataFrame, random_state: int = 42):

    X = df.select_dtypes(include=np.number).copy()

    X = X.drop(columns=["price", "log_price"], errors="ignore")

    y = df["price"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=0.7,
        shuffle=True,
        random_state=random_state
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        train_size=0.5,
        shuffle=True,
        random_state=random_state
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test




class ModelSelector:

    def __init__(self, random_state:int=77, scoring:str="mse"):
      self.random_state = random_state
      self.scoring = scoring

    @staticmethod
    def metrics_report(y_pred: np.ndarray, y_true: np.ndarray):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return mse, mae, r2
    @timeit
    def params_search(self,
                    models: list,
                    models_names: list[str],
                    params_grid: list[dict[str, list[str]]],
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    cv:int=5,
                    scoring:str | None=None,
                    n_iter: int = 20):

        scoring = scoring if scoring else self.scoring

        results_list = []
        best_models_map = {}

        for model, name, grid in zip(models, models_names, params_grid):
            random_search = HalvingRandomSearchCV(cv=cv,
                                               #n_iter=n_iter,
                                               estimator=model,
                                               scoring=scoring,
                                               param_distributions=grid,
                                               verbose=1,
                                               #n_jobs=-1,
                                               error_score="raise",
                                               random_state=self.random_state,
                                               refit=True)
            random_search.fit(X_train, y_train)

            best_models_map[name] = random_search.best_estimator_
            cv_score = random_search.best_score_

            y_train_pred = random_search.best_estimator_.predict(X_train)
            metrics = self.metrics_report(y_true=y_train,
                                        y_pred=y_train_pred)
            row = {
                "data": "train",
                "name": name,
                f"cv_mean_{scoring}" : cv_score,
                "best_params": random_search.best_params_,
                "mse": metrics[0],
                "mae": metrics[1],
                "r2": metrics[2]}

            results_list.append(row)

        return pd.DataFrame(results_list).sort_values(by=f"cv_mean_{scoring}", ascending=False), best_models_map

    def evaluate(self, trained_models_map, X_eval, y_eval):
        results_list = []
        y_preds = {}

        for name, model in trained_models_map.items():
            y_pred = model.predict(X_eval)
            y_preds[name] = y_pred

            metrics = self.metrics_report(y_pred=y_pred, y_true=y_eval)

            row = {
                "data": "eval",
                "model_name": name,
                "mse": metrics[0],
                "mae": metrics[1],
                "r2": metrics[2]}
            results_list.append(row)

        df_results = pd.DataFrame(results_list)

        sort_column = self.scoring if self.scoring in df_results.columns else "mse"

        return df_results.sort_values(by=sort_column, ascending=False), y_preds