import pandas as pd 
import numpy as np


def get_data(url:str)->pd.DataFrame:
    df = pd.read_parquet(url)
    df = df.convert_dtypes()
    return df


def log_transform(df:pd.DataFrame, target:str)->pd.DataFrame:
    df[target] = df[target].apply(np.log1p)
    return df

