import pandas as pd
import numpy as np

CAT_COLS = ["merch", "cat_id", "gender", "one_city", "us_state", "jobs"]
NUM_COLS = [
    "amount","lat","lon","population_city",
    "merchant_lat","merchant_lon","hour","dayofweek"
]
DROP_COLS = ["transaction_time","name_1","name_2","street","post_code"]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")
    df["hour"] = df["transaction_time"].dt.hour
    df["dayofweek"] = df["transaction_time"].dt.dayofweek
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")
    for c in CAT_COLS:
        df[c] = df[c].astype(str).fillna("unknown")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    return df[CAT_COLS + NUM_COLS]
