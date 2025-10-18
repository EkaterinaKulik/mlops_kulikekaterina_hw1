import numpy as np
from catboost import CatBoostClassifier

def score_with_catboost(X, model_path: str) -> np.ndarray:
    model = CatBoostClassifier()
    model.load_model(model_path)
    proba = model.predict_proba(X)[:, 1]
    return proba, model
