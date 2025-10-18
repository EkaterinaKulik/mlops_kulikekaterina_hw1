import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

def export_submission(preds: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sub = pd.DataFrame({"row_id": np.arange(len(preds)), "target": preds})
    sub.to_csv(os.path.join(output_dir, "sample_submission.csv"), index=False)

def export_top5_fi(model: CatBoostClassifier, output_dir: str):
    fi = model.get_feature_importance(prettified=True)
    top5 = dict(fi.head(5).values)
    with open(os.path.join(output_dir, "feature_importance_top5.json"), "w", encoding="utf-8") as f:
        json.dump(top5, f, ensure_ascii=False, indent=2)

def export_density_plot(preds: np.ndarray, output_dir: str):
    plt.figure()
    plt.hist(preds, bins=30, density=True)
    plt.title("Distribution of predicted fraud scores")
    plt.xlabel("Fraud score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pred_density.png"))
