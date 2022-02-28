"""This carries out:

## Task 2
Write a python script to:
1. Load the data from `gs://cloud-samples-data/ai-platform-unified/datasets/tabular/
petfinder-tabular-classification.csv`
2. Uses the model you trained in the previous step to score all the rows in the CSV, excluding of
course the header.
3. Save the output into `output/results.csv` and make sure all files in the `output/` directory is
git ignored.
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from load import load

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
PREPROCESSOR_PATH = ARTIFACTS / "preprocessor.pickle"
MODEL_PATH = ARTIFACTS / "model.json"

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)
RESULTS_PATH = OUTPUTS / "results.csv"


def predict(data: pd.DataFrame) -> np.ndarray:
    """Predict whether a pet will be adopted,
    using the model trained on `petfinder-tabular-classification.csv`.

    Parameters
    ----------
    data : pd.DataFrame
        Data in the format of `petfinder-tabular-classification.csv`.

    Returns
    -------
    np.ndarray
        A 1D array containing predicted classes (0=No, 1=Yes).
    """
    preprocessor = pickle.loads(PREPROCESSOR_PATH.read_bytes())
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    X = data.iloc[:, :-1]
    return model.predict(preprocessor.transform(X))


def y_to_str(y: np.array) -> np.array:
    """Convert y's 0 and 1 values into No/Yes,
    (using NumPy indexing)."""
    return np.array(["No", "Yes"])[y]


def predict_into_csv(results_path: str | Path = RESULTS_PATH):
    """Load `petfinder-tabular-classification.csv`,
    run the model on it, add the predictions as a new column,
    and save the data to csv.

    Parameters
    ----------
    results_path : str | Path, optional
        Where to save the output csv, by default "outputs/results.csv"
    """
    data = load()
    y_pred = predict(data)
    data["Adopted_prediction"] = y_to_str(y_pred)
    data.to_csv(results_path, index=False)


if __name__ == "__main__":
    predict_into_csv()
