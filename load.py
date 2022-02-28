"""This carries out:

## Task 1
1. Read the input from `gs://cloud-samples-data/ai-platform-unified/datasets/tabular/
petfinder-tabular-classification.csv` and load it in a Pandas Dataframe.
2. Split the dataset in 3 splits: train, validation, test with ratio of 60 (train) / 20
(validation) / 20 (test)
"""
from pathlib import Path

from google.cloud import storage
import pandas as pd
from sklearn.model_selection import train_test_split

BUCKET = "cloud-samples-data"
DATA = "ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)
OUTPATH = OUTPUTS / Path(DATA).name  # cache the data here


def load() -> pd.DataFrame:
    """Load petfinder-tabular-classification.csv as a Pandas DataFrame.
    If the data is not available locally,
    it will be downloaded from GCP, and saved to OUTPATH.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the data from
        petfinder-tabular-classification.csv.
    """
    if not OUTPATH.is_file():
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET)
        blob = bucket.blob(DATA)
        downloaded_file = blob.download_as_text(encoding="utf-8")
        OUTPATH.write_text(downloaded_file)
        print(f"Saved `{BUCKET}/{DATA}` to `{OUTPATH}`.")
    return pd.read_csv(OUTPATH)


def split(data: pd.DataFrame) -> tuple[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split the dataset into:
    train 60%, test 20%, validation 20%,
    using stratified sampling.

    Parameters
    ----------
    data : pd.DataFrame
        A Pandas DataFrame containing the data from
        petfinder-tabular-classification.csv,
        e.g. the output of `load`.

    Returns
    -------
    tuple[tuple[pd.DataFrame, pd.DataFrame]]
        Returns a tuple of DataFrames, in the form,
        ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
    """
    X = data.iloc[:, :-1]
    y = (data.iloc[:, -1] == "Yes") * 1.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1, stratify=y
    )  # 60% train, 40% test
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=1, stratify=y_test
    )  # now 20% test, 20% validation
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    # Informal tests:
    data = load()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split(data)
    print(data)
    print(f"{X_train.shape=} {X_test.shape=} {X_val.shape=}")
