"""This carries out:

## Task 2
4. Add a unit test to the prediction function.
"""
from pathlib import Path

import pandas as pd
import pytest

from predict import predict, predict_into_csv


def test_predict():
    """Basic test, that checks:
    - The model runs
    - The model accepts data in the format of `test_data.csv`
    - The model can make a prediction when provided with data in that format
    - The output is of the right length
    - The output values are in the right range
    """
    y = predict(pd.read_csv("test_data.csv"))
    assert len(y) == 4, "Unexpected output length."
    assert all(v in (0, 1) for v in list(y)), "Unexpected output values."


@pytest.fixture()
def tmp_path_predict_into_csv():
    """Use fixture so can clean up the data even if test fails."""
    tmp_results = Path("tmp_test_predict_into_csv.csv")
    assert not tmp_results.is_file(), "`tmp_results` should not exist yet."
    yield tmp_results
    tmp_results.unlink()  # delete the temporary data


def test_predict_into_csv(tmp_path_predict_into_csv):
    predict_into_csv(tmp_path_predict_into_csv)
    data = pd.read_csv(tmp_path_predict_into_csv)
    assert list(data.columns) == [
        "Type",
        "Age",
        "Breed1",
        "Gender",
        "Color1",
        "Color2",
        "MaturitySize",
        "FurLength",
        "Vaccinated",
        "Sterilized",
        "Health",
        "Fee",
        "PhotoAmt",
        "Adopted",
        "Adopted_prediction",
    ], "Columns are not as specified."
    assert set(data.iloc[:, -1]) == set(["Yes", "No"]), "Output values are invalid."
    assert data.shape == (11537, 15), "Output data has unexpected shape."
