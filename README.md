# Virgin tech test

This repo contains code for the test set out in `TEST.md`.

## Instructions

### Set up
- Create a Python 3.10 environment, and install `requirements.txt`.
  - Or use conda and `environment.yml`, e.g. `conda env create -f environment.yml`
    (will create a conda env with the name "virgin-tech-test").
- Activate the environment, e.g. `conda activate virgin-tech-test`

### Run the code

#### Part 1

To run the code that carries out part 1:

```bash
python train.py
```

This will download the data, train the model, and save the model to `artifacts/model`.

The code for part 1 is in the files: 

  - train.py
  - load.py
  - preprocessor.py

#### Part 2

To run the code that carries out part 2:
```bash
python predict.py
```

This will load the model from disk, run it on `petfinder-tabular-classification.csv`, and save the data with the predicted values to `output/results.csv`.

The code for part 2 is in the files: 

  - predict.py
  - test_predict.py

To run the tests:

```bash
pytest
```

## References

In no particular order (I copied these from my open tabs).

- https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python 
- https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/
- https://scikit-learn.org/stable/modules/model_persistence.html
- https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
- https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
- https://github.com/googleapis/google-api-python-client