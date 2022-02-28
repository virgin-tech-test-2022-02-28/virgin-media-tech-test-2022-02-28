"""This carries out:

## Task 1
4. Train an ML model using XGB to predict whether a pet will be adopted or not `Adopted` is the
target feature. You will need to use the validation to assess early stopping. You won't need to
hypertune any parameter, the default parameters will be sufficient, with the exception of the
number of trees which gets tuned by the early stopping mechanism.
5. The script needs to log to the user the performances of the model in the test set in terms
of F1 Score, Accuracy, Recall.
"""
from pathlib import Path
import pickle

from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from load import load, split
from preprocessor import Preprocessor

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
PREPROCESSOR_PATH = ARTIFACTS / "preprocessor.pickle"
MODEL_PATH = ARTIFACTS / "model.json"


def train():
    """Load the data, train the model,
    save the model, print classification report.
    """
    data = load()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split(data)
    # Fit the preprocessor and model
    preprocessor = Preprocessor()
    preprocessor.fit(X_train)
    model = XGBClassifier(use_label_encoder=False)
    model.fit(
        preprocessor.transform(X_train),
        y_train,
        eval_metric="error",
        eval_set=[(preprocessor.transform(X_val), y_val)],
        verbose=True,
    )
    # Save the preprocessor and model to disk
    PREPROCESSOR_PATH.write_bytes(pickle.dumps(preprocessor))
    model.save_model(MODEL_PATH)
    # Print classification report
    y_pred = model.predict(preprocessor.transform(X_test))
    print("\nXGBClassifier performance report:")
    print(classification_report(y_test, y_pred))
    # Print most frequent classifier baseline:
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(preprocessor.transform(X_train), y_train)
    y_pred = dummy_clf.predict(preprocessor.transform(X_test))
    print("\nMost-frequent classifier performance report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    train()
