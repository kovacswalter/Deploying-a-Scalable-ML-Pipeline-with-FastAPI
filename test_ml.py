import pytest
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import train_model as tm
import os
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_length_data_pre_postsplit():
    """
    # Tests that the length of data remains the same after the split
    """
    data_length = len(tm.data)
    split_len = len(tm.train)+len(tm.test)
    assert data_length ==split_len, f"Expected {data_length}, but got {split_len}"


# TODO: implement the second test. Change the function name and input as needed
def test_model_type():
    """
    Tests that model has the RandomForestClassifier type.
    """
    with open(tm.model_path, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model,RandomForestClassifier), "Model issues"

# TODO: implement the third test. Change the function name and input as needed
def test_metric_size():
    """
    Test that precision, recall, and F1 score are within the range [0, 1].
    """
    assert tm.p <= 1
    assert tm.r <= 1
    assert tm.fb <= 1


   