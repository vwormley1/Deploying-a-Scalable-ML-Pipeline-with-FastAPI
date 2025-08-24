import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
# TODO: add necessary import

# Load a small sample dataset for testing
data = pd.DataFrame({
    "age": [25, 38, 28, 44],
    "workclass": ["Private", "Self-emp-not-inc", "Private", "Private"],
    "education": ["Bachelors", "HS-grad", "HS-grad", "Some-college"],
    "marital-status": ["Never-married", "Married", "Divorced", "Married"],
    "occupation": ["Tech-support", "Exec-managerial", "Handlers-cleaners", "Adm-clerical"],
    "relationship": ["Not-in-family", "Husband", "Own-child", "Husband"],
    "race": ["White", "White", "Black", "White"],
    "sex": ["White", "White", "Black", "White"],
    "native-country": ["United-States", "United-States", "United-States", "United-States"],
    "salary": ["<=50K", ">50K", "<=50K", ">50K"]
})

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

X, y, encoder, lb = process_data(
    data,
    categorical_features=cat_features,
    label="salary",
    training=True
)

model = train_model(X, y)
preds = inference(model, X)
precision, recall, f1 = compute_model_metrics(y, preds)



# TODO: implement the first test. Change the function name and input as needed
def test_function_return_types():
    """
    Test that inference returns a numpy array of predictions
    and compute_model_metrics returns floats.
    """

    assert isinstance(preds, np.ndarray)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)

    # Your code here
    


# TODO: implement the second test. Change the function name and input as needed
def test_inference_invalid_input():
    """
    Test that inference will fail on invalid input.
    """

    invalid_data = None
    with pytest.raises(Exception):
        inference(model, invalid_data)

    # Your code here
   


# TODO: implement the third test. Change the function name and input as needed
def test_training_data_shape():
    """
    Test that processed training data has correct number of samples and features.
    """

    assert X.shape[0] == data.shape[0]
    assert X.shape[1] >= len(cat_features)

    # Your code here
    
