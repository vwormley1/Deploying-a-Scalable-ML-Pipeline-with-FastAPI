import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# TODO: load the cencus.csv data
project_path = Path(__file__).resolve().parent
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
    # your code here
    # use the train dataset 
    # use training=True
    # do not need to pass encoder and lb as input
    )

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# load the model
model = load_model(
    model_path
) 

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

overall_precision = p
overall_recall = r
overall_f1 = fb

overall_precision, overall_recall, overall_f1 = compute_model_metrics(y_test, preds)

# Number of samples and features
num_training_samples = X_train.shape[0]
num_features = X_train.shape[1]
num_test_samples = X_test.shape[0]

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
slice_output_path = project_path / "slice_output.txt"
with open (slice_output_path, "w") as f:
    f.write("Model Performance on Categorical Feature Slices:\n")
    f.write("=" * 60 + "n")

slice_metrics = []

for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model

            # your code here
            # use test, col and slicevalue as part of the input
        )

        slice_metrics.append({
            "feature": col,
            "value": slicevalue,
            "precision": p,
            "recall": r,
            "f1": fb
        })


        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)

model_card_path = project_path / "model_card_template.md"
with open(model_card_path, "w") as f:
    f.write("# Model Card\n\n")
    f.write("For additional information, see the [Model Card paper](https://arxiv.org/pdf/1810.03993.pdf).\n\n")
    
    f.write("## Model Details\n")
    f.write("This model is a Random Forest Classifier implemented using scikit-learn. "
            "It predicts if an individual's income exceeds $50,000 annually.\n")
    f.write("- Hyperparameters:\n")
    f.write("  - n_estimators: 100 (default)\n")
    f.write("  - max_depth: None (default)\n")
    f.write("  - random_state: 42 (explicit)\n")
    f.write("  - criterion: 'gini' (default)\n\n")

    f.write("## Intended Use\n")
    f.write("The model predicts income categories (`<=50K` or `>50K`) from demographic and employment features. "
            "It is designed for educational and experimental purposes, not for real-world decision-making.\n\n")

    f.write("## Training Data\n")
    f.write(f"- Number of training samples: {num_training_samples}\n")
    f.write(f"- Number of features after encoding: {num_features}\n")
    f.write("- Features:\n")
    f.write("  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country\n")
    f.write("  - Continuous: age, fnlwgt, capital-gain, capital-loss, hours-per-week\n")
    f.write("- Target: salary (binary: `<=50K` or `>50K`)\n\n")

    f.write("## Evaluation Data\n")
    f.write(f"- Test set: {num_test_samples} samples\n\n")

    f.write("## Metrics\n")
    f.write("The model was evaluated using precision, recall, and F1-score.\n\n")
    
    f.write("### Overall Performance\n")
    f.write(f"- Precision: {overall_precision:.4f}\n")
    f.write(f"- Recall: {overall_recall:.4f}\n")
    f.write(f"- F1-score: {overall_f1:.4f}\n\n")

    f.write("### Slice-based Performance\n")
    f.write("| Feature | Value | Precision | Recall | F1-score |\n")
    f.write("|---------|-------|-----------|--------|----------|\n")
    for m in slice_metrics:
        f.write(f"| {m['feature']} | {m['value']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n")
    
    f.write("\n## Ethical Considerations\n")
    f.write("The dataset may contain historical biases, e.g., gender, race, education. "
            "Predictions may reinforce existing biases if used in employment or financial decisions. "
            "This model is not intended for decisions affecting individuals’ lives.\n\n")

    f.write("## Caveats and Recommendations\n")
    f.write("- Model performance may vary across demographic groups.\n")
    f.write("- Hyperparameters are mostly defaults; further tuning could improve accuracy.\n")
    f.write("- The model is intended for educational use and demonstration of deployment and fairness evaluation.\n")

print(f"Model Card generated at: {model_card_path}")
