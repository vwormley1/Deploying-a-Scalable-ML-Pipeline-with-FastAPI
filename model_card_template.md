# Model Card

For additional information, see the [Model Card paper](https://arxiv.org/pdf/1810.03993.pdf).

## Model Details
This model is a Random Forest Classifier implemented using scikit-learn. It predicts whether an individual's income exceeds $50,000 annually.
- Hyperparameters:
  - n_estimators: 100 (default)
  - max_depth: None (default)
  - random_state: 42 (explicit)
  - criterion: 'gini' (default)

## Intended Use
The model predicts income categories (`<=50K` or `>50K`) from demographic and employment features. It is designed for educational and experimental purposes, not for real-world decision-making.

## Training Data
- Number of training samples: 26048
- Number of features after encoding: 108
- Features:
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
  - Continuous: age, fnlwgt, capital-gain, capital-loss, hours-per-week
- Target: salary (binary: `<=50K` or `>50K`)

## Evaluation Data
- Test set: 6513 samples

## Metrics
The model was evaluated using precision, recall, and F1-score.

### Overall Performance
- Precision: 0.7419
- Recall: 0.6384
- F1-score: 0.6863

### Slice-based Performance
| Feature | Value | Precision | Recall | F1-score |
|---------|-------|-----------|--------|----------|
| workclass | ? | 0.6538 | 0.4048 | 0.5000 |
| workclass | Federal-gov | 0.7971 | 0.7857 | 0.7914 |
| workclass | Local-gov | 0.7576 | 0.6818 | 0.7177 |
| workclass | Private | 0.7376 | 0.6404 | 0.6856 |
| workclass | Self-emp-inc | 0.7807 | 0.7542 | 0.7672 |
| workclass | Self-emp-not-inc | 0.7064 | 0.4904 | 0.5789 |
| workclass | State-gov | 0.7424 | 0.6712 | 0.7050 |
| workclass | Without-pay | 1.0000 | 1.0000 | 1.0000 |
| education | 10th | 0.4000 | 0.1667 | 0.2353 |
| education | 11th | 1.0000 | 0.2727 | 0.4286 |
| education | 12th | 1.0000 | 0.4000 | 0.5714 |
| education | 1st-4th | 1.0000 | 1.0000 | 1.0000 |
| education | 5th-6th | 1.0000 | 0.5000 | 0.6667 |
| education | 7th-8th | 0.0000 | 0.0000 | 0.0000 |
| education | 9th | 1.0000 | 0.3333 | 0.5000 |
| education | Assoc-acdm | 0.7000 | 0.5957 | 0.6437 |
| education | Assoc-voc | 0.6471 | 0.5238 | 0.5789 |
| education | Bachelors | 0.7523 | 0.7289 | 0.7404 |
| education | Doctorate | 0.8644 | 0.8947 | 0.8793 |
| education | HS-grad | 0.6594 | 0.4377 | 0.5261 |
| education | Masters | 0.8271 | 0.8551 | 0.8409 |
| education | Preschool | 1.0000 | 1.0000 | 1.0000 |
| education | Prof-school | 0.8182 | 0.9643 | 0.8852 |
| education | Some-college | 0.6857 | 0.5199 | 0.5914 |
| marital-status | Divorced | 0.7600 | 0.3689 | 0.4967 |
| marital-status | Married-AF-spouse | 1.0000 | 0.0000 | 0.0000 |
| marital-status | Married-civ-spouse | 0.7346 | 0.6900 | 0.7116 |
| marital-status | Married-spouse-absent | 1.0000 | 0.2500 | 0.4000 |
| marital-status | Never-married | 0.8302 | 0.4272 | 0.5641 |
| marital-status | Separated | 1.0000 | 0.4211 | 0.5926 |
| marital-status | Widowed | 1.0000 | 0.1579 | 0.2727 |
| occupation | ? | 0.6538 | 0.4048 | 0.5000 |
| occupation | Adm-clerical | 0.6338 | 0.4688 | 0.5389 |
| occupation | Armed-Forces | 1.0000 | 1.0000 | 1.0000 |
| occupation | Craft-repair | 0.6567 | 0.4862 | 0.5587 |
| occupation | Exec-managerial | 0.7952 | 0.7531 | 0.7736 |
| occupation | Farming-fishing | 0.5455 | 0.2143 | 0.3077 |
| occupation | Handlers-cleaners | 0.5714 | 0.3333 | 0.4211 |
| occupation | Machine-op-inspct | 0.5938 | 0.4043 | 0.4810 |
| occupation | Other-service | 1.0000 | 0.1923 | 0.3226 |
| occupation | Priv-house-serv | 1.0000 | 1.0000 | 1.0000 |
| occupation | Prof-specialty | 0.7880 | 0.7679 | 0.7778 |
| occupation | Protective-serv | 0.7353 | 0.5952 | 0.6579 |
| occupation | Sales | 0.7273 | 0.6667 | 0.6957 |
| occupation | Tech-support | 0.7143 | 0.6863 | 0.7000 |
| occupation | Transport-moving | 0.6250 | 0.4688 | 0.5357 |
| relationship | Husband | 0.7370 | 0.6923 | 0.7140 |
| relationship | Not-in-family | 0.7959 | 0.4149 | 0.5455 |
| relationship | Other-relative | 1.0000 | 0.3750 | 0.5455 |
| relationship | Own-child | 1.0000 | 0.1765 | 0.3000 |
| relationship | Unmarried | 0.9231 | 0.2667 | 0.4138 |
| relationship | Wife | 0.7132 | 0.6783 | 0.6953 |
| race | Amer-Indian-Eskimo | 0.6250 | 0.5000 | 0.5556 |
| race | Asian-Pac-Islander | 0.7857 | 0.7097 | 0.7458 |
| race | Black | 0.7273 | 0.6154 | 0.6667 |
| race | Other | 1.0000 | 0.6667 | 0.8000 |
| race | White | 0.7404 | 0.6373 | 0.6850 |
| sex | Female | 0.7229 | 0.5150 | 0.6015 |
| sex | Male | 0.7445 | 0.6599 | 0.6997 |
| native-country | ? | 0.7500 | 0.6774 | 0.7119 |
| native-country | Cambodia | 1.0000 | 1.0000 | 1.0000 |
| native-country | Canada | 0.6667 | 0.7500 | 0.7059 |
| native-country | China | 1.0000 | 1.0000 | 1.0000 |
| native-country | Columbia | 1.0000 | 1.0000 | 1.0000 |
| native-country | Cuba | 0.6667 | 0.8000 | 0.7273 |
| native-country | Dominican-Republic | 1.0000 | 1.0000 | 1.0000 |
| native-country | Ecuador | 1.0000 | 0.5000 | 0.6667 |
| native-country | El-Salvador | 1.0000 | 1.0000 | 1.0000 |
| native-country | England | 0.6667 | 0.5000 | 0.5714 |
| native-country | France | 1.0000 | 1.0000 | 1.0000 |
| native-country | Germany | 0.8462 | 0.8462 | 0.8462 |
| native-country | Greece | 0.0000 | 0.0000 | 0.0000 |
| native-country | Guatemala | 1.0000 | 1.0000 | 1.0000 |
| native-country | Haiti | 1.0000 | 1.0000 | 1.0000 |
| native-country | Honduras | 1.0000 | 1.0000 | 1.0000 |
| native-country | Hong | 0.5000 | 1.0000 | 0.6667 |
| native-country | Hungary | 1.0000 | 0.5000 | 0.6667 |
| native-country | India | 0.8750 | 0.8750 | 0.8750 |
| native-country | Iran | 0.3333 | 0.2000 | 0.2500 |
| native-country | Ireland | 1.0000 | 1.0000 | 1.0000 |
| native-country | Italy | 0.7500 | 0.7500 | 0.7500 |
| native-country | Jamaica | 0.0000 | 1.0000 | 0.0000 |
| native-country | Japan | 0.7500 | 0.7500 | 0.7500 |
| native-country | Laos | 1.0000 | 0.0000 | 0.0000 |
| native-country | Mexico | 1.0000 | 0.3333 | 0.5000 |
| native-country | Nicaragua | 1.0000 | 1.0000 | 1.0000 |
| native-country | Peru | 0.0000 | 0.0000 | 0.0000 |
| native-country | Philippines | 1.0000 | 0.6875 | 0.8148 |
| native-country | Poland | 0.6667 | 1.0000 | 0.8000 |
| native-country | Portugal | 1.0000 | 1.0000 | 1.0000 |
| native-country | Puerto-Rico | 0.8333 | 0.8333 | 0.8333 |
| native-country | Scotland | 1.0000 | 1.0000 | 1.0000 |
| native-country | South | 0.3333 | 0.5000 | 0.4000 |
| native-country | Taiwan | 0.7500 | 0.7500 | 0.7500 |
| native-country | Thailand | 1.0000 | 1.0000 | 1.0000 |
| native-country | Trinadad&Tobago | 1.0000 | 1.0000 | 1.0000 |
| native-country | United-States | 0.7392 | 0.6321 | 0.6814 |
| native-country | Vietnam | 1.0000 | 1.0000 | 1.0000 |
| native-country | Yugoslavia | 1.0000 | 1.0000 | 1.0000 |

## Ethical Considerations
The dataset may contain historical biases, e.g., gender, race, education. Predictions may reinforce existing biases if used in employment or financial decisions. This model is not intended for decisions affecting individuals’ lives.

## Caveats and Recommendations
- Model performance may vary across demographic groups.
- Hyperparameters are mostly defaults; further tuning could improve accuracy.
- The model is intended for educational use and demonstration of deployment and fairness evaluation.
