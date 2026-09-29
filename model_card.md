# Model Card

## Model Details

This model was created by **Marcella Harris**. It is a **Random Forest (RF) Classifier** from scikit-learn, version 1.5.1. The RF was set with default hyperparameters, however, the random state (`random_state`) was set to 42. Given RF handles mixed data, both numeric and one-hot categorical, and with the census data having six numeric features, eight categorical features, and the process data function (`process_data()`) being able to do one-hot encoding, I thought this was a good model to use.

## Intended Use

This model is intended to classify whether a person's salary is *<=50K* or *>50K* from publicly available Census Bureau data. The intended users are individuals familiar with Machine Learning (ML).

## Training Data

The training data is Census Bureau data, acquired from the data folder in the starter repository: `data/census.csv`. The full **dataset has 32,561 rows** and **15 columns**, and **80%** (26,048 rows) was **used for training**. It was **processed** in the data script's (`ml/data.py`) process data function (`process_data()`) with a OneHotEncoder and label binarizer, with training set to True.

## Evaluation Data

The **evaluation data** is **20%** (6,513 rows) of the `census.csv` dataset; it was split using `train_test_split()` with `test_size=0.2` and `random_state=45`, and it was processed with the OneHotEncoder encoder and label binarizer fitted on the training data, with training set to False.

## Metrics

The model was evaluated using `precision`, `recall`, and `F1 score`. On the test data, the model scored a precision of **0.7431**, a recall of **0.6236**, and an F1 score of **0.6781**.

## Ethical Considerations

Bias may be inherent in the data because personal characteristics such as race, sex, and native country are represented unevenly across groups. The model’s **slice results** show differences in `F1 scores`, including `0.6836` for males and `0.6434` for females.

## Caveats and Recommendations

One caveat of the RF model was that it was trained without accounting for the ~76/24 class (*<=50K* / *>50K*) imbalance within the salary label. A strategy I could've used was to stratify the sample in the train/test split function (`train_test_split()`) in `train_model.py` so that both sets maintain the ~76/24 ratio.
