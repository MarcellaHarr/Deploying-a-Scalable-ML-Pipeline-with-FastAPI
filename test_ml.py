# == Import modules and libraries ==
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import apply_label


# == first test function ==
def test_dataset_info():
    """
    Test the training and test datasets expected row counts and verify they're pandas dataframes.
    """
    # == get directory ==
    path_dir = os.getcwd()
    data_path = os.path.join(
        path_dir,
        "data",
        "census.csv"
    )

    # == load data ==
    data = pd.read_csv(data_path)

    # == split training/test data ==
    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=45
    )

    # == check size ==
    assert train.shape[0] == 26048
    assert test.shape[0] == 6513

    # == check data type ==
    assert isinstance(
        train,
        pd.DataFrame
    )
    assert isinstance(
        test,
        pd.DataFrame
    )


# == second test function ==
def test_qst_nan_counts():
    """
    Test the dataset for zero NaN/NULL values and the expected number of `?` values.
    """
    # == get directory ==
    path_dir = os.getcwd()
    data_path = os.path.join(
        path_dir,
        "data",
        "census.csv"
    )

    # == load data ==
    data = pd.read_csv(data_path)

    # == count values ==
    na_count = data.isna().sum().sum()
    qstn_count = (
        data == "?"
    ).sum().sum()

    # == check counts ==
    assert na_count == 0
    assert qstn_count == 4262


# == third test function ==
def test_pred_label():
    """
    Test model's prediction label is a string and is expected to be `>50K`.
    """
    # == run the function ==
    pred_label = apply_label([1])

    # == check label and type ==
    assert isinstance(
        pred_label,
        str
    )
    assert pred_label == ">50K"