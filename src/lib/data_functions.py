import pandas as pd
import os
import math
import logging
import numpy as np
import scipy.stats as stats


def get_logger():
    """
    Get a logger instance with configured handlers and formatters.

    Returns:
        logging.Logger: The logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler("src/logs/my.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def number_to_string(n):
    names = ["", " thousand", " million", " billion", " trillion"]
    n = float(n)
    idx = max(
        0,
        min(
            len(names) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    return f"{n/10**(3*idx):.0f}{names[idx]}"


def load_csv(paths: list) -> list:
    """
    Load multiple CSV files into a list of pandas DataFrames.

    Args:
        paths (list(str)): List of file paths to the CSV files.

    Returns:
        dictionary(pd.DataFrame): Dictionary of pandas DataFrames containing the loaded CSV data.
    """
    return {
        os.path.splitext(os.path.basename(path))[0]: pd.read_csv(path, encoding="latin1")
        for path in paths
    }


def print_memory_usage(data: dict, logger=None) -> None:
    """
    Print the memory usage of each DataFrame in the input dictionary.

    Args:
        data (dict(pd.DataFrame)): Dictionary of pandas DataFrames to print the memory usage of.
    """
    total_memory = 0

    for key in data.keys():
        mem = data[key].memory_usage(deep=True).sum() * 1e-6
        total_memory += mem
        if logger is None:
            print(f"Memory usage of dataset {key}: {mem:.2f} MB")
        else:
            logger.info(f"Memory usage of dataset {key}: {mem:.2f} MB")

    if logger is None:
        print(f"\nTotal memory usage: {total_memory:.2f} MB")
    else:
        logger.info("")
        logger.info(f"Total memory usage: {total_memory:.2f} MB")


def get_eda_dataset(data: dict) -> pd.DataFrame:
    df = data["application_train"].copy()

    # Prepare merged dataset
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype("int64")
    data["bureau"]["SK_ID_CURR"] = data["bureau"]["SK_ID_CURR"].astype("int64")
    merged = pd.merge(df, data["bureau"], on="SK_ID_CURR", how="left")

    # Get the amount of closed loans
    closed_loans = merged[merged["CREDIT_ACTIVE"] == "Closed"]
    closed_loans_count = (
        closed_loans.groupby("SK_ID_CURR").size().reset_index(name="closed_loans")
    )
    df_results = pd.merge(df, closed_loans_count, on="SK_ID_CURR", how="left")
    df["closed_loans"] = df_results["closed_loans"].fillna(0)

    # Get the amount of active loans
    active_loans = merged[merged["CREDIT_ACTIVE"] == "Active"]
    active_loans_count = (
        active_loans.groupby("SK_ID_CURR").size().reset_index(name="active_loans")
    )
    df_results = pd.merge(df, active_loans_count, on="SK_ID_CURR", how="left")
    df["active_loans"] = df_results["active_loans"].fillna(0)

    # Get the total open credit amount
    open_credit_sum = (
        active_loans.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"]
        .sum()
        .reset_index(name="open_credit")
    )
    df_results = pd.merge(df, open_credit_sum, on="SK_ID_CURR", how="left")
    df["open_credit"] = df_results["open_credit"].fillna(0)

    # Get the total closed credit amount
    closed_credit_sum = (
        closed_loans.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"]
        .sum()
        .reset_index(name="closed_credit")
    )
    df_results = pd.merge(df, closed_credit_sum, on="SK_ID_CURR", how="left")
    df["closed_credit"] = df_results["closed_credit"].fillna(0)

    df["total_credit"] = df["AMT_CREDIT"] + df["open_credit"]
    df["dti"] = df["total_credit"] / df["AMT_INCOME_TOTAL"]

    return df.drop(columns=["SK_ID_CURR"], axis=1)


def get_correlations(data: pd.DataFrame):
    """
    Calculate the correlation matrix for the given DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: The correlation matrix of the input DataFrame.
    """
    df_corr = data.copy()
    categorical_columns = df_corr.select_dtypes(include=["category", "object"]).columns

    df_dummies = pd.get_dummies(df_corr[categorical_columns])
    df_corr = pd.concat([df_corr, df_dummies], axis=1)
    df_corr.drop(
        columns=categorical_columns,
        axis=1,
        inplace=True,
    )
    correlations = df_corr.corr()
    correlations.drop(columns=df_dummies.columns, inplace=True, axis=0)
    return correlations


def train_test_valid_split(
    data,
    label_column,
    stratify: str = None,
    test_size=0.2,
    valid_size=0.2,
    random_state=42,
):
    """
    Splits the data into training, validation and test sets.

    Parameters:
    - data: The input data.
    - label_column: The name of the label column.
    - stratify: The name of the column to use for stratification.
    - test_size: The size of the test set.
    - valid_size: The size of the validation set.
    - random_state: The random state for reproducibility.

    Returns:
    - train: The training set.
    - valid: The validation set.
    - test: The test set.
    """
    # Split into train and test set
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[stratify] if stratify else None,
    )

    # Split train set into train and validation set
    train, valid = train_test_split(
        train,
        test_size=valid_size,
        random_state=random_state,
        stratify=train[stratify] if stratify else None,
    )

    # Split into input features and target labels
    x_train = train.drop(columns=label_column)
    y_train = train[label_column]
    x_valid = valid.drop(columns=label_column)
    y_valid = valid[label_column]
    x_test = test.drop(columns=label_column)
    y_test = test[label_column]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "x_test": x_test,
        "y_test": y_test,
    }


def get_correlations(data: pd.DataFrame, as_list=False):
    """
    Calculate the correlation matrix for the given DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: The correlation matrix of the input DataFrame.
    """
    df_corr = data.copy()
    categorical_columns = df_corr.select_dtypes(include=["category", "object"]).columns

    df_dummies = pd.get_dummies(df_corr[categorical_columns])
    df_corr = pd.concat([df_corr, df_dummies], axis=1)
    df_corr.drop(
        columns=categorical_columns,
        axis=1,
        inplace=True,
    )
    correlations = df_corr.corr()
    correlations.drop(columns=df_dummies.columns, inplace=True, axis=0)

    if as_list:
        return (
            correlations.unstack().sort_values(ascending=False).drop_duplicates().head(10)
        )
    return correlations


def get_confidence_interval(data, confidence_level=0.95):
    mean = data.mean()
    standard_error_mean = stats.sem(data)  # Standard error of the mean
    ci = stats.t.interval(
        confidence_level, len(data) - 1, loc=mean, scale=standard_error_mean
    )
    return ci


def get_one_hot_encoded(data: pd.DataFrame, as_list=False):
    """
    One-hot encodes categorical columns in a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        as_list (bool, optional): If True, returns the encoded DataFrame as a list of column names.
            If False (default), returns the encoded DataFrame as a pandas DataFrame.

    Returns:
        pd.DataFrame or list: The one-hot encoded DataFrame or list of column names, depending on the value of `as_list`.
    """
    df = data.copy()
    categorical_columns = df.select_dtypes(include=["category", "object"]).columns

    df_dummies = pd.get_dummies(df[categorical_columns])
    df = pd.concat([df, df_dummies], axis=1)
    df.drop(
        columns=categorical_columns,
        axis=1,
        inplace=True,
    )
    return df
