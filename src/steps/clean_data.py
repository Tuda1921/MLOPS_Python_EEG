import logging
import numpy as np
from typing import Tuple
from model.data_cleaning import DataSplitStrategy, TimeSeriesDataPreparer, FeatureExtract
import os
logging.basicConfig(level=logging.INFO)


def clean_data(filename: str, time_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        # Log the start of preprocessing
        logging.info("Starting preprocessing...")

        # preprocessing_strategy = DataTranspositionStrategy()
        # data_cleaning = DataCleaning(preprocessing_strategy)
        # X_scaled = data_cleaning.handle_data(filename)

        # Log the shape of X_scaled
        # logging.info(f"Shape of X_scaled: {X_scaled.shape}")

        time_series_preparer = TimeSeriesDataPreparer(time_steps)
        X, y = time_series_preparer.handle_data(filename)
        print(type(X))

        # Log the shape of X and y after time series preparation
        logging.info(f"Shape of X after time series preparation: {X.shape}")
        logging.info(f"Shape of y after time series preparation: {y.shape}")

        # Split the data into training and testing sets
        split_strategy = DataSplitStrategy(test_size=0.2, random_state=43)
        X_train, X_test, y_train, y_test = split_strategy.handle_data(X, y)

        # Log the shapes of the split data
        logging.info(f"Shape of X_train: {X_train.shape}")
        logging.info(f"Shape of X_test: {X_test.shape}")
        logging.info(f"Shape of y_train: {y_train.shape}")
        logging.info(f"Shape of y_test: {y_test.shape}")

        return X_train, y_train, X_test, y_test

    except Exception as e:
        logging.error(e)
        raise e
