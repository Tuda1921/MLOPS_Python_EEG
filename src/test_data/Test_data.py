import logging
import pandas as pd
import numpy as np
from typing import Tuple
from src.model.data_cleaning import DataSplitStrategy, TimeSeriesDataPreparer, FeatureExtract

logging.basicConfig(level=logging.INFO)


def clean_data(filename: str, time_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        # Log the start of preprocessing
        logging.info("Starting preprocessing...")

        time_series_preparer = TimeSeriesDataPreparer(time_steps)
        X, y = time_series_preparer.handle_data(filename)

        # Log the shape of X and y after time series preparation
        logging.info(f"Shape of X after time series preparation: {X.shape}")
        logging.info(f"Shape of y after time series preparation: {y.shape}")

        # Split the data into training and testing sets
        split_strategy = DataSplitStrategy(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = split_strategy.handle_data(X, y)

        # Log the shapes of the split data
        logging.info(f"Shape of X_train: {X_train.shape}")
        logging.info(f"Shape of X_test: {X_test.shape}")
        logging.info(f"Shape of y_train: {y_train.shape}")
        logging.info(f"Shape of y_test: {y_test.shape}")

        return X, y, X_train, y_train, X_test, y_test

    except Exception as e:
        logging.error(e)
        raise e

data_path = r'..\data\dataC3.csv'
# Dữ liệu đầu vào
df = pd.read_csv(data_path, header=None)
data = np.array(df.values)
# Lưu dữ liệu vào file CSV
df = pd.DataFrame(data)
df.to_csv('example_data.csv', header=False, index=False)

time_steps = 80
X, y, X_train, Y_train, X_test, Y_test = clean_data('example_data.csv', time_steps)

print("X:", X)
print("y:", y)
print("X_train:", X_train.shape)

print("\ny_train:", Y_train)

print("\nX_test:", X_test)

print("\ny_test:", Y_test)
