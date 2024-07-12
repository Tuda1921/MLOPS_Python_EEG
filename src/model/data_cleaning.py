import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union, Tuple
import logging
from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self):
        pass

# class DataPreprocessingStrategy(DataStrategy):
#     def __init__(self, scaler=None):
#         self.scaler = scaler if scaler is not None else StandardScaler()
#     def handle_data(self, data: pd.DataFrame) -> pd.DataFrame: #Dùng cho bài toán phân loại
#         try:
#             # Đọc dữ liệu
#             df = pd.read_csv(data)
#             df_t = df.T
#             X = df_t.iloc[:-1,:].values
#             y = df_t.iloc[-1, :].values
#             X_scaled = self.scaler.fit_transform(X)
#             y_scaled = self.scaler.fit_transform(y)
#             return X_scaled, y_scaled
#         except Exception as e:
#             logging.error(e)
#             raise e

# class DataTranspositionStrategy(DataStrategy):
#     def __init__(self, scaler=None):
#         self.scaler = scaler if scaler is not None else StandardScaler()
#     def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
#         try:
#             # Đọc dữ liệu
#             df = pd.read_csv(data)
#             df_t = df.T
#             X = df_t.iloc[:-1,:].values
#             y = df_t.iloc[-1, :].values
#             X_scaled = self.scaler.fit_transform(X)
#             return X_scaled
#         except Exception as e:
#             logging.error(e)
#             raise e

def create_sequences(data, time_steps):
    X = []
    for k in range(len(data)):
        S_W = []
        for i in range(data.shape[1] - time_steps + 1):
            S_W.append(FeatureExtract.FFT(data[k, i:i + time_steps]))
        X.append(S_W)
    X = np.array(X)
    print(X.shape)
    return np.array(X)

class TimeSeriesDataPreparer:
    def __init__(self, time_steps):
        self.time_steps = time_steps

    def handle_data(self, data_path: str):
        try:
            logging.info(f"Preparing time series data with time_steps = {self.time_steps}")
            df = pd.read_csv(data_path, header=None)
            data = df.values

            X = create_sequences(data[:, :-1], self.time_steps)
            y = data[:, -1]

            # encoder = OneHotEncoder(sparse_output=False)
            # y = encoder.fit_transform(y.reshape(-1, 1))
            y = np.array(y)
            return X, y
        except Exception as e:
            logging.error(e)
            raise e

class DataSplitStrategy(DataStrategy):
    def __init__(self, test_size: float, random_state: int):
        self.test_size = test_size
        self.random_state = random_state

    def handle_data(self, X, y):
        try:
            logging.info(f"Splitting data with test_size = {self.test_size} and random_state = {self.random_state}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            logging.info(f"Shape of X_train: {X_train.shape}")
            logging.info(f"Shape of X_test: {X_test.shape}")
            logging.info(f"Shape of y_train: {y_train.shape}")
            logging.info(f"Shape of y_test: {y_test.shape}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(e)
            raise e

class DataCleaning:
    """
    Data cleaning class which preprocesses the data.
    """

    def __init__(self, strategy: DataStrategy):
        """Initializes the DataCleaning class with a specific strategy."""
        self.strategy = strategy

    def handle_data(self, data: str = None, X: pd.DataFrame = None, y: pd.Series = None) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        if data:
            return self.strategy.handle_data(data)
        else:
            return self.strategy.handle_data(X, y)
class FeatureExtract:
    """
    FeatureExtract class which extracts features from the data.
    """
    def __init__(self):
        pass

    def FFT(y: np.ndarray) -> np.ndarray:
        # y trong truong hop nay co do dai timestep
        # Output: FFT(y)
        flm = 160
        L = len(y)
        Y = np.fft.fft(y)
        Y[0] = 0
        P2 = np.abs(Y / L)
        P1 = P2[:L // 2 + 1]
        P1[1:-1] = 2 * P1[1:-1]
        # # Find the indices of the frequency values between 0.5 Hz and 4 Hz
        # f1 = np.arange(len(P1)) * flm / len(P1)
        # indices1 = np.where((f1 >= 0.5) & (f1 <= 4))[0]
        # delta = np.sum(P1[indices1])
        #
        # f1 = np.arange(len(P1)) * flm / len(P1)
        # indices1 = np.where((f1 >= 4) & (f1 <= 8))[0]
        # theta = np.sum(P1[indices1])
        #
        # f1 = np.arange(len(P1)) * flm / len(P1)
        # indices1 = np.where((f1 >= 8) & (f1 <= 13))[0]
        # alpha = np.sum(P1[indices1])
        #
        # f1 = np.arange(len(P1)) * flm / len(P1)
        # indices1 = np.where((f1 >= 13) & (f1 <= 30))[0]
        # beta = np.sum(P1[indices1])
        #
        # abr = alpha / beta
        # tbr = theta / beta
        # dbr = delta / beta
        # tar = theta / alpha
        # dar = delta / alpha
        # dtabr = (alpha + beta) / (delta + theta)
        # dict = {"delta": delta,
        #         "theta": theta,
        #         "alpha": alpha,
        #         "beta": beta,
        #         "abr": abr,
        #         "tbr": tbr,
        #         "dbr": dbr,
        #         "tar": tar,
        #         "dar": dar,
        #         "dtabr": dtabr
        #         }
        # # print(dict)
        return P1
