import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union, Tuple
import logging
from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder
# from keras.optimizers import Adam
import scipy as sp
from scipy.interpolate import interp1d
from tqdm import tqdm
import scipy.stats
import numpy as np
import scipy.stats as stats
import pywt
from statsmodels.tsa.ar_model import AutoReg
import scipy.signal
from scipy.signal import find_peaks
import warnings

# Tắt toàn bộ UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


import os
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
    if not (os.path.exists('X.npy')):
        X = []
        for k in tqdm(range(len(data))):
            S_W = []
            for i in range(0,data.shape[1] - time_steps + 1, time_steps):
                extractor = FeatureExtract(data[k, i:i + time_steps], 160)
                features = extractor.FeatureExtract()
                S_W.append(features)
                # S_W.append(data[k, i:i + time_steps])
            X.append(S_W)
        X = np.array(X)
        print(X.shape)
        np.save('X.npy',X)
    else:
        X = np.load('X.npy')
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

            encoder = OneHotEncoder(sparse_output=False)
            y = encoder.fit_transform(y.reshape(-1, 1))
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

class Preprocessing:
    """
    Preprocessing class which preprocesses the data.
    """
    def __init__(self, strategy: DataSplitStrategy):
        self.strategy = strategy

    def filter_data(data):
        # Bandpass filter
        band = [0.5 / (0.5 * 160), 40 / (0.5 * 160)]
        b, a = sp.signal.butter(0, band, btype='band', analog=False, output='ba')
        data = sp.signal.lfilter(b, a, data)

        # plt.hist(data, bins=10, edgecolor='black')
        # filter for EMG by interpolated
        return data


class FeatureExtract:
    def __init__(self, signal, fs):
        self.signal = signal
        self.fs = fs
        self.features = None

    def dfa_features(self):
        # def dfa(x, scale_lim=[4, 8, 16, 32, 64, 128, 256], overlap=True):
        def dfa(x, scale_lim=[4, 8, 16, 32, 64], overlap=True):
            x = np.array(x)
            N = len(x)
            F = np.zeros(len(scale_lim))
            for i, scale in enumerate(scale_lim):
                segments = N - scale + 1 if overlap else N // scale
                stride = 1 if overlap else scale
                y = np.cumsum(x - np.mean(x))
                F[i] = 0
                for j in range(0, segments * stride, stride):
                    segment = y[j:j + scale]
                    t = np.arange(scale)
                    p = np.polyfit(t, segment, 1)
                    F[i] += np.sum((segment - np.polyval(p, t)) ** 2)
                F[i] = np.sqrt(F[i] / (segments * scale))
            p = np.polyfit(np.log(scale_lim), np.log(F), 1)
            return p[0]

        return {"dfa_exponent": dfa(self.signal)}

    def statistical_features(self):
        features = {
            'mean': np.mean(self.signal),
            'std': np.std(self.signal),
            'mean_abs_diff': np.mean(np.abs(np.diff(self.signal))),
            'mean_abs_second_diff': np.mean(np.abs(np.diff(self.signal, n=2))),
            'skewness': stats.skew(self.signal),
            'kurtosis': stats.kurtosis(self.signal)
        }
        return features

    def ar_features(self, order=4):
        model = AutoReg(self.signal, lags=order).fit()
        return {f'ar_coeff_{i + 1}': coeff for i, coeff in enumerate(model.params[1:])}

    def fractal_dimension(self):
        def higuchi_fd(X, kmax):
            N = len(X)
            L = []
            x = []
            for k in range(1, kmax + 1):
                Lk = []
                for m in range(k):
                    Lmk = sum(abs(X[m + i * k] - X[m + (i - 1) * k]) for i in range(1, int((N - m) / k)))
                    Lmk = (Lmk * (N - 1) / (((N - m) / k) * k)) / k
                    Lk.append(Lmk)
                L.append(np.log(np.mean(Lk)))
                x.append([np.log(1.0 / k), 1])
            (p, _, _, _) = np.linalg.lstsq(x, L, rcond=None)
            return p[0]

        return {'fractal_dimension': higuchi_fd(self.signal, 10)}

    def psd_features(self):
        f, psd = scipy.signal.welch(self.signal, fs=self.fs)
        iwmf = np.sum(f * psd) / np.sum(psd)
        iwbw = np.sqrt(np.sum((f - iwmf) ** 2 * psd) / np.sum(psd))

        def sef(f, psd, percent):
            total_power = np.sum(psd)
            target_power = total_power * percent / 100
            cumulative_power = np.cumsum(psd)
            return f[np.where(cumulative_power >= target_power)[0][0]]

        return {
            'iwmf': iwmf,
            'iwbw': iwbw,
            'sef_95': sef(f, psd, 95)
        }

    def band_power(self):
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }

        f, psd = scipy.signal.welch(self.signal, fs=self.fs)
        total_power = np.sum(psd)

        features = {}
        for band, (low, high) in bands.items():
            idx_band = np.logical_and(f >= low, f <= high)
            power = np.sum(psd[idx_band])
            features[f'{band}_power'] = power
            features[f'{band}_relative_power'] = power / total_power

        band_list = list(bands.keys())
        for i in range(len(band_list)):
            for j in range(i + 1, len(band_list)):
                band1, band2 = band_list[i], band_list[j]
                ratio_name = f'{band1[0]}{band2[0]}r'
                features[ratio_name] = features[f'{band1}_power'] / features[f'{band2}_power']

        return features

    def hilbert_transform_features(self):
        analytic_signal = scipy.signal.hilbert(self.signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.fs

        return {
            'hilbert_mean_amplitude': np.mean(amplitude_envelope),
            'hilbert_std_amplitude': np.std(amplitude_envelope),
            'hilbert_mean_inst_freq': np.mean(instantaneous_frequency),
            'hilbert_std_inst_freq': np.std(instantaneous_frequency),
            'hilbert_phase_coherence': np.abs(np.mean(np.exp(1j * instantaneous_phase)))
        }

    def hjorth_parameters(self):
        diff_first = np.diff(self.signal)
        diff_second = np.diff(diff_first)

        activity = np.var(self.signal)
        mobility = np.sqrt(np.var(diff_first) / np.var(self.signal))
        complexity = np.sqrt(np.var(diff_second) / np.var(diff_first)) / mobility

        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity
        }

    def cwt_features(self):
        scales = np.arange(1, 128)
        coefficients, _ = pywt.cwt(self.signal, scales, 'morl', sampling_period=1.0 / self.fs)

        prob = np.abs(coefficients) ** 2 / np.sum(np.abs(coefficients) ** 2)

        return {
            'cwt_mean': np.mean(np.abs(coefficients)),
            'cwt_std': np.std(np.abs(coefficients)),
            'cwt_energy': np.sum(np.abs(coefficients) ** 2),
            'cwt_entropy': -np.sum(prob * np.log2(prob + 1e-10))
        }

    def dwt_features(self, wavelet='db4', level=3):
        coeffs = pywt.wavedec(self.signal, wavelet, level=level)

        features = {}
        for i, coeff in enumerate(coeffs):
            level_name = f'dwt_level_{i}'
            prob = coeff ** 2 / np.sum(coeff ** 2)
            features.update({
                f'{level_name}_mean': np.mean(np.abs(coeff)),
                f'{level_name}_std': np.std(coeff),
                f'{level_name}_energy': np.sum(coeff ** 2),
                f'{level_name}_entropy': -np.sum(prob * np.log2(prob + 1e-10))
            })

        return features

    def zero_crossing_rate(self):
        return {"zero_crossing_rate": len(np.where(np.diff(np.signbit(self.signal)))[0]) / len(self.signal)}

    def root_mean_square(self):
        return {"rms": np.sqrt(np.mean(self.signal ** 2))}

    def energy(self):
        return {"energy": np.sum(self.signal ** 2)}

    def envelope(self):
        return {"envelope_mean": np.mean(np.abs(scipy.signal.hilbert(self.signal)))}

    def autocorrelation(self):
        result = np.correlate(self.signal, self.signal, mode='full')
        return {"autocorrelation_mean": np.mean(result[result.size // 2:])}

    def peak_analysis(self):
        peaks, _ = find_peaks(self.signal)
        return {
            "peak_count": len(peaks),
            "peak_to_peak": np.ptp(self.signal)
        }

    def spectral_entropy(self):
        _, psd = scipy.signal.welch(self.signal, self.fs, nperseg=64)
        psd_norm = psd / psd.sum()
        return {"spectral_entropy": -np.sum(psd_norm * np.log2(psd_norm + 1e-10))}

    def hurst_exponent(self):
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(self.signal[lag:], self.signal[:-lag]))) for lag in lags]
        return {"hurst_exponent": np.polyfit(np.log(lags), np.log(tau), 1)[0]}

    def approximate_entropy(self, m=2, r=0.2):
        def _phi(m):
            x = np.array([self.signal[i:i + m] for i in range(len(self.signal) - m + 1)])
            C = np.sum(np.abs(x[:, None] - x[None, :]).max(axis=2) <= r, axis=0) / (len(self.signal) - m + 1)
            return np.sum(np.log(C + 1e-10)) / (len(self.signal) - m + 1)

        r = r * np.std(self.signal)
        return {"approximate_entropy": _phi(m) - _phi(m + 1)}

    def FeatureExtract(self):
        feature_methods = [
            self.statistical_features,
            self.ar_features,
            self.dfa_features,
            self.psd_features,
            self.band_power,
            self.hilbert_transform_features,
            self.hjorth_parameters,
            self.fractal_dimension,
            self.cwt_features,
            self.dwt_features,
            self.zero_crossing_rate,
            self.root_mean_square,
            self.energy,
            self.envelope,
            self.autocorrelation,
            self.peak_analysis,
            self.spectral_entropy,
            self.hurst_exponent,
            self.approximate_entropy
        ]

        self.features = {}
        for method in feature_methods:
            self.features.update(method())

        self.features = np.array(list(self.features.values()))
        return self.features
