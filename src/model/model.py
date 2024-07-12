from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, TimeDistributed
from keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import os

from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract base class for all models.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass

    @abstractmethod
    def plot(self, history):
        pass

    @abstractmethod
    def save_model(self, filepath):
        pass

    @abstractmethod
    def save_history(self, history, filepath):
        pass

class LSTMModel:
    def __init__(self, input_shape, units, dropout_rate, num_classes):
        """
        Initializes the LSTM model for sequence classification.

        Args:
            input_shape (tuple): The shape of the input data, excluding the batch size.
            units (int): The number of units in the LSTM layers.
            dropout_rate (float): The dropout rate for regularization.
            num_classes (int): Number of classes for classification.
        """
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units=units, activation='tanh', return_sequences=False), input_shape=input_shape))
        self.model.add(Dense(units=32, activation='tanh'))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(units=num_classes, activation='softmax'))  # Softmax activation for multi-class classification
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size, patience):
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_test, y_test), callbacks=[early_stopping])
        return history

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

    def plot(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def save_model(self, filepath):
        # Tạo thư mục nếu chưa tồn tại
        dir_name = os.path.dirname(filepath)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.model.save(filepath)
        print(f'Model saved to {filepath}')

    def save_history(self, history, filepath):
        # Tạo thư mục nếu chưa tồn tại
        dir_name = os.path.dirname(filepath)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Tạo một DataFrame từ lịch sử huấn luyện
        history_df = pd.DataFrame({
            'epoch': range(1, len(history.history['accuracy']) + 1),
            'accuracy_train': history.history['accuracy'],
            'accuracy_test': history.history['val_accuracy'],
        })
        # Lưu DataFrame vào tệp CSV
        history_df.to_csv(filepath, index=False)
        print(f'History saved to {filepath}')

class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning using Optuna.
    """

    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.X_train, self.y_train, self.X_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params
