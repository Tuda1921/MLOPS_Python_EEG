from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, TimeDistributed
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import optuna
from optuna.integration import TFKerasPruningCallback
import os
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from Encoder.Position_encoding import PositionalEncoding
from Encoder.Time_vector import Time2Vector
from model.Transformer import SingleAttention, MultiAttention, TransformerEncoder
import tensorflow as tf
from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract base class for all models.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target
        """
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
class CustomLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        accuracy = logs.get('accuracy')
        val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_accuracy')

        # In ra kết quả với 8 chữ số sau dấu chấm thập phân
        print(f'Epoch {epoch + 1}: '
              f'loss: {loss:.8f} - accuracy: {accuracy:.8f} - '
              f'val_loss: {val_loss:.8f} - val_accuracy: {val_accuracy:.8f}')
class TransformerModel(Model):
    def __init__(self, input_shape, num_classes, d_k, d_v, n_heads, ff_dim, encoding_type='positional'):
        """
        Initializes the Transformer model for sequence classification.

        Args:
            input_shape (tuple): The shape of the input data, excluding the batch size.
            num_classes (int): Number of classes for classification.
            d_k (int): Dimension of the key vectors.
            d_v (int): Dimension of the value vectors.
            n_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward network.
        """
        super(TransformerModel, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.encoding_type = encoding_type

        # Build the model architecture
        self.build_model()

    def build_model(self):
        input_seq = Input(shape=self.input_shape)

        if self.encoding_type == 'positional':
            positional_encoding = PositionalEncoding(self.input_shape[0], self.input_shape[1])(input_seq)
            x = tf.keras.layers.Add()([input_seq, positional_encoding])
        elif self.encoding_type == 'time2vector':
            time2vector_encoding = Time2Vector(self.input_shape[0])(input_seq)
            x = tf.keras.layers.Add()([input_seq, time2vector_encoding])
        else:
            raise ValueError("Unsupported encoding type. Choose 'positional' or 'time2vector'.")

        attn_layer1 = TransformerEncoder(self.d_k, self.d_v, self.n_heads, self.ff_dim)
        attn_layer2 = TransformerEncoder(self.d_k, self.d_v, self.n_heads, self.ff_dim)

        x = attn_layer1((x, x, x))
        x = attn_layer2((x, x, x))

        x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        out = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = tf.keras.models.Model(inputs=input_seq, outputs=out)

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size, trial=None):
        opt = Adam(learning_rate=1e-2)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        custom_logger = CustomLoggingCallback()
        callbacks = [early_stopping]

        if trial is not None:
            callbacks.append(TFKerasPruningCallback(trial, monitor='val_loss'))

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)
        return history

    def optimize(self, X_train, y_train, X_test, y_test):
        """
        Optimizes hyperparameters of the Transformer model using Optuna.

        Args:
            X_train (numpy.ndarray): Training data input.
            y_train (numpy.ndarray): Training data labels.
            X_test (numpy.ndarray): Validation data input.
            y_test (numpy.ndarray): Validation data labels.

        Returns:
            Study object containing the results of hyperparameter optimization.
        """

        def objective(trial):
            self.d_k = trial.suggest_int('d_k', 2, 128)
            self.d_v = trial.suggest_int('d_v', 2, 128)
            self.n_heads = trial.suggest_int('n_heads', 1, 64)
            self.ff_dim = trial.suggest_int('ff_dim', 2, 512)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
            patience = trial.suggest_int('d_v', 5, 20)
            self.build_model()  # Rebuild model with new hyperparameters

            opt = Adam(learning_rate=learning_rate)
            self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            self.model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test),
                           callbacks=[early_stopping, TFKerasPruningCallback(trial, monitor='val_loss')], verbose=0)

            _, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=1000)

        return study

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

    def plot(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def save_model(self, filepath):
        dir_name = os.path.dirname(filepath)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.model.save(filepath)
        print(f'Model saved to {filepath}')

    def save_history(self, history, filepath):
        dir_name = os.path.dirname(filepath)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        history_df = pd.DataFrame({
            'epoch': range(1, len(history.history['accuracy']) + 1),
            'accuracy_train': history.history['accuracy'],
            'accuracy_test': history.history['val_accuracy'],
        })
        history_df.to_csv(filepath, index=False)
        print(f'History saved to {filepath}')

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
        self.input_shape = input_shape
        self.units = units
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units=units, activation='tanh', return_sequences=False), input_shape=input_shape))
        # self.model.add(Dense(units=32, activation='tanh'))
        # self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(units=num_classes, activation='softmax'))  # Softmax activation for multi-class classification
        opt = Adam(learning_rate=5e-5)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_test, y_test), callbacks=[early_stopping])
        return history

    def optimize(self, X_train, y_train, X_test, y_test):
        """
        Optimizes hyperparameters of the LSTM model using Optuna.

        Args:
            X_train (numpy.ndarray): Training data input.
            y_train (numpy.ndarray): Training data labels.
            X_test (numpy.ndarray): Validation data input.
            y_test (numpy.ndarray): Validation data labels.

        Returns:
            Study object containing the results of hyperparameter optimization.
        """

        def objective(trial):
            units = trial.suggest_int('units', 32, 256)
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

            model = Sequential()
            model.add(Bidirectional(LSTM(units=units, activation='tanh', return_sequences=False), input_shape=self.input_shape))
            model.add(Dense(units=32, activation='tanh'))
            model.add(Dropout(dropout_rate))
            model.add(Dense(units=self.num_classes, activation='softmax'))

            opt = Adam(learning_rate=learning_rate)  # Sử dụng tham số learning_rate thay vì lr
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                      callbacks=[early_stopping], verbose=0)

            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial), n_trials=10)

        return study

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