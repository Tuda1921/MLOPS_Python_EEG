from model.model import LSTMModel, TransformerModel
from steps.clean_data import clean_data
import optuna
from optuna.integration import TFKerasPruningCallback
import logging

def training_pipeline(file_name: str, time_steps: int, units, dropout_rate, epochs, batch_size, model_type='LSTM', use_tuning=True, encoding_type='positional'):
    try:
        X_train, Y_train, X_test, Y_test = clean_data(file_name, time_steps)

        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = 3  # Define your number of classes here

        if model_type == 'LSTM':
            model = LSTMModel(input_shape=input_shape, units=units, dropout_rate=dropout_rate, num_classes=num_classes)
            if use_tuning:
                study = model.optimize(X_train, Y_train, X_test, Y_test)
                best_params = study.best_trial.params
            else:
                best_params = {}  # Use default or preset hyperparameters
        elif model_type == 'Transformer':
            if use_tuning:
                # Create an instance of TransformerModel with initial parameters to call optimize method
                initial_transformer = TransformerModel(input_shape=input_shape, num_classes=num_classes, d_k=32, d_v=32,
                                                       n_heads=2, ff_dim=128, encoding_type=encoding_type)
                study = initial_transformer.optimize(X_train, Y_train, X_test, Y_test)
                best_params = study.best_trial.params
            else:
                best_params = {'d_k': 32, 'd_v': 32, 'n_heads': 2,
                               'ff_dim': 64}  # Use default or preset hyperparameters
            model = TransformerModel(input_shape=input_shape, num_classes=num_classes, encoding_type=encoding_type,
                                     **best_params)
        else:
            raise ValueError("Unsupported model type. Choose 'LSTM' or 'Transformer'.")

        logging.info(f"Best params: {best_params}")

        if model_type == 'LSTM':
            history = model.train(X_train=X_train, y_train=Y_train, X_test=X_test, y_test=Y_test, epochs=epochs, batch_size=batch_size)
        elif model_type == 'Transformer':
            history = model.train(X_train=X_train, y_train=Y_train, X_test=X_test, y_test=Y_test, epochs=epochs, batch_size=batch_size)

        model.save_model(f'save_model/{model_type}_model.h5')
        model.save_history(history, f'save_history/{model_type}_history.csv')

        model.plot(history)

        return model

    except Exception as e:
        logging.error(e)
        raise e