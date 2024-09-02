from pipelines.training_pipeline import training_pipeline
from steps.clean_data import clean_data

if __name__ == "__main__":
    # training = training_pipeline(file_name=r"data\dataC3.csv", time_steps=10, units=50, dropout_rate=0.2, epochs=100, batch_size=32)
    training_pipeline(file_name="data/dataC3.csv", time_steps=80, units=128, dropout_rate=0, epochs=1000, batch_size=32,
                      model_type='Transformer', use_tuning=False, encoding_type='positional') #model_type = LSTMModel, Transformer