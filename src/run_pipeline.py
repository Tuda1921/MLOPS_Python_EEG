from pipelines.training_pipeline import training_pipeline
from steps.clean_data import clean_data

if __name__ == "__main__":
    training = training_pipeline(file_name=r"data\dataC3.csv",
                                 time_steps=80,  # recommend >= 80
                                 units=64,
                                 dropout_rate=0.5,
                                 epochs=10000,
                                 batch_size=4096,
                                 patience=1000,
                                 learning_rate=0.0001,
                                 fc_units=16)
