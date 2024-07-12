from pipelines.training_pipeline import training_pipeline
from steps.clean_data import clean_data

if __name__ == "__main__":
    training = training_pipeline(file_name=r"data\dataC3.csv",
                                 time_steps=80,
                                 units=128,
                                 dropout_rate=0.5,
                                 epochs=1000,
                                 batch_size=512,
                                 patience=500)
