from model.model import LSTMModel
from steps.clean_data import clean_data
import logging
def training_pipeline(file_name: str, time_steps: int, units, dropout_rate, epochs, batch_size, patience):
    try:
        # Sử dụng clean_data để xử lý dữ liệu và chia thành tập huấn luyện và tập kiểm thử
        X_train, Y_train, X_test, Y_test = clean_data(file_name, time_steps)
        # X_train, Y_train, X_test, Y_test = Feature_extract(file_name, time_steps)

        # Kiểm tra lại input_shape yêu cầu bởi mô hình LSTM
        input_shape = (656 - time_steps + 1, time_steps//2+1)  # (6768, 10)

        # Khởi tạo model LSTM với các thông số đã cho và input_dim tự động
        model = LSTMModel(input_shape=input_shape, units=units, dropout_rate=dropout_rate, num_classes=3)

        # Huấn luyện mô hình
        history = model.train(X_train=X_train, y_train=Y_train, X_test=X_test, y_test=Y_test, epochs=epochs, batch_size=batch_size, patience=patience)

        # Lưu mô hình
        model.save_model(r'save_model/LSTM_model.h5')
        model.save_history(history, r'save_history/LSTM_history.csv')

        # Vẽ biểu đồ kết quả của quá trình huấn luyện
        model.plot(history)

        return model

    except Exception as e:
        logging.error(e)
        raise e
