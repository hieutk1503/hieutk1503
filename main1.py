# main.py
import numpy as np
import pandas as pd
from Linear import LinearRegression  # Import lớp LinearRegression

# Đọc dữ liệu từ CSV
file_path = "D:\Python\Advertising.csv"  # Thay đường dẫn thực tế của bạn
data = pd.read_csv(file_path)
data.fillna(0, inplace=True)

# Chuẩn bị dữ liệu
X = data.iloc[:, 4:-1].values  # Các cột đặc trưng
Y = data.iloc[:, 3].values.reshape(-1, 1)  # Cột nhãn (output)

# Tạo và huấn luyện mô hình
lr = LinearRegression(learning_rate=0.01, max_iter=1000, tolerance=1e-6)
lr.fit(X, Y)

# Lấy trọng số và hiển thị kết quả
print("Trọng số w:", lr.get_weights())

# Dự đoán với dữ liệu mới
X_new = np.array([[70, 2000, 5], [50, 1980, 2]])  # Thay thế bằng dữ liệu thực tế
predictions = lr.predict(X_new)
print("Dự đoán:", predictions)