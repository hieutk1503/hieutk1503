# GD.py
import numpy as np

class GD:
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
        """
        Khởi tạo Gradient Descent.
        :param learning_rate: Tốc độ học
        :param max_iter: Số lần lặp tối đa
        :param tolerance: Ngưỡng hội tụ
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.weights = None
        self.history = []  # Lưu giá trị hàm chi phí qua từng vòng lặp
    
    def compute_cost(self, X, Y):
        """Tính toán hàm chi phí (phương thức cần ghi đè)."""
        raise NotImplementedError("Bạn cần định nghĩa hàm compute_cost trong lớp con.")
    
    def compute_gradient(self, X, Y):
        """Tính gradient (phương thức cần ghi đè)."""
        raise NotImplementedError("Bạn cần định nghĩa hàm compute_gradient trong lớp con.")
    
    def fit(self, X, Y):
        """
        Huấn luyện mô hình bằng Gradient Descent.
        :param X: Ma trận đặc trưng
        :param Y: Nhãn (output)
        """
        m, n = X.shape
        self.weights = np.zeros((n, 1))  # Khởi tạo weights với giá trị 0
        
        for i in range(self.max_iter):
            # Tính gradient
            gradient = self.compute_gradient(X, Y)
            
            # Cập nhật weights
            self.weights -= self.learning_rate * gradient
            
            # Tính giá trị hàm chi phí
            cost = self.compute_cost(X, Y)
            self.history.append(cost)
            
            # Kiểm tra hội tụ
            if i > 0 and abs(self.history[-1] - self.history[-2]) < self.tolerance:
                break
    
    def get_weights(self):
        """Lấy giá trị tham số weights."""
        return self.weights